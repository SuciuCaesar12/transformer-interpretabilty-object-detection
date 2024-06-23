import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import coco.api as cocoAPI
import coco.detr as cocoDETR

import xai_detr.base as base
from xai_detr.explainer import DetrExplainer

from panopticapi.evaluation import pq_compute

import torch
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass


PATH_ROOT = Path(r'./coco/data')
PATH_ANNOTATIONS = PATH_ROOT / 'panoptic_annotations_trainval2017/annotations/'


explainer = DetrExplainer(
    model=cocoDETR.build(),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
coco_api = cocoAPI.CocoPanopticAPI(
    root=PATH_ROOT,
    annFile=PATH_ANNOTATIONS / 'panoptic_val2017.json',
    masksDir=PATH_ANNOTATIONS / 'panoptic_val2017'
)

@dataclass
class DetrSegmentationOutput:
    
    logits: torch.Tensor
    pred_boxes: torch.Tensor
    pred_masks: torch.Tensor


class CocoIDGenerator:
    
    def __init__(self):
        self.image_id = 0
    
    def __call__(self, instance_id: int, category_id: int):
       return int(self.image_id * 1e4 + category_id * 1e2 + instance_id)
   
    def update(self):
        self.image_id += 1


def rescale(x: torch.Tensor, min_val: float = 0., max_val: float = 1.):
    x = x - x.min()
    x = x / x.max()
    x = x * (max_val - min_val) + min_val
    return x


def to_mask(rel_map: torch.Tensor) -> torch.Tensor:
    h, w = rel_map.shape
    rel_map = torch.nn.functional.interpolate(
        rel_map.unsqueeze(0).unsqueeze(0), 
        size=(h * 8, w * 8), 
        mode='bilinear').squeeze()
    
    rel_map = rescale(rel_map, min_val=0., max_val=255.)
    mask = cv2.threshold(rel_map.numpy().astype('uint8'), 0., 255., cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return torch.tensor(mask).float()


def to_masks(rel_maps: torch.Tensor) -> torch.Tensor:
    masks = [to_mask(rel_map) for rel_map in rel_maps]
    if masks == []:
        return torch.tensor([])
    return torch.stack(masks, dim=0)


def get_masks_from_rms(explainer_output: base.DetrExplainerOutput):
    return to_masks(
        [exp.relevance_map for exp in explainer_output.explanations]
    )


def get_masks_from_cross_attentions(explainer_output: base.DetrExplainerOutput):
    cross_attention = explainer_output.outputs.cross_attentions[-1]
    h, w = explainer_output.outputs.conv_feature_shape
    q_idx = [exp.detection.query_index for exp in explainer_output.explanations]
    
    return to_masks(
        [cross_attention[:, q_i, :].mean(dim=0).reshape(h, w) for q_i in q_idx]
    )


def inference_panoptic(
    image_id: int,
    threshold: float = 0.9
):
    image = coco_api.loadImgs([image_id])[0]
    
    if image.mode == 'L':
        image = image.convert('RGB')
    
    explainer_output = explainer.explain(
        image=image,
        include_labels='all',
        output_dir=None,
        save_explanations=False,
        write_tensorboard=False,
        verbose=False,
        threshold=threshold
    )
    
    # compute masks from relevance maps
    pred_masks = get_masks_from_rms(explainer_output)
    # compute masks from cross-attention maps
    raw_attn_pred_masks = get_masks_from_cross_attentions(explainer_output)
    
    outputs = explainer_output.outputs
    h, w = image.size
    
    [outputs, raw_attn_outputs] = explainer.model.processor.post_process_panoptic_segmentation(
        outputs=DetrSegmentationOutput(
            logits=torch.stack([outputs.logits] * 2, dim=0),
            pred_boxes=torch.stack([outputs.pred_boxes] * 2, dim=0),
            pred_masks=torch.stack([pred_masks, raw_attn_pred_masks], dim=0)
        ),
        threshold=0.,
        mask_threshold=0.,
        overlap_mask_area_threshold=0.,
        target_sizes=[(w, h)] * 2
    )
    
    outputs['image_id'] = image_id
    raw_attn_outputs['image_id'] = image_id
    
    return outputs, raw_attn_outputs


def convert_to_coco_result(predictions, name: str, output_dir: Path):
    Path(output_dir / name).mkdir(parents=True, exist_ok=True)
    pba = tqdm(predictions, desc='Converting to COCO format', leave=True)
    id_generator = CocoIDGenerator()
    annotations = []
   
    for pred in pba:
        image_id = pred['image_id']
        annotation = {
            'image_id': image_id,
            'file_name': None,
            'segments_info': []
        }
        
        pan_format = np.zeros((pred['segmentation'].shape[0], pred['segmentation'].shape[1], 3), dtype=np.uint8)
        for instance_id, segm_info in enumerate(pred['segments_info']):
            segm_id = id_generator(instance_id, segm_info['label_id'])
            pan_segm_info = {
                'id': segm_id,
                'category_id': segm_info['label_id']
            }
            
            color = [segm_id % 256, segm_id // 256, segm_id // 256 // 256]
            mask = pred['segmentation'] == instance_id + 1
            pan_format[mask] = color
    
            annotation['segments_info'].append(pan_segm_info)
        
        file_name =  f'{image_id}.png'
        annotation['file_name'] = file_name
        Image.fromarray(pan_format).save(output_dir / name / file_name)
        
        annotations.append(annotation)
        id_generator.update()
    
    annotations = {'annotations': annotations}
    with open(output_dir / (name + '.json'), 'w') as f:
        json.dump(annotations, f)


def main():
    # pba = tqdm(val_coco.getImageIds(), desc='Inference', leave=True)
    pba = tqdm([397133], desc='Inference', leave=True)
    
    # Perform inference
    predictions, raw_attn_predictions = [], []
    for image_id in pba:
        out, ra_out = inference_panoptic(image_id, threshold=0.9)
        predictions.append(out)
        raw_attn_predictions.append(ra_out)
    
    for pred, name in zip([predictions, raw_attn_predictions], ['ours', 'raw_attn']):
        convert_to_coco_result(pred, name=name, output_dir=PATH_ANNOTATIONS)
        
        eval_dict = pq_compute(
            pred_folder=PATH_ANNOTATIONS / name,
            pred_json_file=PATH_ANNOTATIONS / f'{name}.json',
            gt_folder=PATH_ANNOTATIONS / 'panoptic_val2017',
            # gt_json_file=PATH_ANNOTATIONS / 'panoptic_val2017.json',
            gt_json_file=PATH_ANNOTATIONS / 'panoptic_val2017_subset.json',
        )
        
        with open(PATH_ANNOTATIONS / f'{name}_evaluation.json', 'w') as f:
            json.dump(eval_dict, f)
        

if __name__ == '__main__':
    main()

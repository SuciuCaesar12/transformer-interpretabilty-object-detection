import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import json
import cv2
import numpy as np
import xai_detr.base as base

from coco.api import CocoPanopticAPI
from coco.detr import build
from xai_detr.explainer import DetrExplainer

from tqdm import tqdm
from PIL import Image
from panopticapi.evaluation import pq_compute


class CocoIDGenerator:
    
    def __init__(self):
        self.image_id = 0
    
    def __call__(self, instance_id: int, category_id: int):
       return int(self.image_id * 1e4 + instance_id * 1e2 + category_id)
   
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


def get_masks(explainer_output: base.DetrExplainerOutput) -> torch.Tensor:
    relevance_maps = [exp.relevance_map for exp in explainer_output.explanations]
    pred_masks = [to_mask(rel_map) for rel_map in relevance_maps]
    
    if pred_masks == []:
        return torch.tensor([])
    return torch.stack(pred_masks, dim=0)


def inference_panoptic(
    image_id: int, 
    cocoAPI: CocoPanopticAPI,
    explainer: DetrExplainer,
    threshold: float = 0.9
):
    image = cocoAPI.loadImgs([image_id])[0]
    h, w = image.size
    
    explainer_output = explainer.explain(
        image=image,
        include_labels='all',
        output_dir=None,
        save_explanations=False,
        write_tensorboard=False,
        verbose=False,
        threshold=threshold
    )
    
    pan_outputs = explainer_output.outputs
    pan_outputs.pred_masks = get_masks(explainer_output)
    pan_outputs = pan_outputs.unsqueeze(0)
    
    outputs = explainer.model.processor.post_process_panoptic_segmentation(
        outputs=pan_outputs,
        threshold=0.,
        mask_threshold=0.,
        overlap_mask_area_threshold=0.,
        target_sizes=[(w, h)]
    )[0]
    
    outputs['image_id'] = image_id
    return outputs


def convert_to_coco_result(predictions, output_dir: Path):
    Path(output_dir / 'predictions').mkdir(parents=True, exist_ok=True)
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
        Image.fromarray(pan_format).save(output_dir / 'predictions' / file_name)
        
        annotations.append(annotation)
        id_generator.update()
    
    annotations = {'annotations': annotations}
    with open(output_dir / 'predictions.json', 'w') as f:
        json.dump(annotations, f)


def get_explainer():
    model = build()
    device = 'cuda'
    return DetrExplainer(model, device)


def main():
    # Set the paths
    PATH_ROOT = Path(r'')
    PATH_ANNOTATIONS = PATH_ROOT / 'panoptic_annotations_trainval2017/annotations/'
    
    # Load the COCO API
    val_coco = CocoPanopticAPI(
        root=PATH_ROOT,
        annFile=PATH_ANNOTATIONS / 'panoptic_val2017.json',
        masksDir=PATH_ANNOTATIONS / 'panoptic_val2017'
    )
    pba = tqdm(val_coco.getImageIds(), desc='Inference', leave=True)
    
    # Load the explainer
    explainer = get_explainer()
    
    # Perform inference
    predictions = []
    for image_id in pba:
        outputs = inference_panoptic(image_id, val_coco, explainer, threshold=0.9)
        predictions.append(outputs)
    
    # Convert the predictions to COCO format
    convert_to_coco_result(predictions, PATH_ANNOTATIONS)
    
    # Perform evaluation
    eval_dict = pq_compute(
        pred_folder=PATH_ANNOTATIONS / 'predictions',
        pred_json_file=PATH_ANNOTATIONS / 'predictions.json',
        gt_folder=PATH_ANNOTATIONS / 'panoptic_val2017',
        gt_json_file=PATH_ANNOTATIONS / 'panoptic_val2017.json',
    )
    
    # Save the evaluation results
    with open(PATH_ANNOTATIONS / 'evaluation.json', 'w') as f:
        json.dump(eval_dict, f)
        

if __name__ == '__main__':
    main()

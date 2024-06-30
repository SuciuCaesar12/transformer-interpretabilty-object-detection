import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import transformers as tr
import coco.api as cocoAPI
import coco.detr as cocoDETR
from xai_detr.explainer import DetrExplainer
from tqdm import tqdm


PATH_ROOT = Path(r'./coco/data/panoptic')
PATH_IMG = Path('val2017')   # relative to PATH_ROOT
PATH_ANNOTATIONS = PATH_ROOT / 'panoptic_annotations_trainval2017/annotations/'
PATH_SAVE_EXPS = Path(r'./coco/explanations')

detr = cocoDETR.build()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

explainer = DetrExplainer(
    model=detr,
    device=device
)

coco_api = cocoAPI.CocoPanopticAPI(
    root=PATH_ROOT,
    annFile=PATH_ANNOTATIONS / 'panoptic_val2017.json',
    imgDir=PATH_IMG,
    masksDir=PATH_ANNOTATIONS / 'panoptic_val2017'
)


def register_forward_hooks_backbone(model: tr.DetrForObjectDetection):
    resnet50 = model.model.backbone.conv_encoder.model
    conv_hooks, conv_features = [], []
    
    conv_hooks.append(
        resnet50.layer1.register_forward_hook(
            lambda m, i, o: conv_features.append(o)
        )
    )
    
    conv_hooks.append(
        resnet50.layer2.register_forward_hook(
            lambda m, i, o: conv_features.append(o)
        )
    )
    
    conv_hooks.append(
        resnet50.layer3.register_forward_hook(
            lambda m, i, o: conv_features.append(o)
        )
    )
    
    conv_hooks.append(
        resnet50.layer4.register_forward_hook(
            lambda m, i, o: conv_features.append(o)
        )
    )
    
    return conv_hooks, conv_features


def generate_exp_coco_dataset(coco_api: cocoAPI.CocoPanopticAPI, explainer: DetrExplainer, save_dir: Path):
    pba = tqdm(coco_api.getImageIds()[:5], desc='Generating explanations')
    conv_hooks, conv_features = register_forward_hooks_backbone(detr.model)
    
    for img_id in pba:
        img, ann = coco_api[img_id]
        
        exp_out = explainer.explain(image=img, threshold=0.8)
        
        setattr(exp_out, 'conv_features', list(reversed(conv_features)))
        conv_features.clear()
        setattr(exp_out, 'image_id', ann['image_id'])
        
        exp_out.save(save_dir / f'exp_{ann["image_id"]}.pkl')

    for hook in conv_hooks:
        hook.remove()
    

if __name__ == '__main__':
    generate_exp_coco_dataset(coco_api, explainer, PATH_SAVE_EXPS)

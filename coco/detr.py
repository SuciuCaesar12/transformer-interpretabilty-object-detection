from PIL import Image
from typing import Any, Dict, Tuple, List

import torch
import transformers as tr
import xai_detr.base as base

class CocoDetrModule(base.AbstractDetrModule):
    
    def __init__(
        self, 
        model: tr.DetrForObjectDetection,
        processor: tr.DetrImageProcessor,
    ):
        super().__init__()
        self.model = model
        self.processor = processor


    def preprocess(self, image: Image.Image, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        encoding = self.processor(
            images=image, 
            annotations={'image_id': 0, 'annotations': []},
            return_tensors="pt")

        pixel_values = encoding["pixel_values"]
        
        encoding = self.processor.pad(list(pixel_values), return_tensors="pt")
        
        inputs = {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
        }
        
        return inputs
    
    
    def predict(self, inputs: Dict[str, torch.Tensor], **kwargs) -> base.DetrOutput:
        conv_features = []
        conv_hook = self.model.model.backbone.register_forward_hook(
            lambda m, i, o: conv_features.append(o))
        
        outputs = self.model(**inputs, output_attentions=True)
        
        [(conv_feature_maps, _), ] = conv_features
        conv_feature, _ = conv_feature_maps[-1]
        h, w = conv_feature.squeeze(0).shape[1:]
        conv_hook.remove()
        
        return base.DetrOutput(
            logits=outputs['logits'],
            pred_boxes=outputs['pred_boxes'],
            encoder_attentions=outputs['encoder_attentions'],
            decoder_attentions=outputs['decoder_attentions'],
            cross_attentions=outputs['cross_attentions'],
            conv_feature_shape=(h, w)
        )
    
    
    def id2label(self) -> Dict[int, str]:
        return self.model.config.id2label
       
        
    def no_object_id(self) -> int:
        return len(self.id2label())


    def num_queries(self) -> int:
        return self.model.config.num_queries
    
    
    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self


    def requires_grad(self, requires_grad: bool):
        self.model.requires_grad_(requires_grad)


    def freeze_backbone(self):
        self.model.model.freeze_backbone()
        
        
    def _zero_grad(self):
        self.model.zero_grad()
    

def build() -> CocoDetrModule:
    model = tr.DetrForSegmentation.from_pretrained(
        "facebook/detr-resnet-50-panoptic",
        ignore_mismatched_sizes=True).detr

    # Load the feature extractor for pre and post processing
    processor = tr.DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")  
    
    return CocoDetrModule(model, processor)
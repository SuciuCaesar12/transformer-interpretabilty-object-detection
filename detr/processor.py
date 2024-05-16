from PIL import Image
from dataclasses import fields

import torch
import transformers as tr


def convert_to_detections(decoded_outputs, categories):
    detections = list(zip(decoded_outputs['scores'].tolist(), decoded_outputs['labels'].tolist(), decoded_outputs['boxes'].tolist()))

    if len(detections) > 0:
        categories = [categories[d[1]] for d in detections]
        scores, _, boxes = list(zip(*detections))
        detections = list(zip(scores, categories, boxes))
    return detections


class DetrProcessor:
    
    def __init__(self, processor: tr.DetrImageProcessor):
        self.processor = processor
    
    def __call__(self, image: Image.Image):
        encoding = self.processor(
            images=image, 
            annotations={'image_id': 0, 'annotations': []},
            return_tensors="pt")

        pixel_values = encoding["pixel_values"]
        labels = encoding["labels"]
        
        encoding = self.processor.pad(list(pixel_values), return_tensors="pt")
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
    
    def post_process_object_detection(self, **kwargs):
        return self.processor.post_process_object_detection(**kwargs)


class DetrPostProcessor:
    
    def __init__(self, processor: tr.DetrImageProcessor):
        self.processor = processor
    
    
    def _to_cpu(self, outputs):
        for field in fields(outputs):
            value = getattr(outputs, field.name)
            
            if value is not None and isinstance(value, torch.Tensor):
                setattr(outputs, field.name, value.detach().cpu())
            elif isinstance(value, tuple):
                setattr(outputs, field.name, tuple(v.detach().cpu() for v in value if isinstance(v, torch.Tensor)))
        
        return outputs
    
    def _squeeze(self, outputs):
        for field in fields(outputs):
            value = getattr(outputs, field.name)
            
            if value is not None and isinstance(value, torch.Tensor):
                setattr(outputs, field.name, value.squeeze(0))
            elif isinstance(value, tuple):
                setattr(outputs, field.name, tuple(v.squeeze(0) for v in value if isinstance(v, torch.Tensor)))
        
        return outputs
    
    
    def _decode_outputs(self, outputs, target_sizes, categories):
        decoded_outputs = self.processor.post_process_object_detection(
            outputs=outputs, 
            threshold=0.0, 
            target_sizes=target_sizes)[0]

        detections = convert_to_detections(decoded_outputs, categories)
        
        return detections
    
    
    def __call__(self, outputs, target_sizes, categories):
        outputs = self._to_cpu(outputs)
        detections = self._decode_outputs(outputs, target_sizes, categories)
        outputs = self._squeeze(outputs)
        
        return {
            'outputs': outputs,
            'detections': detections
        }
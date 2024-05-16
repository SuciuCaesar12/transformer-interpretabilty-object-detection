from detr.processor import DetrProcessor, DetrPostProcessor
from detr.attention import DetrAttentionModuleExplainer
from detr.writer import TensorboardWriter
from typing import Dict, List
from pathlib import Path
from PIL import  Image

import gc
import torch
import transformers as tr


def filter_by_thresh(outputs, no_object_id: int, threshold: float = 0.5):
    is_keep = lambda v, i: v > threshold and i != no_object_id
    logits = outputs['logits'].squeeze(0)
    device = logits.device
    
    vs, idx = logits.softmax(-1).max(-1)
    q_idx = torch.tensor([is_keep(v, i) for v, i in zip(vs, idx)]).nonzero()
    q_idx = q_idx.squeeze(-1).to(device)
    
    return q_idx


def sort_by_area(outputs):
    ws = outputs['pred_boxes'][0, :, 2]
    hs = outputs['pred_boxes'][0, :, 3]
    return (ws * hs).sort(descending=True)[1]


class DetrExplainer:
    
    def __init__(self, model: tr.DetrForObjectDetection,
                 processor: tr.DetrImageProcessor, 
                 device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.requires_grad_(True)
        self.model.model.freeze_backbone()
        
        self.processor = DetrProcessor(processor)
        self.post_processor = DetrPostProcessor(processor)
        self.attn_module_explainer = DetrAttentionModuleExplainer(model, self.device)
    
    
    def _inference(self, inputs: Dict[str, torch.Tensor], no_object_id: int, threshold: float):
        conv_features = []
        conv_hook = self.model.model.backbone.register_forward_hook(
            lambda m, i, o: conv_features.append(o))
        
        # perform inference
        outputs = self.model(
            pixel_values=inputs['pixel_values'],
            pixel_mask=inputs['pixel_mask'], 
            output_attentions=True, 
            output_hidden_states=True, 
            return_dict=True)
        
        # filter detections by confidence threshold
        raw_logits = outputs['logits'].clone().squeeze(0)
        q_idx = filter_by_thresh(outputs, no_object_id, threshold)
        outputs['logits'] = outputs['logits'][:, q_idx, :]
        outputs['pred_boxes'] = outputs['pred_boxes'][:, q_idx, :]
        
        # sort detections by area
        q_idx_temp = sort_by_area(outputs)
        q_idx = q_idx[q_idx_temp]
        outputs['logits'] = outputs['logits'][:, q_idx_temp, :]
        outputs['pred_boxes'] = outputs['pred_boxes'][:, q_idx_temp, :]
        
        # extract the last convolutional feature map
        # used for reshaping the relevance maps
        [(conv_feature_maps, _), ] = conv_features
        conv_feature, _ = conv_feature_maps[-1]
        conv_feature = conv_feature.squeeze(0)
        conv_hook.remove()
        
        return q_idx[:3], outputs, conv_feature, raw_logits
        

    def explain(self,
                image: Image.Image, 
                no_object_id: int, 
                categories: List[str],
                output_dir: Path = None,
                threshold: float = 0.5):
        # clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # transform image in the format Detr uses
        inputs = self.processor(image)
        
        # move inputs and model to device
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        inputs['pixel_mask'] = inputs['pixel_mask'].to(self.device)
        self.model = self.model.to(self.device)
        
        # create tensorboard writer
        if output_dir is not None:  
            writer = TensorboardWriter(output_dir)
        
        # perform inference + filter detections
        q_idx, outputs, conv_feature, orig_logits = self._inference(
            inputs, no_object_id, threshold)
        
        # generate relevance maps for each detection
        explanations = self.attn_module_explainer.generate_rel_maps(
            q_idx, 
            orig_logits, 
            outputs['encoder_attentions'], 
            outputs['decoder_attentions'], 
            outputs['cross_attentions'])
        
        # reshape relevance maps
        h, w = conv_feature.shape[1:]
        for exp in explanations:
            exp['relevance_map'] = exp['relevance_map'].reshape(h, w)

        # move inputs to cpu to save memory
        inputs['pixel_values'] = inputs['pixel_values'].to('cpu')
        inputs['pixel_mask'] = inputs['pixel_mask'].to('cpu')
        
        # decode outputs
        decoded_outputs = self.post_processor(
            outputs=outputs,
            target_sizes=[inputs['labels'][0]['orig_size']],
            categories=categories
        )
        exp_img = {
            'explanations': list(zip(explanations, decoded_outputs['detections'])),
            'outputs': decoded_outputs['outputs'],
        }
        
        # write to tensorboard the explanations for each detection
        if output_dir is not None:
            writer(image, exp_img)
        
        return exp_img

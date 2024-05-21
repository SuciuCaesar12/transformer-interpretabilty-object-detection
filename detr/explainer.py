from detr.processor import DetrProcessor
from detr.attention import DetrAttentionModuleExplainer
from detr.writer import TensorboardWriter
from typing import Dict, List, Union
from pathlib import Path
from PIL import  Image

import gc
import torch
import transformers as tr


class DetrExplainer:
    
    
    def __init__(self, 
                 model: tr.DetrForObjectDetection,
                 processor: tr.DetrImageProcessor, 
                 id2label: List[Dict[int, str]],
                 no_object_id: int,
                 device: str = 'cpu'):
        self.model = model
        self.model.requires_grad_(True)
        self.model.model.freeze_backbone()
        self.id2label = id2label
        self.no_object_id = no_object_id
        self.device = device
        
        self.processor = DetrProcessor(processor, self.id2label)
        self.attn_module_explainer = DetrAttentionModuleExplainer(model, self.device)
        
    
    def _inference(self, inputs: Dict[str, torch.Tensor]):
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
        
        self.raw_logits = outputs['logits'].squeeze(0).clone()
        scores, label_ids = self.raw_logits.softmax(-1).max(-1)
        include_label_ids = self.include_label_ids.unsqueeze(0).repeat(label_ids.shape[0], 1)
    
        q_idx =\
            (label_ids != self.no_object_id) &\
            (scores > self.threshold) &\
            torch.any(label_ids.unsqueeze(-1) == include_label_ids, dim=-1)
        q_idx = q_idx.nonzero().squeeze(-1)
        
        outputs['logits'] = outputs['logits'][:, q_idx, :]
        outputs['pred_boxes'] = outputs['pred_boxes'][:, q_idx, :]

        ws = outputs['pred_boxes'][0, :, 2]
        hs = outputs['pred_boxes'][0, :, 3]
        q_idx_temp = (ws * hs).sort(descending=True)[1]
        
        q_idx = q_idx[q_idx_temp]
        outputs['logits'] = outputs['logits'][:, q_idx_temp, :]
        outputs['pred_boxes'] = outputs['pred_boxes'][:, q_idx_temp, :]
        
        self.q_idx = q_idx
        self.outputs = outputs
        
        # extract the last convolutional feature map
        # used for reshaping the relevance maps
        [(conv_feature_maps, _), ] = conv_features
        conv_feature, _ = conv_feature_maps[-1]
        self.conv_feature = conv_feature.squeeze(0)
        conv_hook.remove()
        

    def explain(self,
                image: Image.Image, 
                include_labels: Union[List[str], str] = 'all', 
                output_dir: Path = None,
                threshold: float = 0.5):
        # clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        if isinstance(include_labels, str):
            if include_labels == 'all':
                include_labels = list(self.id2label.values())
            else:
                raise ValueError('include_labels must be a list of strings or has the value "all"')
        
        assert all([c in self.id2label.values() for c in include_labels]),\
            'All names in include_labels must be in id2label dictionary'
        
        self.threshold = threshold
        self.include_label_ids = torch.Tensor([
            id for id, label in self.id2label.items() if label in include_labels
        ]).to(self.device)
        
        # transform image in the format Detr uses
        inputs = self.processor.preprocess(image)
        
        # move inputs and model to device
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        inputs['pixel_mask'] = inputs['pixel_mask'].to(self.device)
        self.model = self.model.to(self.device)
        
        # perform inference + filter detections
        self._inference(inputs)
        
        # generate relevance maps for each detection
        explanations = self.attn_module_explainer.generate_rel_maps(
            q_idx=self.q_idx, 
            logits=self.raw_logits, 
            encoder_attentions=self.outputs['encoder_attentions'], 
            decoder_attentions=self.outputs['decoder_attentions'], 
            cross_attentions=self.outputs['cross_attentions'])
        
        # reshape relevance maps
        h, w = self.conv_feature.shape[1:]
        for exp in explanations:
            exp['relevance_map'] = exp['relevance_map'].reshape(h, w)

        # move inputs to cpu to save memory
        inputs['pixel_values'] = inputs['pixel_values'].to('cpu')
        inputs['pixel_mask'] = inputs['pixel_mask'].to('cpu')
        
        # decode outputs
        decoded_outputs = self.processor.postprocess(
            outputs=self.outputs,
            target_sizes=[inputs['labels'][0]['orig_size']]
        )
        
        exp_img = {
            'explanations': [{
                'explanation': e,
                'detection': d,
            } for e, d in zip(explanations, decoded_outputs['detections'])],
            'outputs': decoded_outputs['outputs'],
        }
        
        # write to tensorboard the explanations for each detection
        if output_dir is not None:
            TensorboardWriter(output_dir)(image, exp_img)
        
        return exp_img

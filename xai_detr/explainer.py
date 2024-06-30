import copy
from xai_detr.attention import DetrAttentionModuleExplainer
from xai_detr.writer import TensorboardWriter
from typing import Dict, List, Union, Any
from pathlib import Path
from PIL import  Image
from tqdm import tqdm

import gc
import torch
import transformers as tr
import xai_detr.base as base


def move_to_device(inputs: Dict[str, Any], device: str):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = inputs[k].to(device)
        elif isinstance(v, list):
            inputs[k] = [move_to_device(x, device) for x in inputs[k]]
        elif isinstance(v, tuple):
            inputs[k] = tuple([move_to_device(x, device) for x in inputs[k]])
        elif isinstance(v, dict):
            inputs[k] = move_to_device(inputs[k], device)
    return inputs


class DetrExplainer:
    '''
    Explainer for the DETR model.
    '''
    
    
    def __init__(self, 
                 model: base.AbstractDetrModule,
                 device: str = 'cpu'):
        '''
        Attributes:
        ----------
        
        model: detr.DetrModule
            DETR model to explain. It should inherit from the DetrModule class and implement the abstract methods.
        
        device: str
            Device to use for inference.
        '''
        self.model: base.AbstractDetrModule = model
        self.model.requires_grad(True)
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
        self.device = device
        
        self.processor = tr.DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")  
        self.attn_module_explainer = DetrAttentionModuleExplainer(model, self.device)
        
    
    def _inference(self, inputs: Dict[str, Any]):
        '''
        Perform inference and filter detections.
        '''
        outputs = self.model.predict(inputs)
        
        self.raw_logits = outputs.logits.squeeze(0).clone()
        scores, label_ids = self.raw_logits.softmax(-1).max(-1)
        include_label_ids = self.include_label_ids.unsqueeze(0).repeat(label_ids.shape[0], 1)
    
        q_idx =\
            (label_ids != self.model.no_object_id()) &\
            (scores > self.threshold) &\
            torch.any(label_ids.unsqueeze(-1) == include_label_ids, dim=-1)
        q_idx = q_idx.nonzero().squeeze(-1)
        
        outputs.logits = outputs.logits[:, q_idx, :]
        outputs.pred_boxes = outputs.pred_boxes[:, q_idx, :]

        # sort detections by area
        ws = outputs.pred_boxes[0, :, 2]
        hs = outputs.pred_boxes[0, :, 3]
        q_idx_temp = (ws * hs).sort(descending=True)[1]
        
        q_idx = q_idx[q_idx_temp]
        outputs.logits = outputs.logits[:, q_idx_temp, :]
        outputs.pred_boxes = outputs.pred_boxes[:, q_idx_temp, :]
        
        self.q_idx = q_idx[:2]
        self.outputs: base.DetrOutput = outputs
    
    def _postprocess(self):
        decoded_outputs = self.processor.post_process_object_detection(
            outputs=self.outputs, 
            threshold=0.0, 
            target_sizes=[self.original_size[::-1]])[0]
        
        self.outputs = self.outputs.squeeze(0)

        detections = list(zip(
            decoded_outputs['scores'].tolist(), 
            decoded_outputs['labels'].tolist(), 
            decoded_outputs['boxes'].tolist()
        ))

        if len(detections) > 0:
            labels = [self.model.id2label()[d[1]] for d in detections]
            scores, _, boxes = list(zip(*detections))
            
            detection_items: List[base.DetectionItem] = []
            for s, l, b in zip(scores, labels, boxes):
                detection_items.append(base.DetectionItem(
                    score=s, label=l, box=b
                ))
        else:
            detection_items = []
        
        return detection_items
    
    def _write_on_tensorboard(self, image: Image.Image, writer: TensorboardWriter, exp_out: base.DetrExplainerOutput):
        '''
        Write explanations to tensorboard.
        
        Parameters:
        ----------
        
        writer: TensorboardWriter
            Writer to use to write to tensorboard.
            
        exp_out: detr.DetrExplainerOutput
            Output of the explainer.
        '''
        _, w = exp_out.outputs.conv_feature_shape
        pba = tqdm(exp_out.explanations, desc='Writing to tensorboard', leave=False, disable=not self.verbose)
        
        for e in pba:
            pba.set_postfix_str("Detection + Relevance Map")
            writer.write_detection_and_relevance_map(
                image=image,
                explanation=e
            )
            
            pba.set_postfix_str(f'Query {e.detection.query_index} - Cross Attention Maps')
            writer.write_query_cross_attention_maps(
                explainer_output=exp_out,
                query_index=e.detection.query_index,
                prefix_tag=f"query_{e.detection.query_index}_{e.detection.label}/"
            )
            
            flat_relevance_map = e.relevance_map.flatten()
            token_indices = (flat_relevance_map > 0.9).nonzero().squeeze(-1)
            
            if len(token_indices) == 0:
                token_indices = torch.argsort(flat_relevance_map, descending=True)[:1]
            
            for t_idx in token_indices:
                pba.set_postfix_str(f"Token ({t_idx // w}, {t_idx % w}) - Encoder Attention Maps")
                writer.write_image_token_attention_maps(
                    explainer_output=exp_out,
                    token_index=t_idx,
                    prefix_tag=f"query_{e.detection.query_index}_{e.detection.label}/"
                )

        writer.close()

    def explain(self,
                image: Image.Image, 
                include_labels: Union[List[str], str] = 'all', 
                output_dir: Path = None,
                save_image: bool = False,
                save_explanations: bool= False,
                write_tensorboard: bool = False,
                verbose: bool = False,
                threshold: float = 0.5) -> base.DetrExplainerOutput:
        '''
        Explains the predicted detections of the model on the image. It generates relevance maps for each detection.
        The explanations are saved in the output_dir/explanations.pkl file.
        The explanations are also written to tensorboard in output_dir.
        
        Parameters:
        ----------
        
        image: PIL.Image.Image
            Image to explain.
            
        include_labels: Union[List[str], str]
            Labels to include in the explanations. If 'all', include all labels.
        
        output_dir: Path
            Directory to save the explanations.
            
        save_image: bool
            Whether to save the image with the explanations to disk.
        
        save_explanations: bool
            Whether to save the explanations to disk.
        
        write_tensorboard: bool
            Whether to write the explanations to tensorboard.
        
        threshold: float
            Threshold to filter detections based on confidence.
        '''
        # clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        if isinstance(include_labels, str):
            if include_labels == 'all':
                include_labels = list(self.model.id2label().values())
            else:
                raise ValueError('include_labels must be a list of strings or has the value "all"')
        
        assert all([c in self.model.id2label().values() for c in include_labels]),\
            'All names in include_labels must be in id2label dictionary'
        
        if save_explanations or write_tensorboard:
            assert output_dir is not None, 'output_dir must be provided if save_explanations or write_tensorboard is True'
        
        self.verbose = verbose
        self.threshold = threshold
        self.original_size = image.size
        self.include_label_ids = torch.Tensor([
            id for id, label in self.model.id2label().items() if label in include_labels
        ]).to(self.device)
        
        # transform image in the format Detr uses
        inputs = self.model.preprocess(image)
        
        # move inputs and model to device
        inputs = move_to_device(inputs, self.device)
        self.model = self.model.to(self.device)
        
        # perform inference + filter detections
        self._inference(inputs)
        
        # generate relevance maps for each detection
        explanations = self.attn_module_explainer.generate_rel_maps(
            q_idx=self.q_idx, 
            logits=self.raw_logits, 
            outputs=self.outputs
        )

        # move inputs to cpu to save memory
        inputs = move_to_device(inputs, 'cpu')
        self.outputs = self.outputs.detach().to('cpu')
        
        # decode outputs
        detections = self._postprocess()
        
        for q_i, e, d in zip(self.q_idx, explanations, detections):
            d.query_index = q_i
            e.detection = d
        
        explainer_output = base.DetrExplainerOutput(
            image=copy.deepcopy(image) if save_image else None,
            explanations=explanations,
            outputs=self.outputs
        )
        
        # write to tensorboard the explanations for each detection
        if write_tensorboard:
            self._write_on_tensorboard(
                image, 
                TensorboardWriter(output_dir), 
                explainer_output
            )
        
        # save explanations to disk in a pickle file
        if save_explanations:
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            
            explainer_output.save(output_dir / 'explanations.pkl')
        
        return explainer_output

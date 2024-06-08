from tqdm import tqdm
from typing import List

import copy
import torch
import torch.nn.functional as F
import xai_detr.base as base


class DetrAttentionModuleExplainer:
    '''
    Attention module explainer for the DETR model.
    '''
    
    def __init__(self, model: base.AbstractDetrModule, device: str = 'cpu'):
        '''
        Attributes:
        ----------
        
        model: detr.DetrModule
            DETR model to explain. It should inherit from the DetrModule class and implement the abstract methods.
        
        device: str
            Device to use for inference.
        '''
        self.model = model
        self.n_q_t: int = self.model.num_queries()  # number of query tokens
        self.n_i_t: int = None  # number of image tokens
        self.device = device
    
    
    def _init_snapshots(self):
        '''
        Initialize snapshots for the relevance maps.
        '''
        self.R_i_i = torch.eye(self.n_i_t).to(self.device)
        self.R_q_q = torch.eye(self.n_q_t).to(self.device)
        self.R_q_i = torch.zeros(self.n_q_t, self.n_i_t).to(self.device)
        
        self.snapshots: List[base.SnapshotItem] = []
        self._add_snapshot(tag="init_state")
    
    def _add_snapshot(self, tag: str):
        '''
        Add a snapshot of the relevance maps.
        
        Parameters:
        ----------
        
        tag: str
            Tag for the snapshot.
        '''
        self.snapshots.append(base.SnapshotItem(
            tag=tag,
            R_i_i=self.R_i_i.clone().to('cpu'),
            R_q_q=self.R_q_q.clone().to('cpu'),
            R_q_i=self.R_q_i.clone().to('cpu')
        ))
    
    
    def _avg_heads(self, grad: torch.Tensor, attn_map: torch.Tensor):
        return (grad * attn_map).clamp(min=0).mean(dim=1)


    def _norm_rel_map(self, rel_map: torch.Tensor):
        h, w = rel_map.shape
        eye = torch.eye(h, w).to(self.device)
        return F.normalize(rel_map, p=1, dim=0) + eye
        
        
    def _self_attn_encoder_rel_update(self, attn_map: torch.Tensor):
        grad, attn_map = attn_map.grad.detach(), attn_map.detach()
        cam = self._avg_heads(grad, attn_map).reshape(self.n_i_t, self.n_i_t)
        self.R_i_i += cam @ self.R_i_i
    
    
    def _self_attn_decoder_rel_update(self, attn_map: torch.Tensor):
        grad, attn_map = attn_map.grad.detach(), attn_map.detach()
        cam = self._avg_heads(grad, attn_map).reshape(self.n_q_t, self.n_q_t)
        self.R_q_q += cam @ self.R_q_q
        self.R_q_i += cam @ self.R_q_i
    
    
    def _cross_attn_decoder_rel_update(self, attn_map: torch.Tensor):
        grad, attn_map = attn_map.grad.detach(), attn_map.detach()
        cam = self._avg_heads(grad, attn_map).reshape(self.n_q_t, self.n_i_t)
        norm_R_q_q = self._norm_rel_map(self.R_q_q)
        norm_R_i_i = self._norm_rel_map(self.R_i_i)
        self.R_q_i += norm_R_q_q.T @ cam @ norm_R_i_i
    
    
    def _backward_step(self, q_i: int, logits: torch.Tensor):
        '''
        Perform a backward step for the query token q_i.
        It calculates the gradients for maximizing the likelihood of the predicted class.
        
        Parameters:
        ----------
        
        q_i: int
            Index of the query token.
        
        logits: torch.Tensor
            Logits for all detection queries.
        '''
        one_hot = torch.zeros_like(logits[q_i], dtype=torch.float)
        one_hot[logits[q_i].argmax().item()] = 1.
        one_hot.requires_grad = True
        
        loss = (logits[q_i] * one_hot).sum()
        loss.backward(retain_graph=True)
    
    
    def generate_rel_maps(self, 
                          q_idx: torch.Tensor, 
                          logits: torch.Tensor, 
                          outputs: base.DetrOutput,
                          verbose: bool = False) -> List[base.ExplanationItem]:
        '''
        Generate relevance maps for the queries.
        
        Parameters:
        ----------
        
        q_idx: torch.Tensor
            Indices of the queries to explain.
        
        logits: torch.Tensor
            Logits for all detection queries.
        
        outputs: detr.DetrOutput
            Output of the DETR model.
        
        Returns:
        -------
        
        explanations: List[detr.ExplanationItem]
            List of explanations for each query.
        '''
        # Set attention maps to require grad
        for attn in outputs.encoder_attentions + outputs.decoder_attentions + outputs.cross_attentions: 
            attn.requires_grad_(True)
            attn.retain_grad()
        
        self.n_i_t = outputs.encoder_attentions[0].shape[2]  # (batch_size, num_heads, n_img_tokens, n_img_tokens)
        h, w = outputs.conv_feature_shape
        logits.requires_grad_(True)
        
        pba = tqdm(q_idx, desc="Detection", leave=False, disable=not verbose)
        explanations: List[base.ExplanationItem] = []
        
        for step, q_i in enumerate(q_idx):
            prefix = f"Detection {step + 1} / {len(q_idx)}"
            self.model.zero_grad()
            self._init_snapshots()
            q_i = q_i.item()
            
            # calculate gradients
            pba.set_description(prefix + " - Backward pass")
            self._backward_step(q_i, logits)
            
            # update relevance maps for each encoder block
            pba.set_description(prefix + " - Encoder pass")
            for enc_step, enc_attn in enumerate(outputs.encoder_attentions): 
                self._self_attn_encoder_rel_update(enc_attn)
                self._add_snapshot(tag=f"encoder_{enc_step}")
            
            # update relevance maps for each decoder block
            pba.set_description(prefix + " Decoder pass")
            for dec_step, (dec_attn, cross_attn) in enumerate(zip(outputs.decoder_attentions, outputs.cross_attentions)):
                self._self_attn_decoder_rel_update(dec_attn)
                self._cross_attn_decoder_rel_update(cross_attn)
                self._add_snapshot(tag=f"decoder_{dec_step}")
            
            explanations.append(base.ExplanationItem(
                snapshots=copy.deepcopy(self.snapshots),
                relevance_map=self.R_q_i[q_i].to('cpu').reshape(h, w)
            ))
        
        self.model.zero_grad()
        
        return explanations  


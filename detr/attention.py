from tqdm import tqdm
from typing import List

import copy
import torch
import torch.nn.functional as F
import transformers as tr


class DetrAttentionModuleExplainer:
    
    def __init__(self, model: tr.DetrForObjectDetection, device: str = 'cpu'):
        self.model = model
        self.n_q_t: int = self.model.config.num_queries  # number of query tokens
        self.n_i_t: int = None  # number of image tokens
        self.device = device
    
    
    def _init_snapshots(self):
        self.R_i_i = torch.eye(self.n_i_t).to(self.device)
        self.R_q_q = torch.eye(self.n_q_t).to(self.device)
        self.R_q_i = torch.zeros(self.n_q_t, self.n_i_t).to(self.device)
        
        self.snapshots = []
        self._add_snapshot(tag="init_state")
    
    def _add_snapshot(self, tag: str):
        self.snapshots.append({
            'tag': tag,
            "R_i_i": self.R_i_i.clone().to('cpu'),
            "R_q_q": self.R_q_q.clone().to('cpu'),
            "R_q_i": self.R_q_i.clone().to('cpu')
        })
    
    
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
        one_hot = torch.zeros_like(logits[q_i], dtype=torch.float)
        one_hot[logits[q_i].argmax().item()] = 1.
        one_hot.requires_grad = True
        
        loss = (logits[q_i] * one_hot).sum()
        loss.backward(retain_graph=True)
    
    
    def generate_rel_maps(self, 
                          q_idx: torch.Tensor, 
                          logits: torch.Tensor, 
                          encoder_attentions: List[torch.Tensor], 
                          decoder_attentions: List[torch.Tensor], 
                          cross_attentions: List[torch.Tensor]):
        # Set attention maps to require grad
        for attn in encoder_attentions + decoder_attentions + cross_attentions: 
            attn.requires_grad_(True)
            attn.retain_grad()
        
        self.n_i_t = encoder_attentions[0].shape[2]  # (batch_size, num_heads, n_img_tokens, n_img_tokens)
        logits.requires_grad_(True)
        
        pba = tqdm(q_idx, desc="Detection", leave=False)
        output_list = []
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
            for enc_step, enc_attn in enumerate(encoder_attentions): 
                self._self_attn_encoder_rel_update(enc_attn)
                self._add_snapshot(tag=f"encoder_{enc_step}")
            
            # update relevance maps for each decoder block
            pba.set_description(prefix + " Decoder pass")
            for dec_step, (dec_attn, cross_attn) in enumerate(zip(decoder_attentions, cross_attentions)):
                self._self_attn_decoder_rel_update(dec_attn)
                self._cross_attn_decoder_rel_update(cross_attn)
                self._add_snapshot(tag=f"decoder_{dec_step}")
            
            output_list.append({
                "query_idx": q_i,
                "snapshots": copy.deepcopy(self.snapshots),
                "relevance_map": self.R_q_i[q_i].to('cpu')
            })

        self.model.zero_grad()
        
        return output_list  


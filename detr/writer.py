from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List
from tqdm import tqdm

import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def drawDetections(image: Image.Image, detections: List, copy: bool = False):
    image = image.copy() if copy else image
    draw = ImageDraw.Draw(image)
    
    for score, category, (xmin, ymin, xmax, ymax) in detections:
        name, color = category, tuple(np.random.randint(0, 255, 3))
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=5)
        text = f'{name} {score:0.2f}'
        draw.text((xmin, ymin), text, fill='white', font=ImageFont.truetype("arial.ttf", size=20))
    
    return image


class TensorboardWriter:
    
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir)
    
    
    def _write_detection_and_relevance_map(self, image: Image.Image, rel_map: torch.Tensor, tag: str):
        plt.figure(figsize=(9, 9))
        
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        plt.title('Detection')
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        plt.imshow(rel_map, cmap='hot')
        plt.title('Relevance Map')
        plt.axis('off')
        
        plt.tight_layout()
        
        self.writer.add_figure(tag=tag, figure=plt.gcf())
        plt.close()
    
    def _write_attention_maps_block(self, attn_maps: List[torch.Tensor], tag: str, global_step: int = 0):
        grid_size = int(math.ceil(math.sqrt(len(attn_maps))))
        plt.figure(figsize=(11, 5))
        
        for head, attn_map in enumerate(attn_maps):
            plt.subplot(grid_size, grid_size, head + 1)
            plt.imshow(attn_map, cmap='hot')
            plt.title(f'Head {head}')
            plt.axis('off')
        
        plt.tight_layout()
        self.writer.add_figure(tag=tag, figure=plt.gcf(), global_step=global_step)
        plt.close()
    
    def __call__(self, image, exp_img):
        prefix = 'Writing in Tensorboard: '
        pba = tqdm(exp_img['explanations'], desc=prefix, leave=False)
        
        for step, (exp_info, detection) in enumerate(pba):
            query_idx = exp_info['query_idx']
            query_tag = f'Query_{query_idx}_{detection[1]}'
            rel_map = exp_info['relevance_map']
            
            pba.set_description(prefix + f'Detection {step} | Detection + Relevance Map...')
            self._write_detection_and_relevance_map(
                image=drawDetections(image, [detection], copy=True),
                rel_map=rel_map,
                tag=query_tag + "/Detection + Relevance Map")
            
            h, w = rel_map.shape  # resolution of the relevance map
            token_idx = (rel_map.flatten() > 0.9).nonzero()
            if len(token_idx) == 0:
                _, token_idx = torch.sort(rel_map.flatten(), descending=True)
                token_idx = token_idx[:1]
            num_heads = exp_img['outputs']['encoder_attentions'][0].shape[0]
            
            for t_i in token_idx:
                i, j = t_i // w, t_i % w
                i, j = i.item(), j.item()
                token_tag = query_tag + f"/Token_{i}_{j}"
                
                for block, encoder_block_attn in enumerate(exp_img['outputs']['encoder_attentions']):
                    pba.set_description(prefix + f'Detection {step} | Token ({i}, {j}) - Encoder block {block}')
                    self._write_attention_maps_block(
                        attn_maps=[encoder_block_attn[head][t_i].reshape(h, w) for head in range(num_heads)],
                        tag=token_tag + f"/Encoder_Blocks",
                        global_step=block)
            
            num_heads = exp_img['outputs']['cross_attentions'][0].shape[0]
            
            for block, cross_block_attn in enumerate(exp_img['outputs']['cross_attentions']):
                pba.set_description(prefix + f'Detection {step} - Decoder block {block}')
                self._write_attention_maps_block(
                    attn_maps=[cross_block_attn[head][query_idx].reshape(h, w) for head in range(num_heads)],
                    tag=query_tag + f"/Decoder_Blocks",
                    global_step=block)
                
            
        plt.close('all')
        self.writer.flush()
        self.writer.close()
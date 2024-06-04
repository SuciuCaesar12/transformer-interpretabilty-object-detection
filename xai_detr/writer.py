from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from typing import List
from xai_detr.visualization_utils import *

import torch
import matplotlib.pyplot as plt
import xai_detr.base as base


class TensorboardWriter:
    '''
    Tensorboard writer for visualizing explanations.
    '''
    
    def __init__(self, log_dir: Path):
        '''
        Attributes:
        ----------
        
        writer: SummaryWriter
            Summary writer for writing to tensorboard.
        '''
        self.writer = SummaryWriter(log_dir)
    
    def _write_attention_maps_block(
        self, 
        attn_maps: List[torch.Tensor], 
        tag: str, 
        global_step: int = 0
    ):
        '''
        Write attention maps of a layer/block to tensorboard.
        
        Parameters:
        ----------
        
        attn_maps: List[torch.Tensor]
            List of attention maps for each head.
        
        tag: str
            Tag for the tensorboard writer.
        
        global_step: int
            Global step for the tensorboard writer. Used to distinguish between blocks.
        
        '''
        self.writer.add_figure(
            tag=tag, 
            figure=plot_attention_maps_block(attn_maps), 
            global_step=global_step)
        
        plt.close()
    
    def write_detection_and_relevance_map(
        self, 
        image: Image.Image, 
        explanation: base.ExplanationItem,
        prefix_tag: str = ""
    ):
        '''
        Write detection and relevance map to tensorboard.
        
        Parameters:
        ----------
        
        image: PIL.Image.Image
            Image on which the detection is drawn.
        
        explanation: detr.ExplanationItem
            Explanation item containing the detection and relevance map.
        
        prefix_tag: str
            Prefix tag for the tensorboard writer.
        '''
        q_idx = explanation.detection.query_index
        label = explanation.detection.label
        
        self.writer.add_figure(
            tag=prefix_tag + f'query_{q_idx}_{label}/detection_and_relevance_map', 
            figure=plot_detection_and_relevance_map(
                image, 
                explanation.detection, 
                explanation.relevance_map)
            )
        plt.close()
    
    
    def write_image_token_attention_maps(
        self, 
        explainer_output: base.DetrExplainerOutput, 
        token_index: int,
        prefix_tag: str = ""
    ):
        '''
        Write attention maps for an image token in the encoder to tensorboard.
        
        Parameters:
        ----------
        
        explainer_output: detr.DetrExplainerOutput
            Output of the detr explainer.
        
        token_index: int
            Index of the token for which to write the attention maps.
            It is assumed that the token index is in the range [0, h * w).
        
        prefix_tag: str
            Prefix tag for the tensorboard writer.
        '''
        encoder_attentions = explainer_output.outputs.encoder_attentions
        num_heads = encoder_attentions[0].shape[0]
        
        h, w = explainer_output.outputs.conv_feature_shape
        i, j = token_index // w, token_index % w
        
        for block, encoder_block_attn in enumerate(encoder_attentions):
            attn_maps = [encoder_block_attn[head][token_index].reshape(h, w) for head in range(num_heads)]
            self._write_attention_maps_block(
                attn_maps=attn_maps,
                tag=prefix_tag + f"/token_{i}_{j}" + f"/encoder_blocks",
                global_step=block)
    
    
    def write_query_cross_attention_maps(
        self, 
        explainer_output: base.DetrExplainerOutput, 
        query_index: int,
        prefix_tag: str = ""
    ):
        '''
        Write cross attention maps for a query to tensorboard.
        
        Parameters:
        ----------
        
        explainer_output: detr.DetrExplainerOutput
            Output of the detr explainer.
        
        query_index: int
            Index of the query for which to write the attention maps.
        
        prefix_tag: str
            Prefix tag for the tensorboard writer.
        '''
        h, w = explainer_output.outputs.conv_feature_shape
        cross_attentions = explainer_output.outputs.cross_attentions
        num_heads = cross_attentions[0].shape[0]
        
        for block, cross_block_attn in enumerate(cross_attentions):
            attn_maps = [cross_block_attn[head][query_index].reshape(h, w) for head in range(num_heads)]
            self._write_attention_maps_block(
                attn_maps=attn_maps,
                tag=prefix_tag + "/decoder_blocks",
                global_step=block)
    
    def close(self):
        '''Close the tensorboard writer.'''
        self.writer.close()
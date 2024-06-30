from PIL import Image, ImageDraw, ImageFont
from typing import List
from matplotlib import pyplot as plt

import xai_detr.base as base
import numpy as np
import torch
import math


def drawDetections(image: Image.Image, detections: List[base.DetectionItem], copy: bool = False):
    '''
    Draw detection boxes on the image.
    
    Parameters:
    ----------
    
    image: PIL.Image.Image
        Image on which to draw the detections.
    
    detections: List[detr.DetectionItem]
        List of detection items.
    
    copy: bool
        Whether to copy the image before drawing the detections.
    '''
    
    image = image.copy() if copy else image
    draw = ImageDraw.Draw(image)
    
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = tuple(np.random.randint(0, 255, 3))
        text = f'{d.label} {d.score:0.2f}'
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=5)
        draw.text((x1, y1), text, fill='white', font=ImageFont.truetype("arial.ttf", size=20))
    
    return image


def plot_attention_maps_block(attn_maps: List[torch.Tensor]) -> plt.Figure:
    '''
    Plot attention maps for a block of the model.
    
    Parameters:
    ----------
    
    attn_maps: List[torch.Tensor]
        List of attention maps for each head.
    '''
    grid_size = int(math.ceil(math.sqrt(len(attn_maps))))
    plt.figure(figsize=(11, 5))
    
    for head, attn_map in enumerate(attn_maps):
        plt.subplot(grid_size, grid_size, head + 1)
        plt.imshow(attn_map, cmap='hot')
        # plt.title(f'Head {head}')
        plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()


def plot_detection_and_relevance_map(image: Image.Image, detection: base.DetectionItem, relevance_map: torch.Tensor):
    '''
    Plot detection and relevance map side by side.
    
    Parameters:
    ----------
    
    image: PIL.Image.Image
        Image on which the detection is drawn.
    
    detection: detr.DetectionItem
        Detection item containing the detection box.
    
    relevance_map: torch.Tensor
        Relevance map for the detection.
    '''
    plt.figure(figsize=(9, 9))
    
    plt.subplot(2, 1, 1)
    plt.imshow(drawDetections(image, [detection], copy=True), cmap='hot')
    # plt.title('Detection')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(relevance_map, cmap='hot')
    # plt.title('Relevance Map')
    plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()
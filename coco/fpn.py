import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple


class MaskRefinementBlock(nn.Module):
    '''
    Mask Refinement Block. Uses a Feature-Pyramic Network (FPN)-like architecture to refine the mask.
    '''
    
    def __init__(self, 
                 in_channels: int, 
                 d: int = 256, 
                 alpha: float = 0.5, 
                 batch_norm: bool = True, 
                 bias: bool = True):
        '''
        Parameters:
        ----------
        
        in_channels: int
            Number of input channels of the bottom-up feature map.
        
        d: int
            Number of hidden channels.
            It is supposed that the top-down feature map has the same number of channels.
        
        alpha: float
            Weighting factor for the addition of the conditional masks with the predicted masks.
        
        batch_norm: bool
            Whether to use batch normalization.
        
        bias: bool
            Whether to use bias.
        '''
        super(MaskRefinementBlock, self).__init__()
        
        self.in_channels = in_channels
        self.d = d
        self.alpha = alpha
        self.batch_norm = batch_norm
        self.bias = bias
        
        if self.batch_norm:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, d, kernel_size=1, bias=self.bias),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
        else:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, d, kernel_size=1, bias=self.bias),
                nn.ReLU(inplace=True),
            )
        
        if self.batch_norm:
            self.conv_merge = nn.Sequential(
                nn.Conv2d(d, d, kernel_size=3, padding=1, bias=self.bias),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv_merge = nn.Sequential(
                nn.Conv2d(d, d, kernel_size=3, padding=1, bias=self.bias),
                nn.ReLU(inplace=True),
            )
        
        self.conv_mask = nn.Conv2d(d + 1, 1, kernel_size=3, padding=1, bias=self.bias) # + 1 for the conditional masks
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, 
                cond_masks: torch.Tensor, 
                td: torch.Tensor,
                bu: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Parameters:
        ----------
        
        cond_masks: torch.Tensor
            Conditional masks with shape [None, 1, h, w] with values in the range [0, 1].
        
        bu: torch.Tensor
            Bottom-Up feature map from the backbone with shape [1, in_channels, 2 * h, 2 * w].
            
        td: torch.Tensor
            Top-Down feature map from the backbone with shape [1, d, h, w].
        
        Returns:
        
        masks: torch.Tensor
            Refined masks with shape [None, 1, 2 * h, 2 * w].
        
        fpn: torch.Tensor
            Refined feature map with shape [d, 2 * h, 2 * w].
        '''
        # check for mismatched sizes (due to odd input size)
        bu_shape = bu.shape[-2:]
        
        if bu_shape != 2 * td.shape[-2:]:
            td = F.interpolate(td, size=bu_shape, mode='nearest')
            cond_masks = F.interpolate(cond_masks, size=bu_shape, mode='nearest')
        else:
            td = F.interpolate(td, scale_factor=2, mode='nearest')
            cond_masks = F.interpolate(cond_masks, scale_factor=2, mode='nearest')    
        
        # adapt the bottom-up feature map and merge it with the top-down feature map
        fpn = self.conv_merge(td + self.adapter(bu))
        
        # refine the mask
        masks = self.conv_mask(torch.cat([fpn, cond_masks], dim=1))
        masks = self.sigmoid(masks)
        
        return masks, fpn


class MaskRefinementModel(nn.Module):
    
    def __init__(self, 
                 in_channels: List[int], 
                 alphas: Optional[List[int]] = None, 
                 batch_norm: bool = True, 
                 bias: bool = True, 
                 d: int = 256):
        '''
        Parameters:
        ----------
        in_channels: int
            Number of input channels for each feature map.
        '''
        super(MaskRefinementModel, self).__init__()
        
        if alphas is None:
            self.alphas = [0.5] * (len(in_channels) - 1)
        
        self.batch_norm = batch_norm
        self.bias = bias
        self.d = d
        self.in_channels = in_channels
        
        self.blocks = nn.ModuleList([
            MaskRefinementBlock(
                in_channels=self.in_channels[i + 1], 
                alpha=self.alphas[i], 
                batch_norm=self.batch_norm, 
                bias=self.bias, 
                d=self.d
            )
            for i in range(len(in_channels) - 1)
        ])
        
        if self.batch_norm:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels[0], d, kernel_size=1, bias=self.bias),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
        else:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels[0], d, kernel_size=1, bias=self.bias),
                nn.ReLU(inplace=True),
            )
        
    
    def forward(self, cond_masks: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        '''
        Parameters:
        ----------
        
        cond_masks: torch.Tensor
            Conditional masks with shape [None, 1, h, w] with values in the range [0, 1].
        
        features: List[torch.Tensor]
            List of feature maps from the backbone.
        
        Returns:
        -------
        
        masks: torch.Tensor
            Refined masks.
        '''
        fpn = self.adapter(features[0])
        
        for blk, bu in zip(self.blocks, features[1:]):
            cond_masks, fpn = blk(
                cond_masks=cond_masks, 
                td=fpn, 
                bu=bu
            )
        
        return cond_masks


def build() -> MaskRefinementModel:
    return MaskRefinementModel(
        in_channels=[2048, 1024, 512, 256], 
        d=128,
        alphas=None,
        batch_norm=False, 
        bias=False
    )
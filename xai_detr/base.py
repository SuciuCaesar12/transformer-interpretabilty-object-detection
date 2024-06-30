from abc import abstractmethod
from dataclasses import dataclass
from PIL import Image
from typing import Tuple, Dict, List, Any, Union

import torch


@dataclass
class DetectionItem:
    ''' 
    Dataclass for storing detection information.
    
    Attributes
    ----------
    
    query_index: int
        The index of the query token associated with the detection.
    
    box: Tuple[float, float, float, float]
        The bounding box of the detection in the format (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    
    score: float
        The confidence score of the detection.
    
    label: str
        The label of the detection.
    '''
    
    box: Tuple[float, float, float, float]
    score: float
    label: str
    query_index: int = None


@dataclass
class SnapshotItem:
    ''' 
    Dataclass for storing snapshot information.
    
    
    Attributes
    ----------
    
    tag: str
        The tag of the snapshot. Used to identify on which layer of the transformer the snapshot was taken.
    
    R_i_i: torch.Tensor
        The self-relevance map for the image tokens.
        
    R_q_q: torch.Tensor
        The self-relevance map for the query tokens.
        
    R_q_i: torch.Tensor
        The cross-relevance map between the query and image tokens.
    '''
    
    tag: str
    R_i_i: torch.FloatTensor
    R_q_q: torch.FloatTensor
    R_q_i: torch.FloatTensor


@dataclass
class ExplanationItem:
    ''' 
    Dataclass for storing explanation information for one detection.
    
    Attributes
    ----------
    
    detection: DetectionItem
        The detection for which the explanation is computed.
    
    relevance_map: torch.Tensor
        The relevance map for the detection. It shows the relevance score for each image token.
    '''
    
    relevance_map: torch.FloatTensor
    detection: DetectionItem = None


class DetrOutput:
    '''
    The output of the DETR model.
    
    Attributes
    ----------
    logits: torch.Tensor of shape (batch_size, num_queries, num_classes)
        The output logits from the transformer head of the model.
        
    pred_boxes: torch.Tensor of shape (batch_size, num_queries, 4)
        The predicted bounding boxes from the model.
        
    encoder_attentions: Tuple[torch.Tensor] of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
        The attention maps from the encoder self-attention layers.
        
    decoder_attentions: Tuple[torch.Tensor] of shape (num_layers, batch_size, num_heads, num_queries, num_queries)
        The attention maps from the decoder self-attention layers.
        
    cross_attentions: Tuple[torch.Tensor] of shape (num_layers, batch_size, num_heads, num_queries, seq_len)
        The attention maps from the cross-attention layers.
        
    conv_feature_shape: Tuple[int, int]
        The spatial resolution of the last convolutional feature map (H, W). Used to reshape the relevance maps.
        
    Note
    ----
    
    During inference the model will have the gradients enabled. It is important that the output tensors are NOT detached from the computation graph.
    This is because the explainer will compute the gradients of the attention maps with respect to the logits of the model.
    '''
    
    def __init__(self,
                 logits: torch.FloatTensor,
                 pred_boxes: torch.FloatTensor,
                 encoder_attentions: Tuple[torch.FloatTensor],
                 decoder_attentions: Tuple[torch.FloatTensor],
                 cross_attentions: Tuple[torch.FloatTensor],
                 conv_feature_shape: Tuple[int, int],
                 pred_masks: torch.FloatTensor = None
                 ) -> None:
        
        self.logits = logits
        self.pred_boxes = pred_boxes
        self.pred_masks = pred_masks
        self.encoder_attentions = encoder_attentions
        self.decoder_attentions = decoder_attentions
        self.cross_attentions = cross_attentions
        self.conv_feature_shape = conv_feature_shape
    
    
    def to(self, device: Union[torch.device, str]):
        self.logits = self.logits.to(device)
        self.pred_boxes = self.pred_boxes.to(device)
        self.pred_masks = self.pred_masks.to(device) if self.pred_masks is not None else None
        self.encoder_attentions = tuple(att.to(device) for att in self.encoder_attentions)
        self.decoder_attentions = tuple(att.to(device) for att in self.decoder_attentions)
        self.cross_attentions = tuple(att.to(device) for att in self.cross_attentions)
        
        return self


    def detach(self):
        self.logits = self.logits.detach()
        self.pred_boxes = self.pred_boxes.detach()
        self.pred_masks = self.pred_masks.detach() if self.pred_masks is not None else None
        self.encoder_attentions = tuple(att.detach() for att in self.encoder_attentions)
        self.decoder_attentions = tuple(att.detach() for att in self.decoder_attentions)
        self.cross_attentions = tuple(att.detach() for att in self.cross_attentions)
        
        return self


    def squeeze(self, dim: int = 0):
        self.logits = self.logits.squeeze(dim)
        self.pred_boxes = self.pred_boxes.squeeze(dim)
        self.pred_masks = self.pred_masks.squeeze(dim) if self.pred_masks is not None else None
        self.encoder_attentions = tuple(att.squeeze(dim) for att in self.encoder_attentions)
        self.decoder_attentions = tuple(att.squeeze(dim) for att in self.decoder_attentions)
        self.cross_attentions = tuple(att.squeeze(dim) for att in self.cross_attentions)
        
        return self
    
    def unsqueeze(self, dim: int = 0):
        self.logits = self.logits.unsqueeze(dim)
        self.pred_boxes = self.pred_boxes.unsqueeze(dim)
        self.pred_masks = self.pred_masks.unsqueeze(dim) if self.pred_masks is not None else None
        self.encoder_attentions = tuple(att.unsqueeze(dim) for att in self.encoder_attentions)
        self.decoder_attentions = tuple(att.unsqueeze(dim) for att in self.decoder_attentions)
        self.cross_attentions = tuple(att.unsqueeze(dim) for att in self.cross_attentions)
        
        return self


@dataclass
class DetrExplainerOutput:
    ''' 
    Dataclass for storing the output of the DETR explainer.
    
    Attributes
    ----------
    
    image: PIL.Image.Image
        The input image.
    
    outputs: DetrOutput
        The output of the DETR model.
    
    explanations: List[ExplanationItem]
        The explanations for each detection.
    '''
    
    outputs: DetrOutput
    explanations: List[ExplanationItem]
    image: Image.Image = None
    
    def save(self, path: str):
        '''
        Save the output of the explainer to a file.
        
        Parameters
        ----------
        
        path: str
            The path to the file where the output will be saved.
        '''
        torch.save(self, path)
    
    @staticmethod
    def load(path: str) -> 'DetrExplainerOutput':
        '''
        Load the output of the explainer from a file.
        
        Parameters
        ----------
        
        path: str
            The path to the file where the output is saved.
        
        Returns
        -------
        
        explainer_output: DetrExplainerOutput
            The output of the explainer.
        '''
        return torch.load(path)


class AbstractDetrModule:
    '''
    Abstract class for the DETR model. Used to define the interface for the DETR model.
    This class should be subclassed and the methods should be implemented.
    '''
    
    @abstractmethod
    def preprocess(self, image: Image.Image, **kwargs)-> Tuple[Dict[str, Any], Dict[str, Any]]:
        '''
        Preprocess the image for the DETR model.
        
        Parameters
        ----------
        image: PIL.Image.Image
            The input image.
            
        You can pass additional arguments to the method. The arguments should be used to preprocess the image.
            
        Returns
        -------
        
        inputs: Dict[str, torch.Tensor]
            The preprocessed image in the format expected by your DETR model. 
            This will be the input to the predict method.
        '''
        pass
    
    @abstractmethod
    def predict(self, inputs: Dict[str, torch.Tensor], **kwargs) -> DetrOutput:
        '''
        Perform inference on the input image.
        
        Parameters
        ----------
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor]
            The preprocessed image in the format expected by your DETR model. It is the output of the preprocess method.
            
        Returns
        -------
        outputs: DetrOutput
            The output of the DETR model.
        '''
        pass
    
    
    @abstractmethod
    def id2label(self) -> Dict[int, str]:
        '''
        Get the mapping from class id to class label.
        
        Returns
        -------
        
        id2label: Dict[int, str]
            The mapping from class id to class label.
        
        '''
        pass
    
    @abstractmethod
    def no_object_id(self) -> int:
        '''
        Get the id of the "no object" class.
        
        Returns
        -------
        
        no_object_id: int
            The id of the "no object" class.
        '''
        pass
    
    @abstractmethod
    def num_queries(self) -> int:
        '''
        Get the number of query tokens used by the model.
        
        Returns
        -------
        
        num_queries: int
            The number of query tokens used by the model.
        '''
        pass
    
    @abstractmethod
    def zero_grad(self):
        '''
        Clear the gradients of the model.
        '''
        pass
    
    @abstractmethod
    def requires_grad(self, requires_grad: bool):
        '''
        Set the requires_grad attribute of the model.
        
        Parameters
        ----------
        
        requires_grad: bool
            If True, the gradients will be computed for the model parameters. If False, the gradients will not be computed.
        '''
        pass
    
    @abstractmethod
    def to(self, device: torch.device):
        '''
        Move the model to the specified device.
        
        Parameters
        ----------
        
        device: torch.device
            The device to which the model will be moved.
        '''
        pass
    
    @abstractmethod
    def freeze_backbone(self):
        '''
        Freeze the backbone of the model. Used for faster computation of the relevance maps since the gradients of the backbone are not used.
        If not implemented, the explainer will use the model as is.
        '''
        pass
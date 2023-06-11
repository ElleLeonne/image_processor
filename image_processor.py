import numpy as np
import torch
from torch import nn

from typing import List, Union, Optional
from transformers.image_transforms import normalize, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling)
from .preprocess import image_from_source

class VisionEmbeddings(nn.Module):
    """
    Takes a list of images, and converts them into tensors
    To look into: Paralellization
    """
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.get("patch_size", 16)
        self.max_image_height = config.get("max_image_height", 256)
        self.num_channels = config.get("image_channels", 3)
        self.hidden_size = config["hidden_size"]

        #Shifting normalization settings
        self.alpha = 0.999
        self.dynamic_mean = 0.0
        self.dynamic_var = 0.004

        #Image processing layers
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
            
    def _normalize(self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,) -> np.ndarray:
        """Records a shifting window of mean and var rates for dynamic adjustment. var = std**2"""
        self.dynamic_mean = self.alpha * self.dynamic_mean + (1 - self.alpha) * np.mean(image)
        self.dynamic_var = self.alpha * self.dynamic_var + (1 - self.alpha) * np.var(image)

        return normalize(image, mean=self.dynamic_mean, std=np.sqrt(self.dynamic_var)+1e-5, data_format=data_format, **kwargs)

    def resize_image(self,
        image: np.ndarray,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs) -> np.ndarray:
        """Rescales an image per max_height for patching.
        Adds padding to keep aspect_ratio intact."""
        width, height, _ = image.shape
        aspect_ratio = width / height

        new_height = min(self.max_image_height, height)
        new_width = round(new_height * aspect_ratio)

        image = resize(image, size=(new_height, new_width), resample=resample, data_format=data_format, **kwargs)

        pad_height = (self.patch_size - new_height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - new_width % self.patch_size) % self.patch_size

        if pad_height or pad_width:
            image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), 
                mode='constant', constant_values=0)  #pad with black, bottom and right, no padding for channels
        return image
        
    def unwind_meta_list(self, image_list_of_lists):
        """Converts the list-of-lists that the model receives into a single list,
        and remembers their positions for later."""
        image_list_singular = []
        positions = []
        for sublist_idx, sublist in enumerate(image_list_of_lists):
            for img_idx, img in enumerate(sublist):
                image_list_singular.append(img)
                positions.append((sublist_idx, img_idx))
        return image_list_singular, positions
    
    def reconstruct_meta_list(self, processed_images, positions, size):
        """Converts a list of tensors back into its original list-of-lists,
        for proper concatenation in the main model body"""
        reconstructed_list = [[] for _ in range(size)]
        for img, (position, _) in zip(processed_images, positions):
            reconstructed_list[position].append(img)
        return reconstructed_list

    def forward(self,
                image_list: Optional[List] = None,
                image_tensor: Optional[torch.Tensor] = None,
                data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST) -> List[List]:
        """Takes a meta-list (list of lists) of image data (url, directory, np.array), and returns a list of tensors.
        return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`."""
        if image_list is not None and image_tensor is not None:
            raise ValueError("You cannot specify both image_list and image_tensor at the same time")
        if not image_list and not image_tensor:
            return []
        if image_tensor is not None:
            image_list = [image_tensor]

        images, meta_position = self.unwind_meta_list(image_list) #We need to convert our list-of-lists to a normal list for batch processing
        images = [image_from_source(image) for image in images] #Should return an np array
        images = [self.resize_image(image) for image in images] #(h, w, c)
        images = [self._normalize(image) for image in images]
        images = [to_channel_dimension_format(image, data_format) for image in images]
        images = [torch.tensor(image, dtype=self.projection.bias.dtype, device=self.projection.bias.device) for image in images]
        try: #Try to make a batch for our conv2d layer
            images = torch.stack(images) #(c, h, w)
            images = self.projection(images).transpose(1, 3, 4, 2) #b, dim, h, w
            images = images.unbind(0) #Unstack our batch to make sure we're returning the same formats
        except: #Give up and process them separately (padding will ruin patching)
            images = [self.projection(image).transpose(2, 3, 1) for image in images] #shape dim, h, w
        return self.reconstruct_meta_list(images, meta_position, len(image_list))
        #return images

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple
import math


class VGG16(nn.Module):
    def __init__(self, 
                 num_classes: int = 1000, 
                 in_channels: int = 3, 
                 input_size: Tuple[int, int] = (224, 224), 
                 init_weights: bool = True,
                 dropout: float = 0.5):
        """
        Flexible VGG16 implementation supporting different input sizes and channels.

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            input_size: Tuple of (height, width) for input images
            init_weights: Whether to initialize weights
            dropout: Dropout rate for fully connected layers
        """
        super().__init__()

        self.input_size = input_size

        # Calculate the size of feature maps after conv layers
        self.feature_map_size = self._calculate_feature_map_size()

        # Conv layers
        self.features = nn.Sequential(
            # Conv Block #1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block #2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block #3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block #4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block #5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the flattened feature size
        flattened_size = 512 * self.feature_map_size[0] * self.feature_map_size[1]

        # Classification layers with adaptive size
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()
 
    def _calculate_feature_map_size(self) -> Tuple[int, int]:
        """Calculate the size of feature maps after all conv and pooling layers"""
        # VGG16 has 5 max pooling layers, each reducing size by factor of 2
        h, w = self.input_size
        h = math.ceil(h / (2 ** 5))
        w = math.ceil(w / (2 ** 5))
        return (h, w)

    def _initialize_weights(self):
        # Initialize convolutional layers
        for module in self.features:
            if isinstance(module, nn.Conv2d):
                # Initialize weights using Kaiming/He initialization
                fan_in = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
                bound = mx.sqrt(2.0 / fan_in)
                module.weight = mx.random.uniform(
                    low=bound,
                    high=bound,
                    shape=module.weight.shape
                )
                if module.bias is not None:
                    module.bias = mx.zeros(module.bias.shape)
        
        # Initialize linear layers
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                # Initialize weights using Kaiming/He initialization
                fan_in = module.weight.shape[1]
                bound = mx.sqrt(2.0 / fan_in)
                module.weight = mx.random.uniform(
                    low=bound,
                    high=bound,
                    shape=module.weight.shape
                )
                if module.bias is not None:
                    module.bias = mx.zeros(module.bias.shape)
    
    @staticmethod
    def process_image(image: mx.array, normalize: bool = True) -> mx.array:
        """
        Process an input image to prepare it for the model.

        Args:
            image: Input image array (H, W) for grayscale or (H, W, C) for RGB
            normalize: Whether to normalize pixel values to [0, 1]
        
        Returns:
            Processed image array in format (1, C, H, W)
        """
        # Add batch and channel dimensions if needed
        if len(image.shape) == 2:
            # Grayscale image: add channel dimension
            image = mx.expand_dims(image, axis=0)
        elif len(image.shape) == 3:
            # RGB image: transpose to (C, H, W)
            image = mx.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = mx.expand_dims(image, axis=0)

        # Normalize pixel values to [0, 1] if requested
        if normalize:
            image = image / 255.0
        
        return image


    def __call__(self, x):
        # Apply Conv Layers
        x = self.features(x)

        # Flatten the output for the dense layers
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Apply classifier
        x = self.classifier(x)
        return x
    
    
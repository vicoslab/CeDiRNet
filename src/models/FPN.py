import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F

import numpy as np

class FPN(nn.Module):
    def __init__(self, num_classes, backbone='resnet34', use_custom_fpn=False, init_decoder_gain=0.1, fpn_args={}):
        super().__init__()

        self.use_custom_fpn = use_custom_fpn
        print('Creating FPN segmentation with {} classes'.format(num_classes))
        if self.use_custom_fpn:
            self.model = Custom_FPN(backbone, classes=num_classes, activation=None, encoder_depth=4, upsampling=2,
                                    decoder_merge_policy='add', encoder_weights='imagenet', **fpn_args)
        else:
            self.model = smp.FPN(backbone, classes=num_classes, activation=None, encoder_depth=4, upsampling=2,
                                 decoder_merge_policy='add', encoder_weights='imagenet', **fpn_args)

        self.preprocess_params = smp.encoders.get_preprocessing_params(backbone, pretrained="imagenet")
        self.init_decoder_gain = init_decoder_gain

    def init_output(self):
        with torch.no_grad():
            decoder = self.model.decoder
            output_conv = self.model.segmentation_head[0]

            decoder_convs = []

            if self.use_custom_fpn:
                decoder_convs += [output_conv.block[0],
                                  self.model.segmentation_head[2]]
                decoder_convs += [decoder.p2.skip_conv, decoder.p3.skip_conv, decoder.p4.skip_conv, decoder.p5]
            else:
                decoder_convs += [output_conv]
                decoder_convs += [decoder.p2.skip_conv, decoder.p3.skip_conv, decoder.p4.skip_conv, decoder.p5]

            for seg_block in decoder.seg_blocks:
                for conv_block in seg_block.block:
                    decoder_convs.append(conv_block.block[0])

            for c in decoder_convs:
                if type(c) == torch.nn.modules.conv.Conv2d:
                    print('initialize decoder layer with size: ', c.weight.size())
                    torch.nn.init.xavier_normal_(c.weight,gain=self.init_decoder_gain)
                    if c.bias is not None:
                        torch.nn.init.zeros_(c.bias)


    def forward(self, input):
        input = preprocess_input(input,**self.preprocess_params)
        output = self.model.forward(input)

        return output


from typing import Optional, Union
from segmentation_models_pytorch.fpn.decoder import FPNBlock, MergeBlock, Conv3x3GNReLU, FPNDecoder
from segmentation_models_pytorch.base import SegmentationModel, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.modules import Flatten, Activation

class Custom_FPN(SegmentationModel):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        decoder_merge_policy: determines how to merge outputs inside FPN.
            One of [``add``, ``cat``]
        decoder_dropout: spatial dropout rate in range (0, 1).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_segmentation_head_channels: int = 64,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 2,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            segmentation_channels=decoder_segmentation_head_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels,  segmentation_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d_1 = Conv3x3GNReLU(in_channels, segmentation_channels, upsample=False)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d_2 = nn.Conv2d(segmentation_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = Activation(activation)

        super().__init__(conv2d_1, upsampling, conv2d_2, activation)


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = torch.tensor(mean)
        mean = mean.reshape([1, -1, 1, 1])
        x = x - mean.to(x.device)

    if std is not None:
        std = torch.tensor(std)
        std = std.reshape([1, -1, 1, 1])
        x = x / std.to(x.device)

    return x
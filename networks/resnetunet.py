import torch
import torch.nn as nn


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)




class ResNetUNet(nn.Module):

    def __init__(self, n_input_class, n_scores, n_out_class):
        super(ResNetUNet, self).__init__()

        if True:

            base_model = resnet34(pretrained=False)

            self.base_layers = list(base_model.children())

            self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
            self.layer0_1x1 = convrelu(64, 64, 1, 0)
            self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 256, x.H/4, x.W/4)
            self.layer1_1x1 = convrelu(64, 64, 1, 0)
            self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
            self.layer2_1x1 = convrelu(128, 128, 1, 0)
            self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
            self.layer3_1x1 = convrelu(256, 256, 1, 0)
            self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
            self.layer4_1x1 = convrelu(512, 256, 1, 0)

            self.latent_space = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # self.fc_output = nn.Linear(512, n_scores, bias=True)
            self.fc_output = nn.Linear(512, n_scores, bias=True)

            self.conv_up3 = convrelu(512, 256, 3, 1)
            self.conv_up2 = convrelu(384, 256, 3, 1)
            self.conv_up1 = convrelu(320, 128, 3, 1)
            self.conv_up0 = convrelu(192, 64, 3, 1)

            self.conv_original_size0 = convrelu(n_input_class, 64, 3, 1)
            self.conv_original_size1 = convrelu(64, 64, 3, 1)
            self.conv_original_size2 = convrelu(128, 64, 3, 1)

        else:

            base_model = resnet50(pretrained=False)

            self.base_layers = list(base_model.children())

            self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
            self.layer0_1x1 = convrelu(64, 64, 1, 0)
            self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 256, x.H/4, x.W/4)
            self.layer1_1x1 = convrelu(256, 256, 1, 0)
            self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
            self.layer2_1x1 = convrelu(512, 512, 1, 0)
            self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
            self.layer3_1x1 = convrelu(1024, 512, 1, 0)
            self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
            self.layer4_1x1 = convrelu(2048, 1024, 1, 0)

            self.latent_space = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.fc_output = nn.Linear(2048, n_scores, bias=True)

        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.upsample = nn.functional.interpolate(scale_factor=2, mode='bilinear',align_corners=True)


            self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
            self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
            self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
            self.conv_up0 = convrelu(64 + 256, 128, 3, 1)


        #print self.layer0[0]

            self.conv_original_size0 = convrelu(n_input_class, 64, 3, 1)
            self.conv_original_size1 = convrelu(64, 64, 3, 1)
            self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_out_class, 1)

    def forward(self, input, is_training = True, verbose = False):
        if True:#is_training == True:
            x_original = self.conv_original_size0(input)
            x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        if verbose == True: print(layer0.size(), 'layer0')
        layer1 = self.layer1(layer0)
        if verbose == True: print(layer1.size(), 'layer1')
        layer2 = self.layer2(layer1)
        if verbose == True: print(layer2.size(), 'layer2')
        layer3 = self.layer3(layer2)
        if verbose == True: print(layer3.size(), 'layer3')
        layer4 = self.layer4(layer3)
        if verbose == True: print(layer4.size(), 'layer4')

        scores = self.latent_space(layer4)
        if verbose == True: print(scores.size())
        scores = self.fc_output(scores.squeeze())


        if True:#is_training == True:
            layer4 = self.layer4_1x1(layer4)
            if verbose == True: print(layer4.size(), 'layer 4 1x1')

            x = nn.functional.interpolate(layer4, scale_factor=2, mode='bilinear',align_corners=True)
            if verbose == True: print(x.size(), 'xinterp')
            layer3 = self.layer3_1x1(layer3)
            if verbose == True: print(layer3.size(), 'layer3 1x1')
            x = torch.cat([x, layer3], dim=1)
            if verbose == True: print(x.size(), 'cat')
            x = self.conv_up3(x)
            if verbose == True: print(x.size(), 'conv3')

            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
            if verbose == True: print(x.size(), 'xinterp')
            layer2 = self.layer2_1x1(layer2)
            if verbose == True: print(layer2.size(), 'layer2 1x1')
            x = torch.cat([x, layer2], dim=1)
            if verbose == True: print(x.size(), 'cat')
            x = self.conv_up2(x)
            if verbose == True: print(x.size(), 'conv2')

            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
            if verbose == True: print(x.size(), 'xinterp')
            layer1 = self.layer1_1x1(layer1)
            if verbose == True: print(layer1.size(), 'layer1 1x1')
            x = torch.cat([x, layer1], dim=1)
            if verbose == True: print(x.size(), 'cat')
            x = self.conv_up1(x)
            if verbose == True: print(x.size(), 'conv1')

            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
            if verbose == True: print(x.size(), 'xinterp')
            layer0 = self.layer0_1x1(layer0)
            if verbose == True: print(layer0.size(), 'layer0 1x1')
            x = torch.cat([x, layer0], dim=1)
            if verbose == True: print(x.size(), 'cat')
            x = self.conv_up0(x)
            if verbose == True: print(x.size(), 'conv0')

            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
            if verbose == True: print(x.size(), 'xinterp')
            x = torch.cat([x, x_original], dim=1)
            if verbose == True: print(x.size(), 'cat')
            x = self.conv_original_size2(x)
            if verbose == True: print(x.size(), '1x1')

            recon = self.conv_last(x)
            if verbose == True: print(recon.size(), '1x1')
            recon = recon[:, :, :, 5:-5]
            if verbose == True: print(recon.size(), '1x1')
        else:
            recon = None

        return scores, recon

    def forward_half(self, input):
        #x_original = self.conv_original_size0(input)
        #x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        if verbose == True: print(layer0.size())
        layer1 = self.layer1(layer0)
        if verbose == True: print(layer1.size())
        layer2 = self.layer2(layer1)
        if verbose == True: print(layer2.size())
        layer3 = self.layer3(layer2)
        if verbose == True: print(layer3.size())
        layer4 = self.layer4(layer3)
        if verbose == True: print(layer4.size())
        scores = self.latent_space(layer4)
        if verbose == True: print(scores.size())
        scores = self.fc_output(scores.squeeze())

        return scores



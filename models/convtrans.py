import torch
import torch.nn as nn
import torchvision
import collections
from models.transformer_module import Transformer
from models.convolution_module import ConvBlock, OutputNet


class VGG16Trans(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False):
        super().__init__()
        self.scale_factor = 16//dcsize
        self.encoder = nn.Sequential(
            ConvBlock(cin=3, cout=64),
            ConvBlock(cin=64, cout=64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=64, cout=128),
            ConvBlock(cin=128, cout=128),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=128, cout=256),
            ConvBlock(cin=256, cout=256),
            ConvBlock(cin=256, cout=256),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=256, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
        )
        self.conv_bbox = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size = 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
        )
        self.tran_decoder = Transformer(layers=4)
        self.tran_decoder_p2 = OutputNet(dim=512)

        # self.output_bbox = nn.Sequential(

        # )
        
        # self.conv_decoder = nn.Sequential(
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        # )
        # self.conv_decoder_p2 = OutputNet(dim=512)

        self._initialize_weights()
        if not load_weights: 
        # load the trained parameters from pretrained VGG16 model into self.encoder
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.encoder.state_dict().items())):
                temp_key = list(self.encoder.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.encoder.load_state_dict(fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, ex):
        raw_x = self.encoder(x)
        # raw_ex = self.encoder(ex) #bbox
        # raw_ex = self.conv_bbox(raw_ex) #bbox
        bs, c, h, w = raw_x.shape
        # ebs, ec, eh, ew = raw_ex.shape #bbox
        # print(" >>>>> Shape_x: ", raw_x.shape) #Shape_x:  torch.Size([8, 512, 24, 24]) from[384, 384]
        # print(" >>>>> Shape_ex: ", raw_ex.shape) #Shape_ex:  torch.Size([8, 512, 6, 6]) from [96, 96]
        #v.2 make it to torch.Size([8, 512, 1, 1])

        # path-transformer
        x = raw_x.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        # ex = raw_ex.flatten(2).permute(2, 0, 1) #bbox  # -> bs c hw -> hw b c
        # print(" >>>>> Shape_xx_val: ", x.shape) #Shape_x:  torch.Size([936, 8, 512]) from[384, 384]
        # print(" >>>>> Shape_exx_val: ", ex.shape) #Shape_ex:  torch.Size([36, 8, 512])from [96, 96]
        # scale_bbox = int(x.shape[0] / ex.shape[0]) #bbox
        # x = x + ex.repeat(scale_bbox, 1, 1) #bbox
        x = self.tran_decoder(x, (h, w))
        # ex = self.tran_decoder(ex, (eh, ew))
        # print(" >>>>> Shape_xx: ", x.shape) #Shape_x:  torch.Size([576, 8, 512]) from[384, 384]
        # print(" >>>>> Shape_exx: ", ex.shape) #Shape_ex:  torch.Size([36, 8, 512])from [96, 96]
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x)

        return y

import math
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class Residual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim_RB = round(inp / expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim_RB, hidden_dim_RB, 3, stride, 1, groups=hidden_dim_RB, bias=False),
                BatchNorm2d(hidden_dim_RB),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim_RB, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim_RB, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim_RB),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim_RB, hidden_dim_RB, 3, stride, 1, groups=hidden_dim_RB, bias=False),
                BatchNorm2d(hidden_dim_RB),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim_RB, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim_IRB = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim_IRB, 1, 1, 0, bias=False),
            BatchNorm2d(hidden_dim_IRB),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim_IRB, hidden_dim_IRB, 3, stride, 1, groups=hidden_dim_IRB, bias=False),
            BatchNorm2d(hidden_dim_IRB),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim_IRB, oup, 1, 1, 0, bias=False),
            BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block_RB = Residual
        block_IRB = InvertedResidual
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # t, c, n, s, l
            [1, 16, 1, 1, 0],
            [6, 24, 4, 2, 1],
            [6, 32, 4, 2, 2],
            [6, 64, 4, 2, 3],
            [6, 96, 4, 1, 4],
            [6, 160, 4, 2, 5],
            [6, 320, 1, 1, 6],
        ]
        
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s, l in interverted_residual_setting:
            output_channel = int(c * width_mult)
            if l < 3:
                for i in range(n):
                    if i == 0:
                        self.features.append(block_RB(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_RB(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            else:
                for i in range(n):
                    if i == 0:
                        self.features.append(block_IRB(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_IRB(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # Classify.
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(pretrained=False, **kwargs):
    # We only used the name mobilenetv2, which is actually a modified structure.
    model = MobileNetV2(n_class=1000, **kwargs)
    return model

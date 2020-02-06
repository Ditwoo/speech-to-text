import math
import torch
import torch.nn as nn
from torchvision.models import resnet18
from .utils import (
    GELU,
)


class LightConv(nn.Module):
    def __init__(self, 
                 feat_dim: int, 
                 num_classes: int):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        feats = self.conv_shapes(feat_dim, 20, 1, 41, 2)
        feats = self.conv_shapes(feats, 10, 1,21, 2)
        feats *= 32
        self.head = nn.Sequential(
            nn.Linear(feats, num_classes * 4),
            GELU(),
            nn.BatchNorm1d(num_classes * 4),
            nn.Linear(num_classes * 4, num_classes),
        )
    
    @staticmethod
    def conv_shapes(shapes: int, 
                    padding: int, 
                    dilation: int, 
                    kernel_size: int, 
                    stride: int) -> int:
        out_shapes = int((shapes + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        return out_shapes

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.convs.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.long()

    def forward(self, features, lengths):
        """
        :param sequences: float tensor with shapes (batch size, input_dim, sequence length)
        :return: tensor with shapes (batch size, sequnce length, num_classes)
        """
        out_lenghts = self.get_seq_lens(lengths)

        x = features.unsqueeze(1)  # (batch, 1, feats, seq)
        x = self.convs(x)
        b, c, f, s = x.shape
        x = x.view(b, c * f, s).transpose(1, 2).reshape(b * s, c * f)
        x = self.head(x)
        x = x.view(s, b, -1)

        return x, out_lenghts


if __name__ == "__main__":
    feat_dim = 161

    inp = torch.randn(3, feat_dim, 384)
    lengths = torch.IntTensor([328 // 3, 328, 328 // 2])

    m = LightConv(feat_dim, 35)

    out, sizes = m(inp, lengths)

    print(sizes)

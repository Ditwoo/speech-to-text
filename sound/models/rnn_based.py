import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    SequenceWise,
    InferenceBatchSoftmax,
)

class RNNLookahead(nn.Module):
    def __init__(self, feat_dim: int, context: int):
        super(RNNLookahead, self).__init__()

        if context <= 0:
            raise ValueError("'context' shold be greater than zero!")
        
        self.feat_dim = feat_dim
        self.context = context
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            in_channels=self.feat_dim,
            out_channels=self.feat_dim,
            kernel_size=self.context,
            stride=1,
            groups=self.feat_dim,
            padding=0,
            bias=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        :param features: float tensor with shapes: <batch> * <sequence> * <features>
        :return: float tensor with shapes
        """

        x = x.transpose(1, 2)  # swap sequence and feature dims
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2)  # respose shapes back
        return x


class LightLSTM(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 64, 
                 num_lstms: int = 3,
                 dropout_rate: float = 0.2, 
                 num_classes: int = 1):
        super(LightLSTM, self).__init__()
        self.lstms = nn.LSTM(
            input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_lstms,
            batch_first=True,
            bias=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        self.clsf_head = SequenceWise(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, num_classes * 4),
                nn.ReLU(),
                nn.BatchNorm1d(num_classes * 4),
                nn.Dropout(dropout_rate),
                nn.Linear(num_classes * 4, num_classes),
                # InferenceBatchSoftmax(),
                nn.LogSoftmax(1),
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        :param sequences: float tensor with shapes (batch size, input_dim, sequence length)
        :return: tensor with shapes (sequence len, batch size, num classes)
        """

        # import pdb; pdb.set_trace()

        bs, ss = features.size(0), features.size(2)
        x, _ = self.lstms(features.transpose(1, 2))
        x = self.clsf_head(x)  # (batch size, seq len, features)
        x = x.transpose(0, 1)  # (seq len, batch size, num classes)
        return x


class LookaheadLSTM(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 dropout_rate: float = 0.2,
                 context: int = 10,
                 num_classes: int = 1):
        super(LookaheadLSTM, self).__init__()

        self.lstms = nn.LSTM(
            input_dim, 
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            bias=True,
            dropout=dropout_rate,
            bidirectional=False,
        )
        self.lookahed = RNNLookahead(
            feat_dim=hidden_dim, 
            context=context
        )
        self.clsf_head = SequenceWise(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, num_classes * 4),
                nn.ReLU(),
                nn.BatchNorm1d(num_classes * 4),
                nn.Dropout(dropout_rate),
                nn.Linear(num_classes * 4, num_classes),
                InferenceBatchSoftmax(),
            )
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        :param sequences: float tensor with shapes (batch size, input_dim, sequence length)
        :return: tensor with shapes (sequence len, batch size, num classes)
        """
        bs, ss = features.size(0), features.size(2)
        x, _ = self.lstms(features.transpose(1, 2))
        x_lookahead = self.lookahed(x)
        x = torch.cat([x, x_lookahead], dim=-1)
        x = self.clsf_head(x)  # (batch size, seq len, features)
        x = x.transpose(0, 1)  # (seq len, batch size, num classes)
        return x



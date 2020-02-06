import math
from collections import OrderedDict
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    SequenceWise,
    InferenceBatchSoftmax,
    OverLastDim,
    GELU,
)


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = {v: k for k, v in supported_rnns.items()}


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)

        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class BatchRNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 rnn_type: torch.nn.Module = nn.LSTM, 
                 bidirectional: bool = False, 
                 batch_norm: bool =True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(
            input_size=input_size, 
            hidden_size=hidden_size,
            bidirectional=bidirectional, 
            bias=True
        )
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, 
                 n_features: int, 
                 context: int):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features, self.n_features, 
            kernel_size=self.context, 
            stride=1,
            groups=self.n_features, 
            padding=0, 
            bias=None
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


def _dense_layer(in_f: int, out_f: int, relu_clip: float = 0.0, dropout_rate: float = 0.0) -> nn.Module:
    layers = [nn.Linear(in_f, out_f)]
    if relu_clip > 0.0:
        layers.append(
            nn.Hardtanh(0, relu_clip, inplace=True)
            # GELU()
        )
    if dropout_rate > 0.0:
        layers.append(nn.Dropout(p=dropout_rate))
    return nn.Sequential(*layers)


def _lstm_layer(input_dim: int, hidden_dim: int, forget_gate_bias: float = None, num_layers: int = 1):
    lstm = nn.LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=True,
        # dropout=dropout_rate,
    )
    if forget_gate_bias is not None:
        for name in ['bias_ih_l0', 'bias_ih_l0_reverse']:
            bias = getattr(lstm, name)
            bias.data[hidden_dim:2 * hidden_dim].fill_(forget_gate_bias)
        for name in ['bias_hh_l0', 'bias_hh_l0_reverse']:
            bias = getattr(lstm, name)
            bias.data[hidden_dim:2 * hidden_dim].fill_(0)
    return lstm


class DeepSpeech(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_classes: int = 1,
                 dropout_rate: float = 0.1,
                 relu_clip: float = 20.0,
                 forget_gate_bias: float = 1.0):
        super(DeepSpeech, self).__init__()

        self._relu_clip = relu_clip
        self._dropout_rate = dropout_rate

        
        self.input_linears = OverLastDim(
            nn.Sequential(
                OrderedDict([
                    ("fc1", _dense_layer(input_dim, hidden_dim, relu_clip, dropout_rate=dropout_rate)),
                    ("fc2", _dense_layer(hidden_dim, hidden_dim, relu_clip, dropout_rate=dropout_rate)),
                    ("fc3", _dense_layer(hidden_dim, 2 * hidden_dim, relu_clip, dropout_rate=dropout_rate)),
                ])
            )
        )

        self.bi_lstm = _lstm_layer(2 * hidden_dim, hidden_dim, forget_gate_bias)

        self.head_linears = OverLastDim(
            nn.Sequential(
                OrderedDict([
                    ("out_fc_1", _dense_layer(2 * hidden_dim, hidden_dim, relu_clip, dropout_rate=dropout_rate)),
                    ("logits", _dense_layer(hidden_dim, num_classes)),
                    ("log_softmax", nn.LogSoftmax(1)),  # over last dimension
                ])
            )
        )
    
    def forward(self, features: torch.LongTensor) -> torch.FloatTensor:
        """
        :param sequences: float tensor with shapes (batch size, input_dim, sequence length)
        :return: tensor with shapes (sequence len, batch size, num classes)
        """

        """
        Computes a single forward pass through the network.
        Args:
            x: A tensor of shape (batch, in_features, sequence).

        Returns:
            LogSoftmax of shape (sequence, batch, num_classes).
        """
        bs, ss = features.size(0), features.size(2)
        features = features.transpose(1, 2)  # (batch, sequence, input_dim)
        x = self.input_linears(features)
        self.bi_lstm.flatten_parameters()
        x, _ = self.bi_lstm(x)
        out = self.head_linears(x)  
        out = out.transpose(0, 1)  # (sequence, batch, num_classes)
        return out


class DeepSpeechV2(nn.Module):
    def __init__(self, 
                 rnn_type: str = "lstm", 
                 num_classes: int = 1, 
                 rnn_hidden_size: int = 768, 
                 nb_layers: int = 5, 
                 audio_sample_rate: int = 16_000,
                 audio_window_size: float = 0.05,
                 bidirectional: bool = True, 
                 context: int = 20):
        super(DeepSpeechV2, self).__init__()
        rnn_type = supported_rnns.get(rnn_type, supported_rnns["lstm"])

        self.bidirectional = bidirectional
        sample_rate = audio_sample_rate
        window_size = audio_window_size

        self.conv = MaskConv(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(
            input_size=rnn_input_size, 
            hidden_size=rnn_hidden_size, 
            rnn_type=rnn_type,
            bidirectional=bidirectional, 
            batch_norm=False
        )
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(
                input_size=rnn_hidden_size, 
                hidden_size=rnn_hidden_size, 
                rnn_type=rnn_type,
                bidirectional=bidirectional
            )
            rnns.append(('%d' % (x + 1), rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(
                n_features=rnn_hidden_size, 
                context=context
            ),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    def forward(self, features: torch.FloatTensor, lengths: torch.IntTensor) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        :param x: float tensor with shapes (batch size, input_dim, sequence length)
        :param lengths: long tensor with sequence length of sound
        """
        x = features.unsqueeze(1)  # (b, 1, window_size, seqlen)
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        # x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths



if __name__ == "__main__":
    m = DeepSpeech(128, 256, 35, 0.2)
    inp = torch.randn(3, 128, 5)
    out = m(inp)
    print(out.shape)
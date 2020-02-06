import torch
from catalyst.dl import (
    Callback, 
    CallbackOrder, 
    RunnerState,
)
from typing import List
from sound.decoder import GreedyDecoder, BeamCTCDecoder
from sound.datasets.constants import TOKENIZER


class CharErrorRateCallback(Callback):
    """
    Combined callback for Char Error Rate (CER) and Word Error Rate (WER).
    """

    def __init__(self, 
                 prefix: str = "CER",
                 blank_char: str = " ",
                 input_key: str = "targets_strs",
                 output_key: str = "logits",
                 **metric_parameters):
        super().__init__(CallbackOrder.Metric)

        self.output_key = output_key
        self.input_key = input_key
        self.metric_parameters = metric_parameters
        self.blank_char = blank_char
        self.prefix = prefix
        
        self.decoder = GreedyDecoder(TOKENIZER.index_to_token)  # using fastest decoder

    def on_batch_end(self, state: RunnerState) -> None:
        outputs = state.output[self.output_key].transpose(0, 1)  # shapes: (batch, seq, num_classes)
        # outputs = torch.argmax(outputs, dim=2)  # most probable class
        outputs = outputs.detach().cpu()

        decoded_text, _ = self.decoder.decode(outputs)

        texts: List[str] = state.input[self.input_key]

        cer_cnt, num_chars = 0, 0
        for idx, tgt_txt in enumerate(texts):
            pred_tgt = decoded_text[idx][0].replace(self.blank_char, "")
            cer_cnt += self.decoder.cer(pred_tgt, tgt_txt)
            num_chars += len(tgt_txt.replace(self.blank_char, ""))

        try:
            cer = cer_cnt / num_chars
        except ZeroDivisionError:
            cer = 1

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: cer,
        })

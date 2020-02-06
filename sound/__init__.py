from catalyst.dl import registry

from .runner import Runner
from .experiment import Experiment
from .callbacks import CharErrorRateCallback
from .optimizers import SWA
from .models import (
    LightLSTM,
    DeepSpeech,
    DeepSpeechV2,
    LightConv,
    LookaheadLSTM,
)


registry.Callback(CharErrorRateCallback)

registry.Model(LightLSTM)
registry.Model(DeepSpeech)
registry.Model(DeepSpeechV2)
registry.Model(LightConv)
registry.Model(LookaheadLSTM)

registry.Optimizer(SWA)

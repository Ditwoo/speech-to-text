import os
from catalyst.dl import SupervisedRunner


if int(os.environ.get("MODEL_OUTPUTS_LENGTH", 0)):
    class Runner(SupervisedRunner):
        def __init__(self, model=None, device=None):
            super().__init__(
                model=model, 
                device=device, 
                input_key=["features", "lengths"],
                output_key=["log_probs", "input_lengths"],
                input_target_key="targets",
            )
else:
    class Runner(SupervisedRunner):
        def __init__(self, model=None, device=None):
            super().__init__(
                model=model, 
                device=device, 
                input_key="features",
                output_key="log_probs",
                input_target_key="targets",
            )

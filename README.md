
## Speech to text with catalyst

Data used for testing:
- `asr_public_stories_2` [__mp3+txt__](https://ru-open-stt.ams3.digitaloceanspaces.com/asr_public_stories_2_mp3.tar.gz) from [this repo](https://github.com/snakers4/open_stt/#links).



## How to run

Create shell file with structure like this:

```bash
#!/bin/bash

CONF=<path to config file>
LOGDIR=<path to log dir>
# num timesteps before model == num timesteps after model
export MODEL_OUTPUTS_LENGTH=0
# if do not need to inform about training process please remove
# appropriate callback from configuration file
export CATALYST_TELEGRAM_TOKEN="<bot token>"
export CATALYST_TELEGRAM_CHAT_ID="<chat id>"

[ -e ${LOGDIR} ] && rm -rf ${LOGDIR} && echo " * Removed already existed logdir - '${LOGDIR}'"
catalyst-dl run --expdir sound --config ${CONF} --logdir ${LOGDIR} --verbose
```




### Implemented models:

- [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/pdf/1412.5567.pdf)
- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf)
- [Lookahead Convolution Layer For Unidirectional Recurrent Neural Networks](https://openreview.net/pdf?id=91EowxONgIkRlNvXUVog)

import os
import json
import torch
from hifigan_fix.env import AttrDict
from hifigan_fix.meldataset import mel_spectrogram
from hifigan_fix.models import Generator
from hifigan_fix.utils import load_checkpoint

h = None
device = None
generator = None


def initialize_hg(config_name, checkpoint_name):
    print('Initializing Inference Process...')

    global h
    hg_dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(hg_dir_path, config_name)) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    global device
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    global generator
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(os.path.join(hg_dir_path, 'checkpoints', checkpoint_name), device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    return h, device


def get_mel_torch(wave):
    wave = torch.FloatTensor(wave).to(device)
    return mel_spectrogram(wave, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def inference_hg(mel):
    return generator(mel).squeeze(1)

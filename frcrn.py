# https://github.com/modelscope/modelscope/blob/master/modelscope/pipelines/audio/ans_pipeline.py
# https://github.com/modelscope/modelscope/blob/master/modelscope/models/audio/ans/frcrn.py
import numpy as np
import torch
from modelscope.models import Model
from modelscope.utils.audio.audio_utils import audio_norm

_device = None
model = None
window = 16000
stride = int(window * 0.75)


def padding(wave, nsamples):
    if nsamples <= window:
        pad_len = window - nsamples
    elif (nsamples - window) % stride != 0:
        pad_len = window - nsamples + ((nsamples - window) // stride + 1) * stride
    else:
        pad_len = 0
    if pad_len:
        wave = np.concatenate((wave, np.zeros((len(wave), pad_len), np.float32)), 1)
    return wave


def initialize_frcrn(device, nsamples):
    global _device, model
    _device = device
    model = Model.from_pretrained('damo/speech_frcrn_ans_cirm_16k').model.to(device).eval()
    with torch.no_grad():
        wave = np.random.random_sample((nsamples,)).astype(np.float32)
        wave = padding(wave[None], nsamples)
        wave = torch.from_numpy(wave).to(device)
        model = torch.jit.trace(model, wave, strict=False)
        model = torch.jit.freeze(model)


def denoise(wave):
    scale = np.amax(wave)
    wave = audio_norm(wave)
    scale /= np.amax(wave)
    nsamples = len(wave)
    wave = padding(wave[None], nsamples)
    wave = torch.from_numpy(wave).to(_device)
    wave = model(wave)[4][0].cpu().numpy()
    return wave[:nsamples] * scale

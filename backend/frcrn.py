import numpy as np
import torch
from modelscope.models import Model

_device = None
model = None
window = 16000
stride = int(window * 0.75)


def audio_norm(x):
    rms = np.sqrt(np.mean(x**2))
    if rms == 0:
        scalar = 0
    else:
        scalar = 10 ** (-25 / 20) / rms
    x = x * scalar
    pow_x = x**2
    avg_pow_x = np.mean(pow_x)
    if avg_pow_x == 0:
        rmsx = 0
    else:
        rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5
    return x, rmsx


def padding(wave, nsamples):
    if nsamples <= window:
        pad_len = window - nsamples
    elif (nsamples - window) % stride != 0:
        pad_len = window - nsamples + ((nsamples - window) // stride + 1) * stride
    else:
        pad_len = 0
    if pad_len:
        wave = np.concatenate((wave, np.zeros((pad_len,), dtype=np.float32)))
    return wave


def initialize_frcrn(device, nsamples):
    global _device, model
    _device = device
    model = (
        Model.from_pretrained("damo/speech_frcrn_ans_cirm_16k").model.to(device).eval()
    )
    with torch.no_grad():
        wave = np.random.random_sample((nsamples,)).astype(np.float32)
        wave = padding(wave, nsamples)
        wave = torch.from_numpy(wave[None]).to(device)
        model = torch.jit.trace(model, wave, strict=False)
        model = torch.jit.freeze(model)


def denoise(wave):
    scale = np.amax(wave)
    wave, _ = audio_norm(wave)
    scale /= np.amax(wave)
    nsamples = len(wave)
    wave = padding(wave, nsamples)
    wave = torch.from_numpy(wave[None]).to(_device)
    wave = model(wave)[4][0].cpu().numpy()
    return wave[:nsamples] * scale


# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_frcrn(device, 16000)
    input_wave = np.random.random_sample((16000,)).astype(np.float32)
    denoised_wave = denoise(input_wave)
    print(denoised_wave)

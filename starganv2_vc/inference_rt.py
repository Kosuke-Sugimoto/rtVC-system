import os
import librosa
import numpy as np
import torch
import yaml
from munch import Munch
import const
from hifigan_fix.meldataset import mel_spectrogram
from starganv2_vc.Utils.JDC.model import JDCNet
from starganv2_vc.models import Generator, MappingNetwork, StyleEncoder

_h = None
_device = None
F0_model = None
starganv2 = None
reference_embeddings = None
min_len_wave = 24000


def build_model(model_params):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    nets_ema = Munch(generator=generator, mapping_network=mapping_network, style_encoder=style_encoder)
    return nets_ema


def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == '':
            label = torch.LongTensor([speaker]).to(_device)
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to(_device), label)
        else:
            wave, sr = librosa.load(path, sr=24000, res_type='soxr_vhq')
            wave, index = librosa.effects.trim(wave, top_db=30)
            if len(wave) < min_len_wave:
                wave = np.pad(wave, (min_len_wave - len(wave)) // 2)
            if len(wave) < min_len_wave:
                wave = np.pad(wave, (0, 1))
            wave_tensor = torch.from_numpy(wave).float()
            mel_tensor = mel_spectrogram(wave_tensor.unsqueeze(0), _h.n_fft, _h.num_mels, _h.sampling_rate,
                                         _h.hop_size, _h.win_size, _h.fmin, _h.fmax).to(_device)
            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    return reference_embeddings


def initialize_vc(h, device, model_dir, model_name, f0_model, f0_model_key):
    global _h, _device
    _h, _device = h, device
    vc_dir_path = os.path.dirname(os.path.abspath(__file__))

    # load F0 model
    global F0_model
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(os.path.join(vc_dir_path, 'Utils', 'JDC', f0_model))[f0_model_key]
    F0_model.load_state_dict(params)
    _ = F0_model.eval()
    F0_model = F0_model.to(device)

    # load starganv2
    global starganv2
    model_path = os.path.join(vc_dir_path, 'Models', model_dir, model_name)
    with open(os.path.join(vc_dir_path, 'Configs', 'config.yml')) as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = build_model(model_params=starganv2_config['model_params'])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to(device)
    starganv2.mapping_network = starganv2.mapping_network.to(device)
    starganv2.generator = starganv2.generator.to(device)

    # with reference, using style encoder
    global reference_embeddings
    speaker_dicts = {}
    for s in const.speakers:
        for r in range(1, const.references + 1):
            speaker_dicts[f'{s}{r:03}'] = (os.path.join(vc_dir_path, 'Data', 'ITA-corpus', s, f'recitation{r:03}.wav'),
                                           const.speakers.index(s) + 1)
    reference_embeddings = compute_style(speaker_dicts)


def conversion(mel, ref_emb_key):
    f0_feat = F0_model.get_feature_GAN(mel.unsqueeze(1))
    out = starganv2.generator(mel.unsqueeze(1), reference_embeddings[ref_emb_key][0], F0=f0_feat)
    return out.squeeze(1)

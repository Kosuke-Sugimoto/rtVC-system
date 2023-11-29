import itertools
import socket
import struct
import threading
import numpy as np
import soxr
import torch
import torch.backends.cudnn
import const
from frcrn import initialize_frcrn, denoise
from hifigan_fix.inference_rt import initialize_hg, get_mel_torch, inference_hg
from starganv2_vc.inference_rt import initialize_vc, conversion
from utils import net

torch.backends.cudnn.benchmark = True
nwarmup = 50

stft_factor = 300
down_factor = 2
inout_samplerate = 48000
denoise_samplerate = 16000
inter_samplerate = inout_samplerate // down_factor
trim_pad_hg = 3
inter_size_hg = 8
inout_size_hg = inter_size_hg * stft_factor * down_factor
inter_size_vc = 80
inout_size_vc = inter_size_vc * stft_factor * down_factor
assert inter_size_vc >= inter_size_hg + trim_pad_hg * 2, 'Segment size mismatch'
hg_start = inter_size_vc - trim_pad_hg - inter_size_hg
hg_end = inter_size_vc - trim_pad_hg
vad_threshold = 0.00005  # 要調整
# vad_threshold = 0.00001  # 要調整
sil_threshold = 8

ref_emb_key = 'zundamon127'


def wait_req():
    global ref_emb_key
    while True:
        req = net.recvall(conn_ctrl, 2)
        if not req:
            break
        command = struct.unpack('!h', req)[0]
        if command == 0:
            ref_num = int(ref_emb_key[-3:])
            if ref_num != const.references:
                ref_num += 1
            else:
                ref_num = 1
            ref_emb_key = f'{ref_emb_key[:-3]}{ref_num:03}'
            print(ref_emb_key)
        elif 0 < command <= const.references:
            ref_emb_key = f'{ref_emb_key[:-3]}{command:03}'
            print(ref_emb_key)
        elif -len(const.speakers) <= command < 0:
            ref_emb_key = f'{const.speakers[-command - 1]}{ref_emb_key[-3:]}'
            print(ref_emb_key)


def initialize():
    # h, device = initialize_hg('config_v1_mod.json', 'g_01155000')
    h, device = initialize_hg('config_v1_mod_2.json', 'g_07180000_2')
    # initialize_vc(h, device, 'ita4jvs20_pre_en', 'epoch_00294.pth', 'bst.t7', 'net')
    initialize_vc(h, device, 'ita4jvs20_pre_alljp', 'epoch_00294.pth', 'ep50_200bat32lr5_alljp.pth', 'model')
    initialize_frcrn(device, int(inout_size_vc / inout_samplerate * denoise_samplerate))

    print('Warm up...')
    with torch.no_grad():
        for _ in range(nwarmup):
            input_wave = np.random.random_sample((inout_size_vc,)).astype(np.float32)
            input_wave = soxr.resample(input_wave, inout_samplerate, denoise_samplerate, 'VHQ')
            input_wave = denoise(input_wave)
            input_wave = soxr.resample(input_wave, denoise_samplerate, inter_samplerate, 'VHQ')
            input_mel = get_mel_torch(input_wave[None])
            output_mel = conversion(input_mel, ref_emb_key)
            output_wave = inference_hg(output_mel[..., hg_start:hg_end]).cpu().numpy()[0]
            output_wave = soxr.resample(output_wave, inter_samplerate, inout_samplerate, 'VHQ')
    print('Done.')


def main():
    thread = threading.Thread(target=wait_req)
    thread.start()

    initialize()
    sample_size = buffer_size // 4
    block_count = (inout_size_hg // 2) // sample_size
    assert block_count == (inout_size_hg / 2) / sample_size, 'Buffer size mismatch'
    hann_win = np.hanning(inout_size_hg - 1).astype(np.float32)
    hann_win_prev, hann_win = hann_win[inout_size_hg // 2 - 1:], hann_win[:inout_size_hg // 2]

    conn_ctrl.sendall(struct.pack('!H', 0))
    with torch.no_grad():
        input_buffer = np.zeros(inout_size_vc, np.float32)
        output_buffer = np.zeros(inout_size_hg // 2, np.float32)
        output_wave_prev = np.zeros(inout_size_hg, np.float32)
        sil_count = 0
        for i in itertools.cycle(range(block_count)):
            indata = net.recvall(conn_data, buffer_size)
            if len(indata) != buffer_size:
                break
            input_buffer = np.concatenate((input_buffer[sample_size:], np.frombuffer(indata, np.float32)))
            conn_data.sendall(output_buffer[sample_size * i:sample_size * (i + 1)].tobytes())
            if i == block_count - 1:
                input_wave = soxr.resample(input_buffer, inout_samplerate, denoise_samplerate, 'VHQ')
                input_wave = denoise(input_wave)
                input_wave = soxr.resample(input_wave, denoise_samplerate, inter_samplerate, 'VHQ')
                input_mel = get_mel_torch(input_wave[None])
                output_mel = conversion(input_mel, ref_emb_key)
                output_wave = inference_hg(output_mel[..., hg_start:hg_end]).cpu().numpy()[0]
                if np.average(np.power(input_wave[hg_start * stft_factor:hg_end * stft_factor], 2)) < vad_threshold:
                    sil_count += 1
                    output_wave /= sil_count
                else:
                    sil_count = 0
                if sil_count > sil_threshold:
                    output_wave.fill(0.0)
                output_wave = soxr.resample(output_wave, inter_samplerate, inout_samplerate, 'VHQ')
                output_buffer = output_wave_prev[inout_size_hg // 2:] * hann_win_prev \
                                + output_wave[:inout_size_hg // 2] * hann_win
                output_wave_prev = output_wave

    thread.join()


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_ctrl, \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_data:
        net.init_sock(sock_ctrl, const.PORT_CTRL)
        net.init_sock(sock_data, const.PORT_DATA)
        with sock_ctrl.accept()[0] as conn_ctrl, sock_data.accept()[0] as conn_data:
            buffer_size = struct.unpack('!H', net.recvall(conn_ctrl, 2))[0]
            main()

import socket
import struct
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import const
from utils import net

# print(sd.query_devices())  # デバイス一覧取得
sd.default.device = 'MOTU M Series, ASIO'  # Buffer: 128 samples
# sd.default.device = 'Focusrite USB ASIO, ASIO'  # Buffer: 128 samples
# sd.default.device = 'US-2x2 & US-4x4, ASIO'  # Buffer: 128 samples
# sd.default.device = ('マイク (NVIDIA Broadcast), Windows WASAPI', 'スピーカー (2- US-2x2), Windows WASAPI')
sd.default.channels = 1
sd.default.samplerate = 48000.0
sd.default.latency = 'low'
sd.default.blocksize = 300
buffer_size = 4 * sd.default.blocksize
buffer_num = 8  # 要調整

indata_record = bytearray()
outdata_record = bytearray()


def write_wav(name, rate, data, outdtype):  # input: float, [-1.0, (1.0)]
    if data.dtype != outdtype:
        if outdtype.kind == 'f':
            data = data.astype(outdtype)
        else:
            middtype = np.promote_types(data.dtype, outdtype)
            if data.dtype != middtype:
                data = data.astype(middtype)
            scale = np.iinfo(outdtype).max + 1
            if outdtype.kind == 'u':
                scale /= 2
                data += 1.0
            data = np.rint(data * scale).clip(np.iinfo(outdtype).min, np.iinfo(outdtype).max).astype(outdtype)
    wavfile.write(name, rate, data)


def sd_callback(indata, outdata, frames, time, status):
    global buffer_num
    conn_data.sendall(indata)
    indata_record.extend(indata)
    if buffer_num > 0:
        data = bytearray(buffer_size)
        buffer_num -= 1
    else:
        data = net.recvall(conn_data, buffer_size)
    outdata_record.extend(data)
    outdata[:] = data


def main():
    net.recvall(conn_ctrl, 2)
    buffer_num_bak = buffer_num
    with sd.RawStream(callback=sd_callback):
        try:
            while True:
                command = input()
                if command:
                    command = int(command)
                else:
                    command = 0
                conn_ctrl.sendall(struct.pack('!h', command))
        except ValueError:
            pass
    indata_record_np = np.frombuffer(indata_record, np.float32)
    outdata_record_np = np.frombuffer(outdata_record, np.float32)
    indata_record_np /= np.amax(np.abs(indata_record_np))
    outdata_record_np /= np.amax(np.abs(outdata_record_np))
    write_wav('input.wav', 48000, indata_record_np, np.dtype('i2'))
    write_wav('output.wav', 48000, outdata_record_np, np.dtype('i2'))
    for _ in range(buffer_num_bak):
        net.recvall(conn_data, buffer_size)


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn_ctrl, \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn_data:
        conn_ctrl.connect((const.HOST, const.PORT_CTRL))
        conn_data.connect((const.HOST, const.PORT_DATA))
        conn_ctrl.sendall(struct.pack('!H', buffer_size))
        main()

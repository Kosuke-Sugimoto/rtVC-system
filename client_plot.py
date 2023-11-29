import os
import socket
import struct
import subprocess
import sys
import sounddevice as sd
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


def sd_callback(indata, outdata, frames, time, status):
    global buffer_num
    conn_data.sendall(indata)
    conn_plot.sendall(indata)
    if buffer_num > 0:
        data = bytearray(buffer_size)
        buffer_num -= 1
    else:
        data = net.recvall(conn_data, buffer_size)
    outdata[:] = data
    conn_plot.sendall(data)


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
    for _ in range(buffer_num_bak):
        net.recvall(conn_data, buffer_size)


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_plot, \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn_ctrl, \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn_data:
        net.init_sock(sock_plot, const.PORT_PLOT)
        p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), const.PLOT_PY)])
        with sock_plot.accept()[0] as conn_plot:
            conn_plot.sendall(struct.pack('!H', buffer_size))
            conn_ctrl.connect((const.HOST, const.PORT_CTRL))
            conn_data.connect((const.HOST, const.PORT_DATA))
            conn_ctrl.sendall(struct.pack('!H', buffer_size))
            main()
        p.wait()

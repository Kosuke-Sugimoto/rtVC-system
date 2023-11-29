import socket
import struct
import threading
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import const
from utils import net

len_wave_plot = 2 * 48000
indata_buffer = np.zeros(len_wave_plot, np.float32)
outdata_buffer = np.zeros(len_wave_plot, np.float32)


def update():
    global indata_buffer, outdata_buffer
    while True:
        indata = net.recvall(conn_plot, buffer_size)
        if not indata:
            break
        indata_buffer = np.concatenate((indata_buffer[blocksize:], np.frombuffer(indata, np.float32)))
        outdata = net.recvall(conn_plot, buffer_size)
        outdata_buffer = np.concatenate((outdata_buffer[blocksize:], np.frombuffer(outdata, np.float32)))


def plot():
    def init_plot(index, buffer, title):
        p_wave, = axs[index].plot(buffer, linewidth=0.5)
        axs[index].set_xlim(0, len(buffer) - 1)
        axs[index].set_ylim(-1.1, 1.1)
        axs[index].set_xticks([])
        axs[index].set_title(title)
        return p_wave

    def animate(frame):
        p_wave_i.set_ydata(indata_buffer)
        p_wave_o.set_ydata(outdata_buffer)
        return [p_wave_i, p_wave_o]

    fig, axs = plt.subplots(2, num='ZRVC - Zundamon Realtime Voice Conversion')
    p_wave_i = init_plot(0, indata_buffer, 'Source')
    p_wave_o = init_plot(1, outdata_buffer, 'Converted')
    fig.tight_layout()
    ani = animation.FuncAnimation(fig, animate, interval=0, blit=True, cache_frame_data=False)
    plt.show()


def main():
    thread = threading.Thread(target=update)
    thread.start()
    plot()
    thread.join()


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn_plot:
        conn_plot.connect(('localhost', const.PORT_PLOT))
        buffer_size = struct.unpack('!H', net.recvall(conn_plot, 2))[0]
        blocksize = buffer_size // 4
        main()

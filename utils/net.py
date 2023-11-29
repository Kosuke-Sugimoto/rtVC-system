import socket


def init_sock(sock, port):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # TIME_WAIT対策
    sock.bind(('', port))
    sock.listen()


def recvall(sock, size):
    data = bytearray()
    while len(data) < size:
        buf = sock.recv(min(size - len(data), 4096))
        if not buf:
            break
        data.extend(buf)
    return data

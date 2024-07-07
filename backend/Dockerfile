FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    python3 \
    python3-pip \
    python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask \
    websockets \
    "numpy<2" \
    soxr \
    sounddevice \
    torch==2.0.0\
    torchaudio==2.0.1

WORKDIR /work

# CMD [ "python", "server.py" ]
CMD [ "bash" ]

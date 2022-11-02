FROM ubuntu:focal-20221019

# Ensure apt won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -yqq --no-install-recommends \
        curl \
        wget \
        pkg-config \
        build-essential \
        python3-pip \
        python3-dev \
        python3-numpy \
        python3-grpcio \
        python3-scipy \
        python3-numba \
        ca-certificates \
        libhdf5-dev \
        libffi-dev \
        libssl-dev \
        python3-paho-mqtt \
        portaudio19-dev \
        pulseaudio && \
        rm -rf /var/lib/apt/lists/* && \
        wget https://images.getsmarter.io/ml-models/audio-client-models.tar.gz && \
        tar -xvzf audio-client-models.tar.gz && \
        rm audio-client-models.tar.gz

RUN python3 -m pip install --upgrade \
        wheel \
        setuptools \
        tritonclient[all] \
        resampy \
        pyaudio \
        requests

COPY *.py vggish_pca_params.npz *.classes *.pbtxt ./

CMD [ "bash" ]



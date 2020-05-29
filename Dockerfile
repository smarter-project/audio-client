FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            autoconf \
            automake \
            build-essential \
            cmake \
            curl \
            git \
            libb64-dev \
            libopencv-dev \
            libopencv-core-dev \
            libssl-dev \
            libtool \
            pkg-config \
            python3 \
            python3-pip \
            python3-dev \
            rapidjson-dev && \
    pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade grpcio-tools

# Build expects "python" executable (not python3).
RUN rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
RUN git clone -b r20.05 https://github.com/NVIDIA/triton-inference-server
RUN cd triton-inference-server/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX:PATH=/workspace/install \
            -DTRTIS_ENABLE_GRPC_V2=ON \
            -DTRTIS_ENABLE_HTTP_V2=ON && \
    make -j$(getconf _NPROCESSORS_ONLN) trtis-clients

RUN pip3 install --upgrade  \
            install/python/tensorrtserver*.whl \
            install/python/triton*.whl \
            numpy resampy requests

ENV PATH //workspace/install/bin:${PATH}
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends ca-certificates portaudio19-dev pulseaudio \
    libhdf5-dev python3-opencv python3-paho-mqtt && \
    pip3 install pyaudio && \
    rm -rf /var/lib/apt/lists/*

COPY samples /samples
COPY *.py vggish_pca_params.npz ./

CMD [ "bash" ]

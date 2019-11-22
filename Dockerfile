FROM python:3.6-slim

RUN apt update &&\
    apt install -yqq --no-install-recommends portaudio19-dev pulseaudio build-essential && \
    pip3 install --no-cache-dir paho-mqtt requests pyaudio jaeger-client opentracing_instrumentation && \
    rm -rf /var/lib/apt/lists/* && \
    apt purge -yqq build-essential


COPY audio-record.py audio-record.py
COPY client.conf /etc/pulse/client.conf
ENTRYPOINT [ "python3", "audio-record.py" ]

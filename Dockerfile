FROM python:3.7-alpine

RUN apk update && \
    apk add --no-cache --virtual .build-deps alpine-sdk && \
    apk add --no-cache portaudio-dev pulseaudio pulseaudio-alsa alsa-plugins-pulse && \
    pip3 install --no-cache-dir paho-mqtt requests pyaudio jaeger-client opentracing_instrumentation && \
    apk del .build-deps

COPY samples /samples
COPY audio-record.py audio-record.py
COPY client.conf /etc/pulse/client.conf
ENTRYPOINT [ "python3", "audio-record.py" ]

import sys
import os
import wave
import requests
import pyaudio
import signal
import paho.mqtt as mqtt
import paho.mqtt.publish as publish
import logging
import json
from time import sleep

# Set env variables
DEMO = os.getenv('DEMO', '')
HOSTNAME = os.getenv('CLASSIFY_SERVICE_HOST', 'sound-classifier')
PORT = str(os.getenv('CLASSIFY_SERVICE_PORT', '5000'))
SOUND_POLL_FREQUENCY = int(os.getenv('CLASSIFY_SERVICE_POLL_FREQUENCY', 10))
MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST', 'mqtt-debug')
TOPIC = os.getenv('TOPIC', '/demo/sound_class')
RECORD_SECONDS = int(os.getenv('RECORD_SECONDS', 10))

# Set log level
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
if loglevel == 'info':
    logging.basicConfig(level=logging.INFO)
elif loglevel == 'warning':
    logging.basicConfig(level=logging.WARN)
elif loglevel == 'debug':
    logging.basicConfig(level=logging.DEBUG)

def record_clip(seconds):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 16000

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

    frames = []

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        frames.append(data)

        if len(frames) == int(RATE * seconds / CHUNK):
            waveFile = wave.open('current.wav', 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            return

def classify_sound(file_path):
    # Post audio clip to sound classification api
    model_endpoint = 'http://{}:{}/model/predict'.format(HOSTNAME, PORT)

    with open(file_path, 'rb') as file:
        file_form = {'audio': (file_path, file, 'audio/wav')}
        try:
            r = requests.post(url=model_endpoint, files=file_form)
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            logging.info('Request to sound classifier failed with error: {}'.format(e))
            return

    if r.status_code != 200:
        logging.info('Request to sound classifier failed with status code: {}'.format(r.status_code))
        return

    response = r.json()

    if response['status'] != 'ok':
        logging.info('Error response from classifier: {}'.format(response['status']))
        return

    # Publish to mqtt
    result = publish.single(TOPIC, json.dumps(response), hostname=MQTT_BROKER_HOST)
    if result is mqtt.MQTT_ERR_SUCCESS:
        logging.info('Sound classified successfully and results published to mqtt')
    elif result is mqtt.MQTT_ERR_NO_CONN:
        logging.info('Sound classified successfully but mqtt not connected to server')
    elif result is mqtt.MQTT_ERR_QUEUE_SIZE:
        logging.info('Sound classified successfully but mqtt message message queue currently full')

def handler_stop_signals(signum, frame):
    sys.exit(0)

if __name__ == '__main__':
    while True:
        if DEMO:
            for file in os.listdir("/samples"):
                file_path = os.path.join("/samples", file)
                classify_sound(file_path)
                logging.debug('Clip {} classified'.format(file_path))
                sleep(SOUND_POLL_FREQUENCY)

        else:
            record_clip(RECORD_SECONDS)
            logging.debug('Clip recorded')
            classify_sound('current.wav')
            logging.debug('Clip classified')
            sleep(SOUND_POLL_FREQUENCY)

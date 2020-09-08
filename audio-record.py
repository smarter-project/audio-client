import sys
import os
import wave
import requests
import pyaudio
import signal
import paho.mqtt.publish as publish
import logging
import json
from time import sleep

# Set env variables
DEMO = os.getenv('DEMO', '')
FAAS = os.getenv('FAAS', '')
SQUASH_FUNCTION_URL = os.getenv('SQUASH_FUCTION_URL', 'http://edgefaas:8080/squasher')
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

def record_clip(stream, seconds):
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

    # Squash json
    if FAAS:
        try:
            faas_r = requests.post(url=SQUASH_FUNCTION_URL, json=response)
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            logging.info('Request to faas squash failed with error: {}'.format(e))
            return

        squashed_json = faas_r.json()
        
    else:
        # place holder squash function if faas unused
        dictionary = {}
        idx = 0

        for item in response['predictions']:
            lbl = "label" + str(idx)
            prb = "probability" + str(idx)
            dictionary[lbl] = item['label']
            dictionary[prb] = item['probability']
            idx += 1

        del response['predictions']
        response.update(dictionary)

        squashed_json = response
    
    # Published squashed json to mqtt
    try:
        publish.single(TOPIC, json.dumps(squashed_json), hostname=MQTT_BROKER_HOST)
        logging.info('Sound classified successfully and results published to mqtt')
    except Exception as e:
        logging.info('Sound classified successfully but mqtt publish failed with error: {}'.format(e))

def handler_stop_signals(signum, frame):
    sys.exit(0)

if __name__ == '__main__':
    if not DEMO:
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

    while True:
        if DEMO:
            for file in os.listdir("/samples"):
                file_path = os.path.join("/samples", file)
                classify_sound(file_path)
                logging.debug('Clip {} classified'.format(file_path))
                sleep(SOUND_POLL_FREQUENCY)

        else:
            record_clip(stream, RECORD_SECONDS)
            logging.debug('Clip recorded')
            classify_sound('current.wav')
            logging.debug('Clip classified')
            sleep(SOUND_POLL_FREQUENCY)

import sys
import os
import wave
import pyaudio
import signal
import paho.mqtt.publish as publish
import logging
import json
from time import sleep
import numpy as np
import vggish_input
import vggish_params
import vggish_postprocess
import argparse
from triton_client import *
from config import DEFAULT_PCA_PARAMS

def generate_embeddings(ctx, input_name, output_name, wav_file):
    """
    Generates embeddings as per the Audioset VGG-ish model.
    Post processes embeddings with PCA Quantization
    Input args:
        wav_file   = /path/to/audio/in/wav/format.wav
    Returns:
        An nparray of the same shape as the input but of type uint8,
          containing the PCA-transformed and quantized version of the input.
    """
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    result = ctx.run(
        { input_name : (examples_batch.astype(np.float32),) },
        { output_name : InferContext.ResultFormat.RAW })
    return vggish_postprocess.Postprocessor(DEFAULT_PCA_PARAMS).postprocess(result['vggish/embedding'][0]) # todo: turn this into a custom backend to create triton ensemble

def classifier_pre_process(embeddings, time_stamp):
    """
    Helper function to make sure input to classifier the model is of standard size.
    * Clips/Crops audio clips embeddings to start at time_stamp if not default and throws error if invalid
    * Augments audio embeddings shorter than 10 seconds (10x128 tensor) to a multiple of itself
    closest to 10 seconds.
    * Clips/Crops audio clips embeddings than 10 seconds to 10 seconds.
    * Converts dtype of embeddings from uint8 to float32

    Input args :
        embeddings = numpy array of shape (x,128) where x is any arbitrary whole number >1.
    Returns:
        embeddings = numpy array of shape (1,10,128), dtype=float32.
    """
    embeddings_ts = int(time_stamp / vggish_params.EXAMPLE_HOP_SECONDS)
    embeddings_len = embeddings.shape[0]
    if 0 < embeddings_ts < embeddings_len:
        end_ts = embeddings_ts + 10
        end_ts = end_ts if end_ts < embeddings_len else embeddings_len
        embeddings = embeddings[embeddings_ts:end_ts, :]
    elif embeddings_ts < 0 or embeddings_ts >= embeddings_len:
        raise ValueError

    embeddings_len = embeddings.shape[0]
    if embeddings_len < 10:
        while embeddings_len < 10:
            embeddings = np.stack((embeddings, embeddings))
            embeddings_len = embeddings.size / 128
        embeddings = embeddings.reshape((int(embeddings_len), 128))
    else:
        pass
    embeddings = embeddings[0:10, :].reshape([1, 10, 128])
    embeddings = uint8_to_float32(embeddings)
    return embeddings

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.

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

def classify_sound(classes, file_path, ctx_embedding, ctx_classify, input_name_embedding, 
                    output_name_embedding, input_name_classify, output_name_classify):
    raw_embeddings = generate_embeddings(ctx_embedding, input_name_embedding, output_name_embedding, file_path)
    embeddings_processed = classifier_pre_process(raw_embeddings, 0)

    result = ctx_classify.run(
        { input_name_classify : (embeddings_processed,) },
        { output_name_classify : (InferContext.ResultFormat.CLASS, classes) })
    
    # iterate through top CLASSES results and construct mqtt message
    msg = {}
    for idx, classification in enumerate(result['activation_4/Sigmoid'][0]):
        msg['label' + str(idx)] = classification[2]
        msg['probability' + str(idx)] = classification[1]
    logging.info(msg)

    # Publish msg to mqtt
    try:
        publish.single(TOPIC, json.dumps(msg), hostname=MQTT_BROKER_HOST)
        logging.info('Sound classified successfully and results published to mqtt')
    except Exception as e:
        logging.info('Sound classified successfully but mqtt publish failed with error: {}'.format(e))

def handler_stop_signals(signum, frame):
    # Close all                                                                                                                           
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sys.exit(0)

if __name__ == '__main__':

    # Set env variables
    MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST', 'mqtt-debug')
    TOPIC = os.getenv('TOPIC', '/demo/sound_class')

    # Set log level
    loglevel = os.getenv('LOG_LEVEL', 'info').lower()
    if loglevel == 'info':
        logging.basicConfig(level=logging.INFO)
    elif loglevel == 'warning':
        logging.basicConfig(level=logging.WARN)
    elif loglevel == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classes', type=int, required=False, default=os.getenv('CLASSES', 5),
                        help='Number of class results to report. Default is 5.')
    parser.add_argument('-m', '--model-name-classify', type=str, required=False, default=os.getenv('MODEL_NAME_CLASSIFY', 'ambient_sound_clf'),
                        help='Name of audio classification model')
    parser.add_argument('-e', '--model-name-embedding', type=str, required=False, default=os.getenv('MODEL_NAME_EMBEDDING', 'vggish'),
                        help='Name of embedding model')
    parser.add_argument('-x', '--model-version', type=int, required=False,
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-u', '--url', type=str, required=False, default=os.getenv('TRITON_URL', 'localhost:8000'),
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-p', '--sound-poll-freq', type=int, required=False, default=os.getenv('CLASSIFY_SERVICE_POLL_FREQUENCY', 10),
                        help='Sound poll frequency.')
    parser.add_argument('-r', '--record-secs', type=int, required=False, default=os.getenv('RECORD_SECONDS', 10),
                        help='Seconds to record. Default is 10')
    parser.add_argument('-d', '--use-clips', action="store_true")
    parser.add_argument('--audio_file_dir', type=str, required=False, default='/samples')

    args = parser.parse_args()

    protocol = ProtocolType.from_str('HTTP')

    # Fetch model information for embedding model from triton server
    input_name_embedding, output_name_embedding, format_embedding, dtype_embedding = parse_model(
        args.url, protocol, args.model_name_embedding, 1)

    # Fetch model information for classificaiton model from triton server
    input_name_classify, output_name_classify, format_classify, dtype_classify = parse_model(
        args.url, protocol, args.model_name_classify, 1)

    # Create embedding model context used to pass tensors to triton
    ctx_embedding = InferContext(args.url, protocol, args.model_name_embedding, args.model_version)

    # Create classification model context used to pass tensors to triton
    ctx_classify = InferContext(args.url, protocol, args.model_name_classify, args.model_version)

    if not args.use_clips:
        # Create pyaudio stream and start recording in background
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
        if args.use_clips:
            for file in os.listdir(args.audio_file_dir):
                file_path = os.path.join(args.audio_file_dir, file)
                classify_sound(args.classes, file_path, ctx_embedding, ctx_classify, input_name_embedding, 
                    output_name_embedding, input_name_classify, output_name_classify)
                logging.info('Clip {} classified'.format(file_path))
                sleep(args.sound_poll_freq)

        else:
            record_clip(stream, args.record_secs)
            logging.info('Clip recorded')
            classify_sound(args.classes, 'current.wav', ctx_embedding, ctx_classify, input_name_embedding, 
                output_name_embedding, input_name_classify, output_name_classify)
            logging.info('Clip classified')
            sleep(args.sound_poll_freq)
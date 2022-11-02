import argparse
import json
import logging
import os
import sys
import wave
from time import sleep

import numpy as np
import paho.mqtt.publish as publish
import pyaudio
import requests
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import vggish_input
import vggish_params
import vggish_postprocess
from config import DEFAULT_PCA_PARAMS

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=LOG_LEVEL)


def generate_embeddings(
    endpoint_uuid, model_version, client_class, client, wav_file
):
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

    logging.info(examples_batch.shape)
    # Create request input for embeddings model
    request_input = client_class.InferInput(
        "vggish/input_features", examples_batch.shape, "FP32"
    )
    request_input.set_data_from_numpy(examples_batch.astype(np.float32))

    # Create Request Output containers
    embeddings = client_class.InferRequestedOutput("vggish/embedding")

    # Run inference
    result = client.infer(
        endpoint_uuid,
        (request_input,),
        model_version=model_version,
        outputs=(embeddings,),
    )

    # TODO: turn this into a custom backend to create triton ensemble
    return vggish_postprocess.Postprocessor(DEFAULT_PCA_PARAMS).postprocess(
        result.as_numpy("vggish/embedding")
    )


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


def upload_model(
    url,
    model_type,
    model_name,
    model_filepath,
    model_config_filepath,
    profile_data_filepath=None,
    classes_filepath=None,
):
    """
    Upload a model and its triton model config to the AC
    Returns status code
    """
    url = f"http://{url}/upload/{model_type}"
    req_params = {"model_name": model_name}

    upload_files = [
        ("files", open(model_filepath, "rb")),
        ("files", open(model_config_filepath, "rb")),
    ]

    if profile_data_filepath:
        upload_files.append(
            (
                "files",
                open(profile_data_filepath, "rb"),
            )
        )
    if classes_filepath:
        upload_files.append(
            (
                "files",
                open(classes_filepath, "rb"),
            )
        )

    return requests.post(url, params=req_params, files=upload_files)


def load_model(
    url,
    model_name,
    load_type,
    method,
    request_batch_size=1,
    throughput_objective_weight=1,
    latency_objective_weight=1,
    latency_constraint=1,
):
    # Create a load request
    load_request = {
        "model_name": model_name,
        "load_type": load_type,
        "method": method,
        "batch_size": request_batch_size,
        "perf_targets": {
            "objectives": {
                "perf_throughput": throughput_objective_weight,
                "perf_latency": latency_objective_weight,
            },
            "constraints": {
                "perf_throughput": 1 / latency_constraint,
                "perf_latency": latency_constraint,
            },
        },
    }

    url = f"http://{url}/load"
    return requests.post(url, json=load_request)


def uint8_to_float32(x):
    return (np.float32(x) - 128.0) / 128.0


def record_clip(stream, seconds):
    frames = []

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        frames.append(data)

        if len(frames) == int(RATE * seconds / CHUNK):
            wavefile = wave.open("current.wav", "wb")
            wavefile.setnchannels(CHANNELS)
            wavefile.setsampwidth(audio.get_sample_size(FORMAT))
            wavefile.setframerate(RATE)
            wavefile.writeframes(b"".join(frames))
            wavefile.close()
            return


def classify_sound(
    endpoint_uuid_embeddings,
    endpoint_uuid_classify,
    classes,
    file_path,
    client_class,
    client,
    model_version,
):
    raw_embeddings = generate_embeddings(
        endpoint_uuid_embeddings,
        model_version,
        client_class,
        client,
        file_path,
    )
    embeddings_processed = classifier_pre_process(raw_embeddings, 0)

    # Create request input for embeddings model
    request_input = client_class.InferInput(
        "input_1", embeddings_processed.shape, "FP32"
    )
    request_input.set_data_from_numpy(embeddings_processed.astype(np.float32))

    # Create Request Output containers
    if client_class == httpclient:
        classifications_request = client_class.InferRequestedOutput(
            "activation_4/Sigmoid", binary_data=True, class_count=classes
        )
    else:
        classifications_request = client_class.InferRequestedOutput(
            "activation_4/Sigmoid", class_count=classes
        )

    # Run inference
    results = client.infer(
        endpoint_uuid_classify,
        (request_input,),
        model_version=model_version,
        outputs=(classifications_request,),
    )

    classifications = results.as_numpy("activation_4/Sigmoid")

    # iterate through top CLASSES results and construct mqtt message
    msg = {}
    for idx, classification in enumerate(classifications):
        msg["label" + str(idx)] = classification.decode("ascii").split(":")[2]
        msg["probability" + str(idx)] = classification.decode("ascii").split(
            ":"
        )[0]
    logging.info(msg)

    # Publish msg to mqtt
    try:
        publish.single(
            args.mqtt_topic, json.dumps(msg), hostname=args.mqtt_broker_host
        )
        logging.info(
            "Sound classified successfully and results published to mqtt"
        )
    except Exception as e:
        logging.info(
            "Sound classified successfully but mqtt publish failed with"
            " error: {}".format(e)
        )


def handler_stop_signals(signum, frame):
    # Close all
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sys.exit(0)


if __name__ == "__main__":
    # Set log level
    loglevel = os.getenv("LOG_LEVEL", "info").lower()
    if loglevel == "info":
        logging.basicConfig(level=logging.INFO)
    elif loglevel == "warning":
        logging.basicConfig(level=logging.WARN)
    elif loglevel == "debug":
        logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output for triton",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=False,
        default=os.getenv("CLASSES", 5),
        help="Number of class results to report. Default is 5.",
    )
    parser.add_argument(
        "-m",
        "--model-name-classify",
        type=str,
        required=False,
        default=os.getenv("MODEL_NAME_CLASSIFY", "ambient_sound_clf"),
        help="Name of audio classification model",
    )
    parser.add_argument(
        "-e",
        "--model-name-embedding",
        type=str,
        required=False,
        default=os.getenv("MODEL_NAME_EMBEDDING", "vggish"),
        help="Name of embedding model",
    )
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    parser.add_argument(
        "-u",
        "--triton-url",
        type=str,
        required=False,
        default=os.getenv("TRITON_URL", "localhost:8000"),
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "--admission-controller-url",
        type=str,
        required=False,
        default=os.getenv("ADMISSION_CONTROLLER_URL", ""),
        help="Admission Controller URL. Default is localhost:2520.",
    )
    parser.add_argument(
        "-p",
        "--sound-poll-freq",
        type=int,
        required=False,
        default=os.getenv("CLASSIFY_SERVICE_POLL_FREQUENCY", 10),
        help="Sound poll frequency.",
    )
    parser.add_argument(
        "--protocol", type=str, default=os.getenv("PROTOCOL", "HTTP")
    )
    parser.add_argument(
        "-r",
        "--record-secs",
        type=int,
        required=False,
        default=os.getenv("RECORD_SECONDS", 10),
        choices=range(1, 40),
        help="Seconds to record. Default is 10",
    )
    parser.add_argument("-d", "--use-clips", action="store_true")
    parser.add_argument(
        "--audio-file-dir",
        type=str,
        required=False,
        default=os.getenv("AUDIO_FILES", "/samples"),
    )
    parser.add_argument(
        "-b",
        "--mqtt-broker-host",
        type=str,
        required=False,
        default=os.getenv("MQTT_BROKER_HOST", "fluent-bit"),
        help="mqtt broker host",
    )
    parser.add_argument(
        "--mqtt-broker-port",
        type=int,
        required=False,
        default=os.getenv("MQTT_BROKER_PORT", "1883"),
        help="port number of the mqtt server (1024 to 65535) default 1883",
    )
    parser.add_argument(
        "-t",
        "--mqtt-topic",
        type=str,
        required=False,
        default=os.getenv("MQTT_TOPIC", "/demo"),
        help="mqtt broker topic",
    )

    args = parser.parse_args()

    if args.admission_controller_url:
        # Use admission control api to upload models then request to load
        logging.info("Uploading vggish model")
        try:
            res = upload_model(
                args.admission_controller_url,
                "tf",
                "vggish",
                "vggish.graphdef",
                "vggish_config.pbtxt",
            )
            assert res.status_code in [201, 303]
        except Exception:
            logging.error(f"Upload model failed with response {res.text}")
            sys.exit(1)

        logging.info("Loading vggish model")
        try:
            res = load_model(
                args.admission_controller_url,
                "vggish",
                "auto_gen",
                "passthrough",
                latency_constraint=args.sound_poll_freq,
            )

            # endpoint_uuid holds the translated model name after loading
            # for the client to request from using the triton client api
            assert res.status_code in [201, 303]
        except AssertionError:
            logging.error(f"Load model failed with response {res.text}")
            sys.exit(1)

        res_json = res.json()
        endpoint_uuid_embeddings = res_json["request_uuid"]
        triton_url = args.admission_controller_url.split(":")[0] + ":" + "2521"
        logging.info(res_json["model_config"])

        logging.info("Uploading sound classifier model")
        try:
            res = upload_model(
                args.admission_controller_url,
                "tf",
                "ambient_sound_clf",
                "ambient_sound_clf.graphdef",
                "ambient_sound_clf_config.pbtxt",
                classes_filepath="ambient_sound_clf.classes",
            )
            assert res.status_code in [201, 303]
        except AssertionError:
            logging.error(f"Upload model failed with response {res.text}")
            sys.exit(1)

        logging.info("Loading sound classifier model")
        try:
            res = load_model(
                args.admission_controller_url,
                "ambient_sound_clf",
                "auto_gen",
                "passthrough",
                latency_constraint=args.sound_poll_freq,
            )

            assert res.status_code in [201, 303]
        except AssertionError:
            logging.error(f"Load model failed with response {res.text}")
            sys.exit(1)

        res_json = res.json()
        endpoint_uuid_classify = res_json["request_uuid"]
        logging.info(res_json["model_config"])
    else:
        endpoint_uuid_embeddings = "vggish"
        endpoint_uuid_classify = "ambient_sound_clf"
        triton_url = args.triton_url

    if args.protocol.lower() == "grpc":
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(
            url=triton_url, verbose=args.verbose
        )
        triton_class = grpcclient
    else:
        # Create HTTP client for communicating with the server
        triton_client = httpclient.InferenceServerClient(
            url=triton_url, verbose=args.verbose
        )
        triton_class = httpclient

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
            frames_per_buffer=CHUNK,
        )

    while True:
        if args.use_clips:
            for file in os.listdir(args.audio_file_dir):
                file_path = os.path.join(args.audio_file_dir, file)
                classify_sound(
                    endpoint_uuid_embeddings,
                    endpoint_uuid_classify,
                    args.classes,
                    file_path,
                    triton_class,
                    triton_client,
                    args.model_version,
                )
                logging.info("Clip {} classified".format(file_path))
                sleep(args.sound_poll_freq)

        else:
            record_clip(stream, args.record_secs)
            logging.info("Clip recorded")
            classify_sound(
                endpoint_uuid_embeddings,
                endpoint_uuid_classify,
                args.classes,
                "current.wav",
                triton_class,
                triton_client,
                args.model_version,
            )
            logging.info("Clip classified")
            sleep(args.sound_poll_freq)

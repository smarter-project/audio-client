# Context

## Functionality
- Records arbitrary length audio clips to send to Nvidia's ML inference server: Triton. Users can either read audio data from pulseaudio, which is configured in the client.conf file of this project, or pass a path to a directory containing audio clips to classify.

- This container assumes that Triton and Pulseaudio are available on the node it is running on. Further, the microphone must be able to output sample at 16Khz. 

## Application Arguments

### Environment Variables
- name: LOG_LEVEL
    - desc: Set to info, warning, or debug
    - default: info
- name: CLASSES
    - desc: Number of classes to report
    - flag: `-c,--classes`
    - default: 5
- name: MODEL_NAME_CLASSIFY
    - desc: Name of model in triton to perform audio inference against
    - flag: `-m,--model-name-classify`
    - default: ambient_sound_clf
- name: MODEL_NAME_EMBEDDING
    - desc: Name of model in triton to generate audio embeddings
    - flag: `-e,--model-name-embeddings`
    - default: vggish
- name: TRITON_URL
    - desc: URL to access triton with
    - flag: `-u,--triton-url`
    - default: localhost:8000
- name: SMARTER_INFERENCE_URL
    - desc: url to access smarter-inference, default is empty string. If set, triton url will be overwritten within smarter-inference inference access point
    - flag: `--smarter-inference-url`
    - default: none
- name: CLASSIFY_SERVICE_POLL_FREQUENCY
    - desc: Seconds in between clip recordings
    - flag: `-p,--sound-poll-freq`
    - default: 10 seconds
- name: PROTOCOL
    - desc: Protocol to access triton with (HTTP or gRPC)
    - flag: `--protocol`
    - default: HTTP
- name: RECORD_SECONDS
    - desc: How long each recorded clip should be
    - flag: `-r,--record-secs`
    - default: 10 seconds
    - max: 40 seconds
- name: AUDIO_FILES
    - desc: Filepath for audio files to be used
    - flag: `--audio-file-dir`
    - default: /samples
- name: MQTT_BROKER_HOST
    - desc: Hostname for MQTT Broker
    - flag: `-b,--mqtt-broker-host`
    - default: fluent-bit
- name: MQTT_BROKER_PORT
    - desc: Hostname for MQTT Port
    - flag: `--mqtt-broker-port`
    - default: 1883
- name: MQTT_TOPIC
    - desc: MQTT message topic string
    - flag: `--mqtt-topic`
    - default: /demo

### Command Line Specific Args
- `-v,--verbose` - enable verbose output for triton if passed
- `-x,--model-version` - version of model, default is latest version
- `-d,--use-clips` - use pre-recorded clips. to be used in conjunction with audio file directory flag
    


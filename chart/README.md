# smarter-audio-client

This chart provides an example workload for Smarter Edge. The example processes audio input coming from a pulseaudio source through a ML model loaded in smarter-admission-controller (Triton) and uploads the result to MQTT (provided by smarter-fluent-bit).

## Functionality
- Records arbitrary length audio clips to send to Nvidia's ML inference server: Triton. Users can either read audio data from pulseaudio, which is configured in the client.conf file of this project, or pass a path to a directory containing audio clips to classify.

- This container assumes that smarter-admission-controller, smarter-pulseaudio and smarter-fluent-bit  are available on the node it is running on. Further, the microphone must be able to output sample at 16Khz. 

## Chart values

### Configuration

* pulsesource
  Specific pulseaudio source to use, by default it is set ad  alsa_input.hw_1_0
* loglevel
  Logging level, set to INFO
* pollfrequency
  Audio lenght and also inference interval default to 20
* mqtt
  MQTT destination information
  * host
    Ip or hostname defaults to smarter-fluent-bit
* admission_controller
  How to access admission-controller
  * host
    defaults to smarter-admission-controller
  * port
    defaults to 2520
* pulseaudio
  How to access pulseaudio
  * host
    defaults smarter-pulseaudio
  * port
    default to 4713

## Usage

```
helm install smarter-audio-client chart
```


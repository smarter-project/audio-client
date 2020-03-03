# Context

## Functionality
- Records arbitrary length audio clips to send to an ML classification service

## Enviornment Variables
- name: DEMO
    - desc: If set to any valid string, microphone will not be used, and instead pre-recorded audio clips will be submitted in loop
    - default: None
- name: FAAS
    - desc: If set to any valid string, faas json formatting function will be called 
    - default: None
- name: SQUASH_FUNCTION_URL
    - desc: URL of faas function to flatten json array
    - default: http://edgefaas:8080/squasher
- name: CLASSIFY_SERVICE_HOST
    - desc: Hostname of audio classification service
    - default: sound-classifier
- name: CLASSIFY_SERVICE_PORT
    - desc: Port number of audio classification service
    - default: 5000
- name: CLASSIFY_SERVICE_POLL_FREQUENCY
    - desc: Seconds in between clip recordings
    - default: 10
- name: MQTT_BROKER_HOST
    - desc: Hostname for MQTT BROKER
    - default: None
- name: TOPIC
    - desc: MQTT message topic string
    - default: None
- name: RECORD_SECONDS
    - desc: How long each recorded clip should be
    - default: None


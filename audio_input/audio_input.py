# stt_stream_v2.py — revised, callback-based and Jupyter-friendly
import sys
import queue
import pyaudio
from google.cloud import speech_v2 as speech

# --- config you should set ---
PROJECT = "map-mcp-471616"
LOCATION = "us-central1"  # choose your region, e.g., us-central1 / europe-west1 / asia-southeast1
LANG = "en-US"  # BCP-47 code
RATE = 16000  # mic sample rate (Hz)
CHUNK = int(RATE * 0.05)  # 50 ms of audio per callback
# -----------------------------

# v2 requires a regional endpoint
client = speech.SpeechClient(
    client_options={"api_endpoint": f"{LOCATION}-speech.googleapis.com"}
)

# We’re streaming raw PCM (LINEAR16) from the mic, so use explicit decoding.
config = speech.RecognitionConfig(
    explicit_decoding_config=speech.ExplicitDecodingConfig(
        encoding="LINEAR16",
        sample_rate_hertz=RATE,
        audio_channel_count=1,
    ),
    features=speech.RecognitionFeatures(enable_automatic_punctuation=True),
    language_codes=[LANG],
    model="latest_long",
)

streaming_cfg = speech.StreamingRecognitionConfig(
    config=config,
    streaming_features=speech.StreamingRecognitionFeatures(
        interim_results=True,
        enable_voice_activity_events=True,
    ),
)

recognizer = f"projects/{PROJECT}/locations/{LOCATION}/recognizers/_"

# --- microphone stream & callback ---
q: "queue.Queue[bytes]" = queue.Queue()
pa = pyaudio.PyAudio()


def _callback(in_data, frame_count, time_info, status_flags):
    """
    PyAudio calls this whenever a CHUNK of mic audio is available.
    - in_data: raw bytes (LINEAR16, little-endian), len = frame_count * 2 bytes (mono, 16-bit)
    - Return (out_data, flag). For input-only, out_data is None.
    """
    q.put(in_data)  # hand off audio to the gRPC request generator
    return (None, pyaudio.paContinue)  # keep the stream running


stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=_callback,
)


def cleanup_the_stream():
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\nStopped.")


def request_generator():
    """Yield the initial config request, then a stream of audio-chunk requests."""
    # 1) first message: recognizer + streaming config, no audio
    yield speech.StreamingRecognizeRequest(
        recognizer=recognizer,
        streaming_config=streaming_cfg,
    )
    # 2) subsequent messages: raw audio bytes
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield speech.StreamingRecognizeRequest(audio=chunk)


class ExitLoop(Exception):
    pass


print("🎤 Speak (Ctrl+C to stop)…")
try:
    stream.start_stream()
    responses = client.streaming_recognize(requests=request_generator())

    for resp in responses:
        # (optional) voice activity events are available in resp.speech_event_type
        for result in resp.results:
            if not result.alternatives:
                continue
            text = result.alternatives[0].transcript.strip()
            if result.is_final:
                print(f"\n✅ {text}")
                if text == "It is done.":
                    raise ExitLoop()
            else:
                # overwrite the same line for interim hypotheses
                sys.stdout.write(f"\r… {text} ")
                sys.stdout.flush()

except KeyboardInterrupt:
    pass
except ExitLoop:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\nStopped.")
finally:
    pass
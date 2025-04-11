import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="audio.wav", duration=5, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, audio)
    print(f"Audio saved to {filename}")
    return filename

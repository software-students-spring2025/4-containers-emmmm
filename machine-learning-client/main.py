from AudioRecorder import record_audio
from EmotionAnalyzer import analyze_emotion

def main():
    audio_path = record_audio()
    emotion = analyze_emotion(audio_path)
    print("Final Result:", emotion)

if __name__ == "__main__":
    main()

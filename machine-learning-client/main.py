"""Main execution entry."""

from audio_recorder import record_audio
from emotion_analyzer import analyze_emotion


def main():
    """The main function to run the overall function."""
    audio_path = record_audio()
    emotion = analyze_emotion(audio_path)
    print("Final Result:", emotion)


if __name__ == "__main__":
    main()

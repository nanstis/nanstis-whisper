import whisper_timestamped as whisper
from dataclasses import dataclass
import json
import ffmpeg
import math


whisper_file_name = './output.json'
subtitle_file_name = './output.srt'


@dataclass
class Segment:
    id: int
    text: float
    start: float
    end: float


def extract_text(path):
    audio = whisper.audio.load_audio(path)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(
        model, audio, beam_size=5, best_of=5, language="en")

    with open(whisper_file_name, 'w') as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))


def create_subtitles(file):
    with open(whisper_file_name, 'r') as json_file:
        segments = json.load(json_file)['segments']
        file.mode = 'a'

        for segment in segments:
            segment = Segment(
                segment['id'],
                segment['text'],
                segment['start'],
                segment['end']
            )

            file.write(
                f"{segment.id}\n{format_time(segment.start)} --> {format_time(segment.end)}\n{segment.text}\n\n")


def format_time(total_seconds):
    hours = math.floor(total_seconds / 3600)
    minutes = math.floor((total_seconds % 3600) / 60)
    seconds = math.floor(total_seconds % 60)
    milliseconds = round((total_seconds % 1) * 100)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:02d}"


def merge_subtitles(input_file):
    video_stream = ffmpeg.input(input_file)
    subtitle_stream = ffmpeg.input(subtitle_file_name, f='srt')

    ffmpeg.output(
        video_stream,
        subtitle_stream,
        '/mnt/c/Users/nicoa/output.mkv',
        map="1",
        crf=23,
    ).run()


def transcribe(file):
    audio = ffmpeg.input(file)
    output_file = file + '.mp3'
    audio = ffmpeg.output(audio, output_file)
    audio.run()

    # Transcribe to json using whisper model
    extract_text(output_file)

    # Format json to an srt file
    with open(subtitle_file_name, 'w') as f:
        create_subtitles(f)

    # Merge subtitles to video
    merge_subtitles(file)


transcribe('S08E01.mkv')

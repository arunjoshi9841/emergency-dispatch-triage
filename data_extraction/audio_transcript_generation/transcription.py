from __future__ import annotations

import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.utils import mediainfo, which as pydub_which

from shared.clients import get_openai_client, call_with_retries

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 1400
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"

TRANSCRIPTION_PROMPT = (
    "This is a 911 emergency call recording. "
    "Preserve all filler words such as um, uh, like, you know. "
    "Note notable audio signals in square brackets, e.g. [sirens in background], [caller crying]."
)


def _configure_ffmpeg() -> None:
    ffmpeg_path = pydub_which("ffmpeg")
    ffprobe_path = pydub_which("ffprobe")
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
    if ffprobe_path:
        AudioSegment.ffprobe = ffprobe_path


def get_audio_duration_seconds(file_path: Path) -> float:
    info = mediainfo(str(file_path))
    duration = info.get("duration")
    if duration is None:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    return float(duration)


def chunk_audio(
    file_path: Path,
    max_bytes: int = MAX_FILE_SIZE_BYTES,
    max_duration_seconds: int = MAX_AUDIO_DURATION_SECONDS,
) -> list[Path]:
    audio = AudioSegment.from_file(file_path)
    max_duration_ms = max_duration_seconds * 1000
    hard_limit_ms = max(1000, max_duration_ms - 10_000)

    chunks: list[Path] = []
    stem = file_path.stem
    tmp_dir = Path(tempfile.mkdtemp(prefix="911_chunks_"))
    total_ms = len(audio)
    start_ms = 0
    chunk_idx = 0

    while start_ms < total_ms:
        target_end = min(start_ms + hard_limit_ms, total_ms)

        if target_end < total_ms:
            window_start = max(start_ms, target_end - 30_000)
            window = audio[window_start:target_end]
            nonsilent_ranges = detect_nonsilent(window, min_silence_len=700, silence_thresh=-40)

            if nonsilent_ranges:
                last_end = nonsilent_ranges[-1][1]
                cut_ms = window_start + last_end
                if cut_ms <= start_ms + 5_000:
                    cut_ms = target_end
            else:
                cut_ms = target_end
        else:
            cut_ms = total_ms

        segment = audio[start_ms:cut_ms]
        chunk_path = tmp_dir / f"{stem}_chunk_{chunk_idx:03d}.mp3"
        segment.export(chunk_path, format="mp3")

        actual_duration = get_audio_duration_seconds(chunk_path)
        actual_size = chunk_path.stat().st_size
        if actual_duration > max_duration_seconds:
            raise RuntimeError(
                f"Chunk {chunk_path.name} duration {actual_duration:.1f}s exceeds {max_duration_seconds}s"
            )
        if actual_size > max_bytes:
            raise RuntimeError(
                f"Chunk {chunk_path.name} size {actual_size / (1024 * 1024):.2f} MB exceeds {max_bytes / (1024 * 1024):.2f} MB"
            )

        chunks.append(chunk_path)
        chunk_idx += 1
        start_ms = cut_ms

    return chunks


def _normalize_to_mp3(file_path: Path) -> Path:
    """Re-encode any audio format to a proper MP3 via ffmpeg.

    Handles files with wrong extensions (e.g. MOV renamed to .mp3).
    Returns path to a temp file that the caller is responsible for deleting.
    """
    audio = AudioSegment.from_file(file_path)
    tmp = Path(tempfile.mktemp(suffix=".mp3", prefix="911_norm_"))
    audio.export(tmp, format="mp3")
    return tmp


def _transcribe_file(audio_path: Path) -> dict:
    client = get_openai_client()

    def _call_once():
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=f,
                prompt=TRANSCRIPTION_PROMPT,
            )

    return call_with_retries(_call_once)


def _transcribe_full_recording(file_path: Path) -> dict:
    file_size = file_path.stat().st_size
    duration_seconds = get_audio_duration_seconds(file_path)

    if file_size <= MAX_FILE_SIZE_BYTES and duration_seconds <= MAX_AUDIO_DURATION_SECONDS:
        return _transcribe_file(file_path)

    chunk_paths = chunk_audio(file_path)
    full_text_parts: list[str] = []
    try:
        for chunk_path in chunk_paths:
            result = _transcribe_file(chunk_path)
            chunk_text = result.text if hasattr(result, "text") else ""
            full_text_parts.append(chunk_text)
        return {"text": "\n\n".join(part for part in full_text_parts if part.strip())}
    finally:
        for chunk_path in chunk_paths:
            chunk_path.unlink(missing_ok=True)


def _format_transcript(response) -> str:
    text = response.text if hasattr(response, "text") else response.get("text", "")
    return text.strip()


def generate_whisper_transcript(audio_path: Path) -> str:
    """Transcribe a 911 audio recording using OpenAI Whisper.

    Handles chunking for large files automatically.
    Returns raw transcript text (before speaker-label postprocessing).
    """
    _configure_ffmpeg()
    normalized: Path | None = None
    try:
        normalized = _normalize_to_mp3(audio_path)
        raw_response = _transcribe_full_recording(normalized)
    finally:
        if normalized and normalized.is_file():
            normalized.unlink(missing_ok=True)
    return _format_transcript(raw_response)

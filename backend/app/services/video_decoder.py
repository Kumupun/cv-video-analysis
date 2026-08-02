from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from app.domain.schemas import VideoMetadata


class VideoDecodeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DecodedBatch:
    chunk_index: int
    context_start_frame: int
    context_end_frame: int
    valid_start_frame: int
    valid_end_frame: int
    frames_rgb: np.ndarray
    is_last: bool


class FFmpegVideoDecoder:
    """Memory-bounded RGB decoder backed by one sequential FFmpeg process.

    Decord ``VideoReader.get_batch`` can keep native decode buffers and allocator
    arenas alive for the lifetime of a reader. With multiple high-resolution
    videos in one ingest process, resident memory may therefore grow toward the
    uncompressed size of the videos even though Python drops each NumPy batch.

    This decoder launches FFmpeg as a child process and reads only the bytes for
    the current chunk into a preallocated NumPy tensor. The child is closed at
    task completion, so codec buffers cannot accumulate across archive entries.
    A non-sequential chunk request restarts FFmpeg at the exact context frame;
    normal ingestion remains sequential and uses one process for the whole task.
    """

    def __init__(
        self,
        path: Path,
        chunk_size: int,
        overlap_frames: int = 0,
        max_decoded_chunk_bytes: int | None = None,
        *,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        decode_threads: int = 1,
        probe_timeout_seconds: float = 900.0,
        process_stop_timeout_seconds: float = 5.0,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap_frames < 0:
            raise ValueError("overlap_frames cannot be negative")
        if overlap_frames >= chunk_size:
            raise ValueError("overlap_frames must be smaller than chunk_size")
        if max_decoded_chunk_bytes is not None and max_decoded_chunk_bytes <= 0:
            raise ValueError("max_decoded_chunk_bytes must be positive")
        if decode_threads <= 0:
            raise ValueError("decode_threads must be positive")
        if probe_timeout_seconds <= 0:
            raise ValueError("probe_timeout_seconds must be positive")
        if process_stop_timeout_seconds <= 0:
            raise ValueError("process_stop_timeout_seconds must be positive")

        self._path = path
        self._configured_chunk_size = chunk_size
        self._configured_overlap_frames = overlap_frames
        self._max_decoded_chunk_bytes = max_decoded_chunk_bytes
        self._ffmpeg_path = ffmpeg_path
        self._ffprobe_path = ffprobe_path
        self._decode_threads = decode_threads
        self._probe_timeout_seconds = probe_timeout_seconds
        self._process_stop_timeout_seconds = process_stop_timeout_seconds

        self._metadata: VideoMetadata | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._next_chunk_index: int | None = None
        self._previous_tail: np.ndarray | None = None

    def open(self) -> VideoMetadata:
        if self._metadata is not None:
            return self._metadata
        self._metadata = self._probe_metadata()
        return self._metadata

    @property
    def metadata(self) -> VideoMetadata:
        return self.open()

    @property
    def effective_chunk_size(self) -> int:
        chunk_size, _ = self._effective_chunking()
        return chunk_size

    @property
    def effective_overlap_frames(self) -> int:
        _, overlap_frames = self._effective_chunking()
        return overlap_frames

    def _effective_chunking(self) -> tuple[int, int]:
        metadata = self.metadata
        if self._max_decoded_chunk_bytes is None:
            return self._configured_chunk_size, self._configured_overlap_frames

        bytes_per_rgb_frame = metadata.width * metadata.height * 3
        if bytes_per_rgb_frame > self._max_decoded_chunk_bytes:
            raise VideoDecodeError(
                "A single decoded RGB frame exceeds the configured chunk byte limit"
            )
        max_context_frames = max(
            1,
            self._max_decoded_chunk_bytes // max(bytes_per_rgb_frame, 1),
        )
        if max_context_frames == 1:
            return 1, 0



        overlap_frames = min(
            self._configured_overlap_frames,
            max_context_frames // 4,
        )
        chunk_size = min(
            self._configured_chunk_size,
            max_context_frames - overlap_frames,
        )
        chunk_size = max(1, chunk_size)
        overlap_frames = min(overlap_frames, max(0, chunk_size - 1))
        return chunk_size, overlap_frames

    @property
    def total_chunks(self) -> int:
        metadata = self.metadata
        chunk_size = self.effective_chunk_size
        return (metadata.frame_count + chunk_size - 1) // chunk_size

    def decode_batch(self, chunk_index: int) -> DecodedBatch:
        """Decode one chunk while reusing FFmpeg for sequential requests.

        The normal ingest path requests ``0, 1, 2, ...`` and therefore streams
        the file once. A retry may jump directly to a later durable chunk. In
        that case FFmpeg is restarted with an exact frame trim beginning at the
        context frame required by that chunk.
        """

        if chunk_index < 0:
            raise IndexError("chunk_index cannot be negative")
        if chunk_index >= self.total_chunks:
            raise IndexError("chunk_index is outside the video")

        if self._process is None or self._next_chunk_index != chunk_index:
            self._restart_stream(chunk_index)

        metadata = self.metadata
        chunk_size, overlap_frames = self._effective_chunking()
        valid_start = chunk_index * chunk_size
        valid_end_exclusive = min(valid_start + chunk_size, metadata.frame_count)
        context_start = max(0, valid_start - overlap_frames)

        if self._previous_tail is None:
            prefix_frames = 0
            frames_to_read = valid_end_exclusive - context_start
        else:
            expected_prefix = valid_start - context_start
            if self._previous_tail.shape[0] != expected_prefix:
                raise VideoDecodeError(
                    "FFmpeg overlap state does not match the requested chunk"
                )
            prefix_frames = expected_prefix
            frames_to_read = valid_end_exclusive - valid_start

        context_frame_count = valid_end_exclusive - context_start
        frames = np.empty(
            (
                context_frame_count,
                metadata.height,
                metadata.width,
                3,
            ),
            dtype=np.uint8,
        )
        if prefix_frames:
            assert self._previous_tail is not None
            frames[:prefix_frames] = self._previous_tail

        self._read_frames_into(frames[prefix_frames:], frames_to_read)

        if overlap_frames:
            tail_count = min(overlap_frames, frames.shape[0])
            self._previous_tail = np.array(
                frames[-tail_count:],
                copy=True,
                order="C",
            )
        else:
            self._previous_tail = None

        is_last = valid_end_exclusive == metadata.frame_count
        self._next_chunk_index = chunk_index + 1
        if is_last:
            self._finish_process()

        return DecodedBatch(
            chunk_index=chunk_index,
            context_start_frame=context_start,
            context_end_frame=valid_end_exclusive - 1,
            valid_start_frame=valid_start,
            valid_end_frame=valid_end_exclusive - 1,
            frames_rgb=frames,
            is_last=is_last,
        )

    def batches(self, start_chunk: int = 0) -> Iterator[DecodedBatch]:
        try:
            for chunk_index in range(start_chunk, self.total_chunks):
                yield self.decode_batch(chunk_index)
        finally:
            self.close()

    def close(self) -> None:
        process = self._process
        self._process = None
        self._next_chunk_index = None
        self._previous_tail = None
        if process is None:
            return

        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self._process_stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self._process_stop_timeout_seconds)
        if process.stderr is not None:
            process.stderr.close()

    def _restart_stream(self, chunk_index: int) -> None:
        self.close()
        metadata = self.metadata
        chunk_size, overlap_frames = self._effective_chunking()
        valid_start = chunk_index * chunk_size
        context_start = max(0, valid_start - overlap_frames)
        remaining_frames = metadata.frame_count - context_start
        if remaining_frames <= 0:
            raise IndexError("chunk_index is outside the video")

        filter_expression = f"trim=start_frame={context_start}"
        command = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-nostdin",
            "-threads",
            str(self._decode_threads),
            "-i",
            str(self._path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_expression,
            "-vsync",
            "0",
            "-frames:v",
            str(remaining_frames),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024 * 1024,
            )
        except OSError as exc:
            raise VideoDecodeError(f"Could not start FFmpeg: {exc}") from exc

        self._next_chunk_index = chunk_index
        self._previous_tail = None

    def _read_frames_into(
        self,
        target: np.ndarray,
        expected_frames: int,
    ) -> None:
        if target.shape[0] != expected_frames:
            raise VideoDecodeError("Internal FFmpeg frame buffer size mismatch")
        process = self._process
        if process is None or process.stdout is None:
            raise VideoDecodeError("FFmpeg decoder process is not running")

        byte_view = memoryview(target).cast("B")
        offset = 0
        while offset < len(byte_view):
            read_count = process.stdout.readinto(byte_view[offset:])
            if not read_count:
                break
            offset += read_count

        if offset != len(byte_view):
            error = self._collect_process_error(process)
            raise VideoDecodeError(
                "FFmpeg ended before the complete RGB chunk was decoded: "
                f"expected={len(byte_view)} bytes, received={offset} bytes. "
                f"{error}"
            )

    def _finish_process(self) -> None:
        process = self._process
        self._process = None
        self._next_chunk_index = None
        self._previous_tail = None
        if process is None:
            return

        if process.stdout is not None:
            process.stdout.close()
        try:
            return_code = process.wait(timeout=self._process_stop_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=self._process_stop_timeout_seconds)
            raise VideoDecodeError(
                "FFmpeg did not stop after decoding the last frame"
            ) from exc

        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
            process.stderr.close()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise VideoDecodeError(
                f"FFmpeg exited with code {return_code}: {detail or 'unknown error'}"
            )

    @staticmethod
    def _collect_process_error(process: subprocess.Popen[bytes]) -> str:
        try:
            return_code = process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=2.0)
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
        detail = stderr.decode("utf-8", errors="replace").strip()
        return f"FFmpeg exit code {return_code}: {detail or 'no error output'}"

    def _probe_metadata(self) -> VideoMetadata:
        if not self._path.is_file():
            raise VideoDecodeError(f"Video file does not exist: {self._path}")

        quick_probe = self._run_ffprobe(count_frames=False)
        stream = self._first_video_stream(quick_probe)
        width = self._positive_int(stream.get("width"), "width")
        height = self._positive_int(stream.get("height"), "height")
        fps = self._parse_fps(stream)
        frame_count = self._optional_positive_int(stream.get("nb_frames"))

        if frame_count is None:
            counted_probe = self._run_ffprobe(count_frames=True)
            counted_stream = self._first_video_stream(counted_probe)
            frame_count = self._optional_positive_int(
                counted_stream.get("nb_read_frames")
            )
        if frame_count is None:
            raise VideoDecodeError("FFprobe could not determine the video frame count")

        format_data = quick_probe.get("format")
        if not isinstance(format_data, dict):
            format_data = {}
        duration = self._optional_positive_float(stream.get("duration"))
        if duration is None:
            duration = self._optional_positive_float(format_data.get("duration"))
        if duration is None:
            duration = frame_count / fps

        codec_raw = stream.get("codec_name")
        codec = str(codec_raw) if codec_raw else None
        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=duration,
            codec=codec,
        )

    def _run_ffprobe(self, *, count_frames: bool) -> dict[str, Any]:
        command = [
            self._ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
        ]
        if count_frames:
            command.append("-count_frames")
        command.extend(
            [
                "-show_entries",
                (
                    "stream=width,height,avg_frame_rate,r_frame_rate,"
                    "nb_frames,nb_read_frames,duration,codec_name:format=duration"
                ),
                "-of",
                "json",
                str(self._path),
            ]
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._probe_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise VideoDecodeError(
                "FFprobe timed out while reading video metadata"
            ) from exc
        except OSError as exc:
            raise VideoDecodeError(f"Could not start FFprobe: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown error"
            raise VideoDecodeError(f"Could not probe video: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise VideoDecodeError("FFprobe returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise VideoDecodeError("FFprobe returned an invalid payload")
        return payload

    @staticmethod
    def _first_video_stream(payload: dict[str, Any]) -> dict[str, Any]:
        streams = payload.get("streams")
        if not isinstance(streams, list) or not streams:
            raise VideoDecodeError("The file contains no decodable video stream")
        stream = streams[0]
        if not isinstance(stream, dict):
            raise VideoDecodeError("FFprobe returned invalid video metadata")
        return stream

    @classmethod
    def _parse_fps(cls, stream: dict[str, Any]) -> float:
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = stream.get(key)
            if raw in (None, "", "0/0"):
                continue
            try:
                value = float(Fraction(str(raw)))
            except (ValueError, ZeroDivisionError):
                continue
            if value > 0:
                return value
        raise VideoDecodeError("The video FPS metadata is invalid")

    @classmethod
    def _positive_int(cls, value: Any, field_name: str) -> int:
        parsed = cls._optional_positive_int(value)
        if parsed is None:
            raise VideoDecodeError(f"The video {field_name} metadata is invalid")
        return parsed

    @staticmethod
    def _optional_positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _optional_positive_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None



DecordVideoDecoder = FFmpegVideoDecoder

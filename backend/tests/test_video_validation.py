from app.core.config import Settings
from app.services.video_validation import VideoValidationError, validate_upload_metadata
from fastapi import UploadFile


def test_audio_upload_is_rejected() -> None:
    upload = UploadFile(
        filename="voice.mp3",
        file=None,
        headers={"content-type": "audio/mpeg"},
    )
    try:
        validate_upload_metadata(upload, Settings())
        raise AssertionError("audio file was accepted")
    except VideoValidationError as exc:
        assert "Audio-only" in str(exc)


def test_video_upload_is_accepted() -> None:
    upload = UploadFile(
        filename="clip.mp4",
        file=None,
        headers={"content-type": "video/mp4"},
    )
    assert validate_upload_metadata(upload, Settings()) == ".mp4"

"""
Download video from YouTube (or other supported sites) via yt-dlp.

Usage:
    python download_video.py "https://youtube.com/watch?v=abc123"
    python download_video.py "https://youtube.com/watch?v=abc123" --output /tmp/videos/

Returns JSON with video path and metadata to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def download_video(url: str, output_dir: str | None = None) -> dict:
    """Download video from URL, return metadata dict.

    Returns:
        {
            "filepath": "/tmp/videos/abc123.mp4",
            "title": "Baby First Words",
            "duration": 312,
            "url": "https://youtube.com/watch?v=abc123",
        }
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Install with: pip install yt-dlp")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="activelearning_video_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    outtmpl = str(Path(output_dir) / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Enable remote JS challenge solver for YouTube (requires deno)
        "remote_components": ["ejs:github"],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)

        return {
            "filepath": filepath,
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "url": url,
        }


async def download_video_async(url: str, output_dir: str | None = None) -> dict:
    """Async wrapper — runs download in thread pool."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, download_video, url, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download video for brain training")
    parser.add_argument("url", help="YouTube URL or other supported video URL")
    parser.add_argument("--output", "-o", help="Output directory (default: temp dir)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found — audio extraction may fail later")

    try:
        result = download_video(args.url, args.output)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

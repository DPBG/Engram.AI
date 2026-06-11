#!/usr/bin/env bash
set -eo pipefail
# Download training videos to /data/videos/
#
# Usage:
#   ./deploy/download-videos.sh              # download all default videos
#   ./deploy/download-videos.sh --dir /tmp   # custom output dir

set -euo pipefail

OUTPUT_DIR="${1:-/data/videos}"
mkdir -p "$OUTPUT_DIR"

echo "=== Downloading training videos to $OUTPUT_DIR ==="

# Check yt-dlp
if ! command -v yt-dlp &>/dev/null; then
    echo "Installing yt-dlp..."
    pip3 install yt-dlp
fi

# Baby vocabulary / first words videos
VIDEOS=(
    "https://www.youtube.com/watch?v=Bf9L8sbrFGk"  # First 50 English Words for Toddlers
    "https://www.youtube.com/watch?v=eUutwwoGL8Q"  # ABC Animals Flashcards with Animal Sounds
    "https://www.youtube.com/watch?v=ZONJoCIaHlk"  # First Words for Babies - Teaching Toddlers
    "https://www.youtube.com/watch?v=QXhK-4jrTjs"  # Baby's First Words - 100 Objects
    "https://www.youtube.com/watch?v=UcGm_PM2IwY"  # Fruits and Vegetables Names
    "https://www.youtube.com/watch?v=LAqLBOh3Ric"  # Learning Fruit Names for Toddlers
    "https://www.youtube.com/watch?v=biX7NNxw_w8"  # Means of Transport for Children
)

for url in "${VIDEOS[@]}"; do
    echo ""
    echo "--- Downloading: $url ---"
    yt-dlp \
        -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
        --merge-output-format mp4 \
        -o "$OUTPUT_DIR/%(id)s.mp4" \
        --no-playlist \
        "$url" || echo "  WARN: Failed to download $url (continuing...)"
done

echo ""
echo "=== Downloads complete ==="
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "No videos found."
echo ""
echo "Total: $(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l) videos"
echo "Size:  $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)"

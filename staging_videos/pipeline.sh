#!/bin/bash
set -euo pipefail

# Engram Video Training Pipeline
# Automates download → compress → upload workflow for Hetzner 1M deployment

# Configuration
SERVER="root@${HETZNER_IP:?Set HETZNER_IP env var}"
SSH_KEY="$HOME/.ssh/${HETZNER_SSH_KEY:-engram_hetzner}"
REMOTE_BASE="/data/videos"
STAGING_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    local missing=0

    if ! command -v yt-dlp &> /dev/null; then
        log_error "yt-dlp not found. Install with: brew install yt-dlp"
        missing=1
    fi

    if ! command -v ffmpeg &> /dev/null; then
        log_error "ffmpeg not found. Install with: brew install ffmpeg"
        missing=1
    fi

    if [ ! -f "$SSH_KEY" ]; then
        log_error "SSH key not found at $SSH_KEY"
        missing=1
    fi

    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

# Sanitize filename (spaces, #, parens → underscores)
sanitize_filename() {
    local filename="$1"
    echo "$filename" | sed 's/[# ()]/_/g'
}

# Get human-readable size
human_size() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt 1048576 ]; then
        echo "$(( bytes / 1024 ))KB"
    elif [ $bytes -lt 1073741824 ]; then
        echo "$(( bytes / 1048576 ))MB"
    else
        echo "$(( bytes / 1073741824 ))GB"
    fi
}

# Download videos from URL file
download_videos() {
    local url_file="$1"
    local category="$2"

    if [ ! -f "$url_file" ]; then
        log_error "URL file not found: $url_file"
        exit 1
    fi

    local raw_dir="$STAGING_BASE/$category/raw"
    mkdir -p "$raw_dir"

    log_info "Downloading videos for category: $category"
    log_info "Reading URLs from: $url_file"
    log_info "Output directory: $raw_dir"

    # Count total URLs (excluding comments and empty lines)
    local total_urls=$(grep -v '^#' "$url_file" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
    local current=0
    local downloaded=0
    local skipped=0
    local failed=0

    while IFS= read -r url || [ -n "$url" ]; do
        # Skip comments and empty lines
        if [[ "$url" =~ ^#.*$ ]] || [[ -z "${url// }" ]]; then
            continue
        fi

        current=$((current + 1))
        log_info "[$current/$total_urls] Processing: $url"

        # Download with yt-dlp
        # Format: best video+audio under 720p, prefer mp4
        # Output template: title with ID to avoid conflicts
        if yt-dlp \
            --format 'bestvideo[height<=720]+bestaudio/best[height<=720]/best' \
            --merge-output-format mp4 \
            --output "$raw_dir/%(title)s-%(id)s.%(ext)s" \
            --no-playlist \
            --no-overwrites \
            "$url" 2>&1 | tee /tmp/yt-dlp-$$.log; then

            if grep -q "has already been downloaded" /tmp/yt-dlp-$$.log; then
                log_warn "Already downloaded, skipping"
                skipped=$((skipped + 1))
            else
                log_success "Downloaded successfully"
                downloaded=$((downloaded + 1))
            fi
        else
            log_error "Failed to download: $url"
            failed=$((failed + 1))
        fi

        rm -f /tmp/yt-dlp-$$.log

    done < "$url_file"

    echo ""
    log_success "Download complete!"
    log_info "Downloaded: $downloaded, Skipped: $skipped, Failed: $failed"
}

# Compress videos
compress_videos() {
    local category="$1"
    local raw_dir="$STAGING_BASE/$category/raw"
    local compressed_dir="$STAGING_BASE/$category/compressed"

    if [ ! -d "$raw_dir" ]; then
        log_error "Raw directory not found: $raw_dir"
        exit 1
    fi

    mkdir -p "$compressed_dir"

    # Find all video files (null-delimited to handle spaces in filenames)
    local videos=()
    while IFS= read -r -d '' f; do
        videos+=("$f")
    done < <(find "$raw_dir" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" -o -name "*.avi" -o -name "*.mov" \) -print0)
    local total=${#videos[@]}

    if [ $total -eq 0 ]; then
        log_warn "No videos found in $raw_dir"
        exit 0
    fi

    log_info "Compressing $total videos for category: $category"
    log_info "Output directory: $compressed_dir"

    local current=0
    local compressed=0
    local skipped=0
    local failed=0
    local total_size_before=0
    local total_size_after=0

    for video in "${videos[@]}"; do
        current=$((current + 1))

        local basename=$(basename "$video")
        local name="${basename%.*}"
        local sanitized_name=$(sanitize_filename "$name")
        local output="$compressed_dir/${sanitized_name}.mp4"

        log_info "[$current/$total] Processing: $basename"

        # Skip if already compressed
        if [ -f "$output" ]; then
            log_warn "Already compressed, skipping"
            skipped=$((skipped + 1))
            continue
        fi

        # Get original size
        local size_before=$(stat -f%z "$video" 2>/dev/null || stat -c%s "$video" 2>/dev/null)
        total_size_before=$((total_size_before + size_before))

        # Compress: 128x128 grayscale, crf 32, 10fps
        # Brain uses 64x64 but CNN preprocessor downscales from 128x128
        if ffmpeg -i "$video" \
            -vf "scale=128:128,format=gray" \
            -c:v libx264 \
            -crf 32 \
            -r 10 \
            -an \
            -y \
            "$output" \
            -loglevel error -stats; then

            local size_after=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null)
            total_size_after=$((total_size_after + size_after))

            local ratio=$((100 - (size_after * 100 / size_before)))
            log_success "Compressed: $(human_size $size_before) → $(human_size $size_after) (${ratio}% reduction)"
            compressed=$((compressed + 1))
        else
            log_error "Failed to compress: $basename"
            failed=$((failed + 1))
            rm -f "$output"
        fi
    done

    echo ""
    log_success "Compression complete!"
    log_info "Compressed: $compressed, Skipped: $skipped, Failed: $failed"
    if [ $total_size_before -gt 0 ]; then
        local overall_ratio=$((100 - (total_size_after * 100 / total_size_before)))
        log_info "Total size: $(human_size $total_size_before) → $(human_size $total_size_after) (${overall_ratio}% reduction)"
    fi
}

# Upload videos to server
upload_videos() {
    local category="$1"
    local subdir="${2:-$category}"
    local compressed_dir="$STAGING_BASE/$category/compressed"
    local remote_dir="$REMOTE_BASE/$subdir"

    if [ ! -d "$compressed_dir" ]; then
        log_error "Compressed directory not found: $compressed_dir"
        exit 1
    fi

    # Count files and calculate total size
    local file_count=$(find "$compressed_dir" -type f -name "*.mp4" | wc -l | tr -d ' ')
    if [ $file_count -eq 0 ]; then
        log_warn "No compressed videos found in $compressed_dir"
        exit 0
    fi

    local total_size=0
    while IFS= read -r file; do
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        total_size=$((total_size + size))
    done < <(find "$compressed_dir" -type f -name "*.mp4")

    log_info "Uploading $file_count videos for category: $category"
    log_info "Total size: $(human_size $total_size)"
    log_info "Remote directory: $remote_dir"

    # Confirm upload
    echo -n "Proceed with upload? [y/N] "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_warn "Upload cancelled"
        exit 0
    fi

    # Create remote directory
    ssh -i "$SSH_KEY" "$SERVER" "mkdir -p $remote_dir" || {
        log_error "Failed to create remote directory"
        exit 1
    }

    # Upload all files
    log_info "Starting upload..."
    if scp -i "$SSH_KEY" -r "$compressed_dir"/*.mp4 "$SERVER:$remote_dir/"; then
        log_success "Upload complete!"
        echo ""
        log_warn "REMINDER: Restart gateway to pick up new content"
        log_info "SSH command: ssh -i $SSH_KEY $SERVER"
        log_info "Then run: tmux attach -t gateway"
        log_info "Press Ctrl+C to stop, then restart gateway"
    else
        log_error "Upload failed"
        exit 1
    fi
}

# Show status of all categories
show_status() {
    log_info "Local staging status:"
    echo ""

    # Find all categories
    local categories=()
    while IFS= read -r d; do
        categories+=("$(basename "$d")")
    done < <(find "$STAGING_BASE" -mindepth 1 -maxdepth 1 -type d | sort)

    if [ ${#categories[@]} -eq 0 ]; then
        log_warn "No categories found in $STAGING_BASE"
        echo ""
    else
        for category in "${categories[@]}"; do
            echo -e "${GREEN}$category${NC}"

            local raw_dir="$STAGING_BASE/$category/raw"
            local compressed_dir="$STAGING_BASE/$category/compressed"

            # Raw videos
            if [ -d "$raw_dir" ]; then
                local raw_count=$(find "$raw_dir" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" -o -name "*.avi" -o -name "*.mov" \) | wc -l | tr -d ' ')
                local raw_size=0
                if [ $raw_count -gt 0 ]; then
                    while IFS= read -r file; do
                        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
                        raw_size=$((raw_size + size))
                    done < <(find "$raw_dir" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.webm" -o -name "*.avi" -o -name "*.mov" \))
                fi
                echo "  Raw: $raw_count videos, $(human_size $raw_size)"
            fi

            # Compressed videos
            if [ -d "$compressed_dir" ]; then
                local comp_count=$(find "$compressed_dir" -type f -name "*.mp4" | wc -l | tr -d ' ')
                local comp_size=0
                if [ $comp_count -gt 0 ]; then
                    while IFS= read -r file; do
                        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
                        comp_size=$((comp_size + size))
                    done < <(find "$compressed_dir" -type f -name "*.mp4")
                fi
                echo "  Compressed: $comp_count videos, $(human_size $comp_size)"
            fi

            echo ""
        done
    fi

    # Server status
    log_info "Server status ($SERVER:$REMOTE_BASE):"
    echo ""

    if ssh -i "$SSH_KEY" "$SERVER" "ls -lh $REMOTE_BASE 2>/dev/null" | tail -n +2; then
        echo ""
    else
        log_warn "Could not connect to server or directory not found"
    fi
}

# Run full pipeline
run_all() {
    local url_file="$1"
    local category="$2"

    log_info "Running full pipeline for category: $category"
    echo ""

    download_videos "$url_file" "$category"
    echo ""

    compress_videos "$category"
    echo ""

    upload_videos "$category" "$category"
}

# Curate videos using Gemini API
curate_videos() {
    local args="$@"
    local venv_python="$STAGING_BASE/.venv/bin/python3"

    # Use venv if available, otherwise system python
    if [ -f "$venv_python" ]; then
        local PYTHON="$venv_python"
    elif command -v python3 &> /dev/null; then
        local PYTHON="python3"
    else
        log_error "python3 not found"
        exit 1
    fi

    # Check for google-genai
    if ! "$PYTHON" -c "from google import genai" 2>/dev/null; then
        log_warn "google-genai not installed."
        log_info "Setting up venv..."
        python3 -m venv "$STAGING_BASE/.venv"
        "$STAGING_BASE/.venv/bin/pip" install google-genai
        PYTHON="$venv_python"
    fi

    log_info "Running Gemini video curator..."
    "$PYTHON" "$STAGING_BASE/curate_videos.py" $args
}

# Show usage
usage() {
    cat << EOF
Engram Video Training Pipeline

Usage:
  $0 curate [--all | --category narrated|visual|causeeffect] [--validate] [--run-pipeline]
    Use Gemini + Google Search to find real YouTube training videos

  $0 download <url_file> <category>
    Download videos from YouTube URLs

  $0 compress <category>
    Compress videos to 128x128 grayscale for training

  $0 upload <category> [subdir]
    Upload compressed videos to Hetzner server

  $0 status
    Show local and remote video counts/sizes

  $0 all <url_file> <category>
    Run download → compress → upload in sequence

Arguments:
  <url_file>   Text file with one YouTube URL per line (# for comments)
  <category>   Category name (creates subdirs: <category>/raw, <category>/compressed)
  [subdir]     Optional remote subdirectory (defaults to category name)

Examples:
  $0 curate --all --validate            # Find URLs for all 3 categories
  $0 curate --all --run-pipeline        # Find + download + compress
  $0 curate --category narrated         # Just narrated speech videos
  $0 download urls_narrated.txt juvenile_narrated
  $0 compress juvenile_narrated
  $0 upload juvenile_narrated
  $0 all urls_sample.txt juvenile_visual

Server: $SERVER
Remote base: $REMOTE_BASE
Staging base: $STAGING_BASE

Video format: 128x128 grayscale, 10fps, crf 32 (brain uses 64x64 at 2fps)
EOF
}

# Main
main() {
    check_dependencies

    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        curate)
            curate_videos "$@"
            ;;
        download)
            if [ $# -ne 2 ]; then
                log_error "Usage: $0 download <url_file> <category>"
                exit 1
            fi
            download_videos "$1" "$2"
            ;;
        compress)
            if [ $# -ne 1 ]; then
                log_error "Usage: $0 compress <category>"
                exit 1
            fi
            compress_videos "$1"
            ;;
        upload)
            if [ $# -lt 1 ] || [ $# -gt 2 ]; then
                log_error "Usage: $0 upload <category> [subdir]"
                exit 1
            fi
            upload_videos "$@"
            ;;
        status)
            show_status
            ;;
        all)
            if [ $# -ne 2 ]; then
                log_error "Usage: $0 all <url_file> <category>"
                exit 1
            fi
            run_all "$1" "$2"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            usage
            exit 1
            ;;
    esac
}

main "$@"

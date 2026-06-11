# Engram Video Training Pipeline

Automated workflow for downloading, compressing, and uploading training videos to the Hetzner 1M deployment.

## Quick Start

```bash
# 1. Create a URL file with YouTube links
cat > my_videos.txt << 'EOF'
# Science experiments
https://www.youtube.com/watch?v=example1
https://www.youtube.com/watch?v=example2
EOF

# 2. Run the full pipeline
./pipeline.sh all my_videos.txt juvenile_science

# Or run steps individually:
./pipeline.sh download my_videos.txt juvenile_science
./pipeline.sh compress juvenile_science
./pipeline.sh upload juvenile_science
```

## Commands

### `download <url_file> <category>`
Downloads videos from YouTube URLs.

- **Input**: Text file with one URL per line (# for comments)
- **Output**: `<category>/raw/` directory with original videos
- **Features**:
  - Best quality under 720p
  - Skips already-downloaded videos
  - Shows progress (N of M)
  - Handles failures gracefully

```bash
./pipeline.sh download urls_robotics.txt juvenile_robotics
```

### `compress <category>`
Compresses videos for training.

- **Input**: Videos in `<category>/raw/`
- **Output**: `<category>/compressed/` directory
- **Format**: 128x128 grayscale, 10fps, crf 32
- **Features**:
  - Sanitizes filenames (spaces/#/parens → underscores)
  - Shows before/after size comparison
  - Typical compression: 50-100x reduction
  - Skips already-compressed videos

```bash
./pipeline.sh compress juvenile_robotics
```

### `upload <category> [subdir]`
Uploads compressed videos to Hetzner server.

- **Input**: Videos in `<category>/compressed/`
- **Output**: `/data/videos/<subdir>/` on server (defaults to category name)
- **Features**:
  - Shows total upload size
  - Confirmation prompt
  - Reminder to restart gateway
  - Creates remote directory if needed

```bash
# Upload to /data/videos/juvenile_robotics/
./pipeline.sh upload juvenile_robotics

# Upload to custom subdirectory
./pipeline.sh upload juvenile_robotics custom_subdir
```

### `status`
Shows local and remote video inventory.

```bash
./pipeline.sh status
```

Output:
- Local: counts and sizes for each category (raw + compressed)
- Server: directory listing from `/data/videos/`

### `all <url_file> <category>`
Runs the full pipeline: download → compress → upload.

```bash
./pipeline.sh all urls_physics.txt juvenile_causeeffect
```

## Category Naming Conventions

Align with brain developmental phases:

- **`juvenile_narrated`** - Educational content with narration (science, how-to, documentaries)
- **`juvenile_visual`** - Visual action sequences (cooking, sports, construction)
- **`juvenile_causeeffect`** - Physics, mechanics, cause-and-effect demonstrations
- **`juvenile_social`** - Human interactions, conversations, social scenarios
- **`adolescent_complex`** - Multi-step tasks, problem-solving, decision trees

## URL File Format

```txt
# Engram Training Videos - Science Category
#
# Lines starting with # are comments
# Empty lines are ignored

# Physics experiments
https://www.youtube.com/watch?v=abc123
https://www.youtube.com/watch?v=def456

# Chemistry demonstrations
https://www.youtube.com/watch?v=ghi789
```

See `urls_sample.txt` for a template.

## Video Processing Details

### Why 128x128?

- Brain uses 64x64 grayscale input
- CNN preprocessor (`cnn_preprocessor.py`) downscales from 128x128 → 64x64
- Compressing to 128x128 preserves detail for the downscaling step
- 10fps gives temporal resolution (brain samples at 2fps via aggregation)

### Filename Sanitization

NATS subjects break on spaces, `#`, and special characters. The script automatically converts:

```
"My Video #5 (final).mp4"  →  "My_Video__5__final_.mp4"
```

### Compression Ratio

Typical results:
- **Input**: 1080p MP4, ~50-200 MB
- **Output**: 128x128 grayscale, ~1-5 MB
- **Reduction**: 50-100x smaller

Example:
```
Science_Experiment.mp4: 156 MB → 2.8 MB (98% reduction)
```

## After Upload

Videos are uploaded to `/data/videos/<category>/` but the gateway won't see them until restarted.

**Restart gateway:**

```bash
# SSH to server
ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP

# Attach to gateway tmux session
tmux attach -t gateway

# Stop gateway (Ctrl+C)
# Then restart with new video directories
python gateway.py --nats nats://localhost:4222 --video /data/videos/*.mp4 --turbo --cnn
```

Or restart remotely:

```bash
ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP \
  "tmux send-keys -t gateway C-c && sleep 2 && tmux send-keys -t gateway 'python gateway.py --nats nats://localhost:4222 --video /data/videos/*.mp4 --turbo --cnn' Enter"
```

## Troubleshooting

### yt-dlp Errors

**Problem**: "ERROR: Video unavailable"
- Video may be deleted, private, or region-restricted
- Pipeline continues with next URL

**Problem**: "HTTP Error 429: Too Many Requests"
- YouTube rate limiting
- Wait 10-15 minutes, then resume

### ffmpeg Errors

**Problem**: "Invalid data found when processing input"
- Corrupted download
- Delete file from `raw/` and re-download

### Upload Errors

**Problem**: "Permission denied (publickey)"
- Check SSH key: `~/.ssh/$HETZNER_SSH_KEY`
- Test connection: `ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP`

**Problem**: "No space left on device"
- Check server disk: `df -h /data`
- Clean up quarantine: `rm -rf /data/videos/_quarantined/*`

## Dependencies

- **yt-dlp**: YouTube downloader (`brew install yt-dlp`)
- **ffmpeg**: Video processing (`brew install ffmpeg`)
- **SSH key**: `~/.ssh/$HETZNER_SSH_KEY` for server access

Check with:
```bash
./pipeline.sh status  # Will check dependencies and exit if missing
```

## Server Configuration

- **Host**: $HETZNER_IP
- **User**: root
- **SSH Key**: `~/.ssh/$HETZNER_SSH_KEY`
- **Remote Base**: `/data/videos/`
- **Gateway Config**: `/data/gateway_config/`

## Performance Notes

### Download
- ~1-5 minutes per video (depends on size and YouTube throttling)
- Network: ~10-50 Mbps sustained

### Compress
- ~30-60 seconds per video (depends on length and CPU)
- CPU: Single-threaded per video (ffmpeg default)

### Upload
- ~1-2 MB/s to Hetzner (depends on local connection)
- 100 videos (~300 MB) ≈ 5 minutes

## Example Workflow

```bash
# Download 20 robotics tutorials
./pipeline.sh download urls_robotics.txt juvenile_robotics

# Compress (20 videos × 45s ≈ 15 minutes)
./pipeline.sh compress juvenile_robotics

# Review compressed videos
ls -lh juvenile_robotics/compressed/

# Upload to server
./pipeline.sh upload juvenile_robotics

# Check server
./pipeline.sh status

# Restart gateway to pick up new content
ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP
tmux attach -t gateway
# Ctrl+C, then restart
```

## Directory Structure

```
staging_videos/
├── pipeline.sh              # This script
├── urls_sample.txt          # Example URL file
├── README.md                # This file
├── <category1>/
│   ├── raw/                 # Original downloads
│   └── compressed/          # Compressed for upload
├── <category2>/
│   ├── raw/
│   └── compressed/
└── ...
```

## Gateway Integration

Once uploaded, videos are discovered by `sensory-gateway/discovery.py`:

1. Gateway scans `/data/videos/` on startup
2. Creates `VideoFileSensor` for each `.mp4`
3. Publishes to NATS: `observation.visual.<filename>`
4. Brain receives 64x64 grayscale frames at 2fps (after CNN + aggregation)

See `sensory-gateway/README.md` for gateway architecture.

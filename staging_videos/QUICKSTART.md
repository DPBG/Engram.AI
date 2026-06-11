# Quick Start Guide

## 1. First Time Setup

Check dependencies:
```bash
./pipeline.sh status
```

If missing tools:
```bash
brew install yt-dlp ffmpeg
```

## 2. Create Your URL File

Copy the template:
```bash
cp urls_template.txt urls_robotics.txt
```

Edit and add YouTube URLs:
```bash
nano urls_robotics.txt
```

Example:
```txt
# Robotics tutorials
https://www.youtube.com/watch?v=abc123
https://www.youtube.com/watch?v=def456
https://www.youtube.com/watch?v=ghi789
```

## 3. Run the Pipeline

**Option A: All-in-one (recommended)**
```bash
./pipeline.sh all urls_robotics.txt juvenile_robotics
```

**Option B: Step-by-step**
```bash
# Download (~5 min for 10 videos)
./pipeline.sh download urls_robotics.txt juvenile_robotics

# Compress (~7 min for 10 videos)
./pipeline.sh compress juvenile_robotics

# Upload (~2 min for 10 videos)
./pipeline.sh upload juvenile_robotics
```

## 4. Restart Gateway on Server

After upload, the gateway needs to restart to discover new videos:

```bash
ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP
```

Then in the SSH session:
```bash
tmux attach -t gateway
# Press Ctrl+C to stop
# Gateway will auto-restart, or manually restart with:
# python sensory_gateway.py --config-dir /data/gateway_config
```

## 5. Verify Videos Are Loading

Watch the gateway logs for new sensors:
```
Discovered: VideoFileSensor sensor.video.My_Video_Title
```

Check brain logs for visual observations:
```bash
ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP
docker logs -f neuromorphic --tail 100
```

Look for:
```
observation.visual messages
sensory_buffer activity in visual neurons
```

## Multiple Categories

Process several URL files at once:
```bash
./batch_process.sh urls_science.txt urls_robotics.txt urls_nature.txt
```

This will:
1. Download all videos from each file
2. Compress each category
3. Upload each category
4. Show final status

## Common Workflows

### Quick Test (1-2 videos)
```bash
# Create test file
echo "https://www.youtube.com/watch?v=EXAMPLE" > urls_test.txt

# Process
./pipeline.sh all urls_test.txt test_category

# Upload to temporary location
./pipeline.sh upload test_category temp_test
```

### Large Batch (100+ videos)
```bash
# Download first (can resume if interrupted)
./pipeline.sh download urls_large.txt juvenile_large

# Compress overnight (CPU intensive)
./pipeline.sh compress juvenile_large

# Upload when ready
./pipeline.sh upload juvenile_large
```

### Update Existing Category
```bash
# New videos for existing category
./pipeline.sh download urls_more_robotics.txt juvenile_robotics_new

# Compress
./pipeline.sh compress juvenile_robotics_new

# Upload to same directory as original
./pipeline.sh upload juvenile_robotics_new juvenile_robotics
```

## Checking Progress

**Local status:**
```bash
./pipeline.sh status
```

Shows:
- How many videos in each category
- Raw vs compressed sizes
- What's already on the server

**Server disk space:**
```bash
ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP df -h /data
```

**Gateway sensor count:**
```bash
ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP
tmux attach -t gateway
# Look for "Total sensors: XXX" in startup logs
```

## Troubleshooting

**"Video unavailable" errors:**
- Video might be deleted or private
- Script continues with next URL
- Check URL in browser

**"Too many requests" from YouTube:**
- Wait 10-15 minutes
- Resume with same command (skips already-downloaded)

**Upload fails:**
- Check SSH key: `ssh -i ~/.ssh/$HETZNER_SSH_KEY root@$HETZNER_IP`
- Check server disk: `df -h /data`

**Videos not appearing in brain:**
- Did you restart gateway?
- Check gateway logs: `tmux attach -t gateway`
- Verify files exist: `ssh root@$HETZNER_IP ls -lh /data/videos/juvenile_robotics/`

## File Organization

Recommended structure:
```
staging_videos/
├── urls_science.txt         # Physics, chemistry, biology
├── urls_robotics.txt        # Robot demos, automation
├── urls_nature.txt          # Animals, plants, weather
├── urls_cooking.txt         # Food preparation
├── urls_sports.txt          # Athletic movements
└── urls_social.txt          # Human interactions
```

Each creates:
```
staging_videos/
├── juvenile_science/
│   ├── raw/           # Original downloads (keep for backup)
│   └── compressed/    # Ready to upload (can delete after upload)
├── juvenile_robotics/
│   ├── raw/
│   └── compressed/
└── ...
```

## Best Practices

1. **Test small batches first** - Try 2-3 videos before downloading 100
2. **Keep raw files** - Don't delete until you verify upload worked
3. **Organize by theme** - Easier to balance training data
4. **Check server space** - 256 GB total, ~150 GB available
5. **Monitor brain logs** - Verify videos are actually processed
6. **Name categories clearly** - `juvenile_X` for developmental phase alignment

## Performance Expectations

| Step | Time per Video | Bottleneck |
|------|----------------|------------|
| Download | 1-3 min | YouTube throttling |
| Compress | 30-60 sec | CPU (single-threaded) |
| Upload | 10-20 sec | Network (1-2 MB/s) |

**Example: 50 videos**
- Download: ~75 minutes
- Compress: ~40 minutes
- Upload: ~10 minutes
- **Total**: ~2 hours

## Next Steps

After uploading videos:

1. **Monitor training** - Watch brain logs for concept formation
2. **Balance categories** - Adjust video mix based on brain performance
3. **Add complexity** - Move from juvenile to adolescent content as brain matures
4. **Check prediction error** - High error = need more examples in that domain

See `README.md` for detailed documentation.

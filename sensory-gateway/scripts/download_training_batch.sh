#!/usr/bin/env bash
# Download additional training videos for brain training content expansion.
# Run on the Hetzner server: bash download_training_batch.sh
#
# Categories designed for robotic brain development:
# - Object manipulation, faces, animals, nature, movement, spatial, social
#
# Videos are short (1-10 min), 720p max, diverse visual content.
# Each video provides novel patterns that drive stronger STDP weight changes.

set -euo pipefail

OUTPUT_DIR="/data/videos/tier4"
mkdir -p "$OUTPUT_DIR"

# Check yt-dlp is available
if ! command -v yt-dlp &>/dev/null; then
    echo "ERROR: yt-dlp not found. Install with: pip install yt-dlp"
    exit 1
fi

download() {
    local url="$1"
    local id
    id=$(echo "$url" | grep -oP '(?:v=|youtu\.be/)([^&?]+)' | head -1 | sed 's/v=//')
    if [ -z "$id" ]; then
        id=$(echo "$url" | md5sum | cut -c1-11)
    fi

    if [ -f "$OUTPUT_DIR/${id}.mp4" ]; then
        echo "SKIP: $id already downloaded"
        return 0
    fi

    echo "Downloading: $url → $OUTPUT_DIR/${id}.mp4"
    yt-dlp \
        -f "bestvideo[height<=720]+bestaudio/best[height<=720]/best" \
        --merge-output-format mp4 \
        -o "$OUTPUT_DIR/%(id)s.%(ext)s" \
        --no-playlist \
        --quiet --no-warnings \
        "$url" 2>/dev/null || {
        echo "  FAILED: $url"
        return 0
    }
    echo "  OK: $id"
}

echo "=== Baby/Infant Development (visual + motor patterns) ==="
# Baby first words, object exploration, crawling
download "https://www.youtube.com/watch?v=p_IjkbM3K4I"  # Baby sign language basics
download "https://www.youtube.com/watch?v=dSWbHK0mI4o"  # Baby sensory stimulation
download "https://www.youtube.com/watch?v=0qg8VvOdMVw"  # High contrast baby visual
download "https://www.youtube.com/watch?v=63dFPVhRV0A"  # Baby learning to walk
download "https://www.youtube.com/watch?v=wG2-y5Yf1Oo"  # Infant visual tracking

echo "=== Object Manipulation ==="
# Hands interacting with objects — core for robotic manipulation learning
download "https://www.youtube.com/watch?v=0Y1HqGN3v0A"  # Building blocks
download "https://www.youtube.com/watch?v=xJVGFb4MxNc"  # Origami folding
download "https://www.youtube.com/watch?v=Q_qdQImtV-c"  # Pottery wheel
download "https://www.youtube.com/watch?v=8lEQU6_l8lU"  # Lock picking (fine motor)
download "https://www.youtube.com/watch?v=SQs7v3gLoRg"  # Woodworking
download "https://www.youtube.com/watch?v=nT5_cetZzwA"  # Cooking prep (chopping, mixing)
download "https://www.youtube.com/watch?v=7dkkCW9JLCo"  # Rubik's cube solve
download "https://www.youtube.com/watch?v=hqKafI7Amd8"  # Pen spinning tricks
download "https://www.youtube.com/watch?v=gUQpKl5dGXo"  # Card shuffling
download "https://www.youtube.com/watch?v=5GIa9_G99jQ"  # Drawing/sketching

echo "=== Faces and Emotions ==="
download "https://www.youtube.com/watch?v=3qYGzaH7uJI"  # Facial expressions guide
download "https://www.youtube.com/watch?v=J3wS1N4nM-M"  # Baby facial expressions
download "https://www.youtube.com/watch?v=g-T1bz16DL0"  # Human emotions compilation
download "https://www.youtube.com/watch?v=f2yoftvJp_8"  # Sign language alphabet
download "https://www.youtube.com/watch?v=bDkQAvTqdGE"  # Lip reading basics

echo "=== Animals and Nature ==="
download "https://www.youtube.com/watch?v=UrMr4iZPFz8"  # Cat compilation
download "https://www.youtube.com/watch?v=z_AbfPXTKms"  # Dog tricks
download "https://www.youtube.com/watch?v=nQhklTMn-Hk"  # Birds flying slow motion
download "https://www.youtube.com/watch?v=6v2L2UGZJAM"  # Fish underwater
download "https://www.youtube.com/watch?v=BZP1rYjoBgI"  # Insects macro
download "https://www.youtube.com/watch?v=HMYkBz5ELNI"  # Flowers blooming timelapse
download "https://www.youtube.com/watch?v=SlPhMPnQ58k"  # Waves crashing
download "https://www.youtube.com/watch?v=dqT-UlYlg1s"  # Rainstorm
download "https://www.youtube.com/watch?v=2OEL4P1Rz04"  # Clouds timelapse
download "https://www.youtube.com/watch?v=CW0DUg63lqU"  # Northern lights

echo "=== Movement and Locomotion ==="
download "https://www.youtube.com/watch?v=6w1AxhFMN4U"  # Parkour compilation
download "https://www.youtube.com/watch?v=8alxBofd_eQ"  # Gymnastics
download "https://www.youtube.com/watch?v=Cj9aZ-OAXIM"  # Dance compilation
download "https://www.youtube.com/watch?v=F7u0VN7dCJ8"  # Robot walking
download "https://www.youtube.com/watch?v=fn3KWM1kuAw"  # Boston Dynamics robots
download "https://www.youtube.com/watch?v=_sBBaNYex3E"  # Martial arts forms

echo "=== Spatial Navigation and Environments ==="
download "https://www.youtube.com/watch?v=1La4QzGeaaQ"  # Walking tour city
download "https://www.youtube.com/watch?v=DHdmKj3Ubhk"  # Drone FPV flight
download "https://www.youtube.com/watch?v=bGGDQ7GGkYI"  # Forest walk POV
download "https://www.youtube.com/watch?v=QC8iQqtG0hg"  # House tour
download "https://www.youtube.com/watch?v=ChOhcHD8fBA"  # Factory tour

echo "=== Tools, Kitchen, Household ==="
download "https://www.youtube.com/watch?v=R1ikWbHfvqE"  # Kitchen gadgets
download "https://www.youtube.com/watch?v=ZxEZ2rPtUn8"  # How things are made
download "https://www.youtube.com/watch?v=tP50uSHLC7g"  # Repair/fix things
download "https://www.youtube.com/watch?v=d86ws7mQYIg"  # Simple machines
download "https://www.youtube.com/watch?v=kUfF2MTnqAw"  # Circuits and electronics

echo "=== Textures and Materials ==="
download "https://www.youtube.com/watch?v=b6z6GDwJCOc"  # ASMR textures (visual)
download "https://www.youtube.com/watch?v=MIw-GCZprKk"  # Sand art
download "https://www.youtube.com/watch?v=1MieluM0c2c"  # Glass blowing
download "https://www.youtube.com/watch?v=qcp-sPXOhtk"  # Blacksmithing
download "https://www.youtube.com/watch?v=GZ-aGDyhyrA"  # Weaving/textile
download "https://www.youtube.com/watch?v=G06wa7n4BBI"  # Painting process

echo "=== Science and Patterns ==="
download "https://www.youtube.com/watch?v=4b33NTAuF5E"  # Chemical reactions
download "https://www.youtube.com/watch?v=2SoGKxWGyHs"  # Pendulum waves
download "https://www.youtube.com/watch?v=6UlsArvbTeo"  # Magnetic fluid
download "https://www.youtube.com/watch?v=PvGY_mOWfPE"  # Microscope footage
download "https://www.youtube.com/watch?v=_FlV6pgwlrk"  # Fractals visualization
download "https://www.youtube.com/watch?v=kkGeOWYOFoA"  # Slow motion physics

echo "=== Social Interaction ==="
download "https://www.youtube.com/watch?v=eWLv5z1gCwY"  # People greeting
download "https://www.youtube.com/watch?v=gZjbnGPGQOo"  # Group activities
download "https://www.youtube.com/watch?v=NpmIqRhWuXg"  # Parent-child play
download "https://www.youtube.com/watch?v=F2bk_9T482g"  # Team sports
download "https://www.youtube.com/watch?v=Xl0OMN2DJDQ"  # Musical performance

echo "=== Abstract/Geometric ==="
download "https://www.youtube.com/watch?v=MX0K32U2oP8"  # Geometric animations
download "https://www.youtube.com/watch?v=sD1xiNLzy3A"  # Op art / visual illusions
download "https://www.youtube.com/watch?v=0jGaio87u3A"  # Kaleidoscope
download "https://www.youtube.com/watch?v=ewOBXph0hP4"  # Abstract art
download "https://www.youtube.com/watch?v=WkmPDOq2WfA"  # Symmetry in nature

# Count results
TOTAL=$(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo ""
echo "=== Download complete: $TOTAL videos in $OUTPUT_DIR ==="
echo "Total disk usage:"
du -sh /data/videos/
echo ""
echo "Restart gateway to include new videos:"
echo "  tmux kill-session -t gateway"
echo "  cd /root/engram/sensory-gateway"
echo "  tmux new-session -d -s gateway '.venv/bin/python gateway.py --nats nats://localhost:4222 --no-camera --no-mic --cnn --video /data/videos/tier1/*.mp4 /data/videos/tier2/*.mp4 /data/videos/tier3/*.mp4 /data/videos/tier4/*.mp4 --video-loop --video-fps 2 --sparse-threshold 0.005 -v'"

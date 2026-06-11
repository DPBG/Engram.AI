#!/usr/bin/env bash
set -eo pipefail
# Download phase-appropriate training videos for 1M neuron PoC.
#
# Videos are organized in 3 tiers matching developmental phases:
#   tier1/ — Infant: high-contrast, single objects, slow, repetitive
#   tier2/ — Toddler: categories, comparison, moderate pace
#   tier3/ — Juvenile: sequences, relationships, complex scenes
#
# Usage:
#   ./deploy/download-videos-1m.sh              # download all tiers
#   ./deploy/download-videos-1m.sh --dir /tmp   # custom output dir
#   ./deploy/download-videos-1m.sh --tier 1     # download only tier 1

set -euo pipefail

OUTPUT_DIR="/data/videos"
TIER_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tier) TIER_FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Engram 1M Training Curriculum ==="
echo "Output: $OUTPUT_DIR"

# Check yt-dlp
if ! command -v yt-dlp &>/dev/null; then
    echo "Installing yt-dlp..."
    pip3 install yt-dlp
fi

download_tier() {
    local tier_name="$1"
    local tier_dir="$OUTPUT_DIR/$tier_name"
    shift
    local urls=("$@")

    mkdir -p "$tier_dir"
    echo ""
    echo "=== Downloading $tier_name (${#urls[@]} videos) ==="

    for url in "${urls[@]}"; do
        echo "  --- $url ---"
        yt-dlp \
            -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
            --merge-output-format mp4 \
            --max-filesize 200M \
            -o "$tier_dir/%(id)s.mp4" \
            --no-playlist \
            "$url" || echo "  WARN: Failed to download $url (continuing...)"
    done

    local count
    count=$(ls "$tier_dir"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
    echo "  $tier_name: $count videos downloaded"
}

# =========================================================================
# TIER 1: INFANT PHASE
# High-contrast, single objects, slow pace, repetitive, simple audio
# =========================================================================
TIER1=(
    # High-contrast visual stimulation for babies
    "https://www.youtube.com/watch?v=nUnGnh1mSAo"  # Baby Sensory - High Contrast Shapes
    "https://www.youtube.com/watch?v=dQRjfHYmMPo"  # High Contrast Baby Visual Stimulation
    # Single objects with clear labels
    "https://www.youtube.com/watch?v=Bf9L8sbrFGk"  # First 50 English Words for Toddlers
    "https://www.youtube.com/watch?v=ZONJoCIaHlk"  # First Words for Babies
    # Simple animal sounds (one animal at a time)
    "https://www.youtube.com/watch?v=eUutwwoGL8Q"  # ABC Animals Flashcards with Sounds
    # Single objects — fruits, household items
    "https://www.youtube.com/watch?v=UcGm_PM2IwY"  # Fruits and Vegetables Names
    "https://www.youtube.com/watch?v=QXhK-4jrTjs"  # Baby's First Words - 100 Objects
    # Colors — single color fills screen
    "https://www.youtube.com/watch?v=m0IaV6lUMJs"  # Learn Colors for Babies
    # Simple shapes
    "https://www.youtube.com/watch?v=UqYtj2w6hUk"  # Learn Shapes for Babies
    # Slow music + simple visuals
    "https://www.youtube.com/watch?v=LAqLBOh3Ric"  # Learning Fruit Names for Toddlers
)

# =========================================================================
# TIER 2: TODDLER PHASE
# Category groups, comparison, moderate pace, multi-object scenes
# =========================================================================
TIER2=(
    # Farm animals with sounds (groups)
    "https://www.youtube.com/watch?v=LGtYpl17VaI"  # Farm Animals for Kids
    # Wild animals
    "https://www.youtube.com/watch?v=OwRmivbNgQk"  # Wild Animals for Kids
    # Shapes in real-world context
    "https://www.youtube.com/watch?v=dDlpEqJSBfE"  # Shapes in Real Life
    # Body parts with actions
    "https://www.youtube.com/watch?v=BwBJ4clu3io"  # Body Parts for Kids
    # Vehicles with sounds
    "https://www.youtube.com/watch?v=biX7NNxw_w8"  # Means of Transport for Children
    # Counting 1-10 with objects
    "https://www.youtube.com/watch?v=TdDnGjl_jAA"  # Counting 1 to 10 for Kids
    # Action words
    "https://www.youtube.com/watch?v=qLaRh5z0wDY"  # Action Words for Toddlers
    # Food categories
    "https://www.youtube.com/watch?v=frN3nvhIHUk"  # Food Vocabulary for Kids
    # Household items in context
    "https://www.youtube.com/watch?v=7JUBuIVLru4"  # Things in the House for Kids
    # Ocean animals
    "https://www.youtube.com/watch?v=lSUCAOJknNI"  # Sea Animals for Kids
)

# =========================================================================
# TIER 3: JUVENILE PHASE
# Complex scenes, sequences, cause-effect, relationships
# =========================================================================
TIER3=(
    # Daily routine sequences
    "https://www.youtube.com/watch?v=eUXkj6j6JIk"  # Daily Routine for Kids
    # Cause and effect
    "https://www.youtube.com/watch?v=jxBMvy6EzK0"  # Cause and Effect for Kids
    # Odd one out / categorization
    "https://www.youtube.com/watch?v=qH4qqaWiSd4"  # Odd One Out for Kids
    # Counting to 20
    "https://www.youtube.com/watch?v=Aq4UAss33qA"  # Count to 20 for Kids
    # Prepositions (spatial relationships)
    "https://www.youtube.com/watch?v=MGC5RR74FXc"  # Prepositions for Kids
    # Emotions and expressions
    "https://www.youtube.com/watch?v=akTRWJZMks0"  # Emotions for Kids
    # Animal habitats
    "https://www.youtube.com/watch?v=nOFPdKrUst8"  # Where Do Animals Live
    # Opposites (comparison/contrast)
    "https://www.youtube.com/watch?v=piAkskJMbSU"  # Opposites for Kids
)

# Download requested tiers
if [[ -z "$TIER_FILTER" || "$TIER_FILTER" == "1" ]]; then
    download_tier "tier1" "${TIER1[@]}"
fi
if [[ -z "$TIER_FILTER" || "$TIER_FILTER" == "2" ]]; then
    download_tier "tier2" "${TIER2[@]}"
fi
if [[ -z "$TIER_FILTER" || "$TIER_FILTER" == "3" ]]; then
    download_tier "tier3" "${TIER3[@]}"
fi

echo ""
echo "=== Download Summary ==="
for tier in tier1 tier2 tier3; do
    if [[ -d "$OUTPUT_DIR/$tier" ]]; then
        count=$(ls "$OUTPUT_DIR/$tier"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
        size=$(du -sh "$OUTPUT_DIR/$tier" 2>/dev/null | cut -f1)
        echo "  $tier: $count videos ($size)"
    fi
done
total=$(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $total videos"
echo ""
echo "Usage with gateway:"
echo "  Infant:   python gateway.py --video $OUTPUT_DIR/tier1/*.mp4 --video-loop --turbo --cnn"
echo "  Toddler:  python gateway.py --video $OUTPUT_DIR/tier1/*.mp4 $OUTPUT_DIR/tier2/*.mp4 --video-loop --turbo --cnn"
echo "  Juvenile: python gateway.py --video $OUTPUT_DIR/tier1/*.mp4 $OUTPUT_DIR/tier2/*.mp4 $OUTPUT_DIR/tier3/*.mp4 --video-loop --turbo --cnn"

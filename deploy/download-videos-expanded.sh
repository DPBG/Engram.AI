#!/usr/bin/env bash
# Download expanded training curriculum — 27 videos across 10 categories
#
# Usage:
#   ./deploy/download-videos-expanded.sh                    # default /data/videos
#   ./deploy/download-videos-expanded.sh /path/to/videos    # custom dir
#
# Videos are organized into tiers:
#   tier1/ — infant phase (first words, animals, food, objects, shapes)
#   tier2/ — toddler phase (all remaining categories)

set -euo pipefail

OUTPUT_DIR="${1:-/data/videos}"

echo "=== Downloading expanded training curriculum to $OUTPUT_DIR ==="

# Check yt-dlp
if ! command -v yt-dlp &>/dev/null; then
    echo "Installing yt-dlp..."
    pip3 install yt-dlp
fi

download() {
    local tier="$1"
    local id="$2"
    local desc="$3"
    local dir="$OUTPUT_DIR/$tier"
    mkdir -p "$dir"

    if [ -f "$dir/$id.mp4" ]; then
        echo "  SKIP (exists): $dir/$id.mp4 — $desc"
        return 0
    fi

    echo ""
    echo "--- [$tier] $desc ---"
    yt-dlp \
        -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
        --merge-output-format mp4 \
        --max-filesize 200M \
        -o "$dir/%(id)s.mp4" \
        --no-playlist \
        "https://www.youtube.com/watch?v=$id" \
        || echo "  WARN: Failed to download $id ($desc)"
}

# ============================================================
# TIER 1 — Infant phase (already running + 1 new shapes video)
# ============================================================

# These 5 are already downloaded, will be skipped:
download tier1 Bf9L8sbrFGk "First 50 English Words for Toddlers [first_words]"
download tier1 eUutwwoGL8Q "ABC Animals Flashcards with Animal Sounds [animals]"
download tier1 ZONJoCIaHlk "First Words for Babies - Teaching Toddlers [first_words]"
download tier1 QXhK-4jrTjs "Baby's First Words - 100 Objects [objects]"
download tier1 UcGm_PM2IwY "Fruits and Vegetables Names [food]"

# NEW — shapes for infant phase
download tier1 Wt9XXtYYmk8 "Learn Shapes for Kids - Video Flash Cards [shapes]"

# ============================================================
# TIER 2 — Toddler phase (all categories)
# ============================================================

# Shapes (2 more)
download tier2 D8ilAn5ORiU "Shape Flashcards - Totcards 4K [shapes]"
download tier2 jlzX8jt0Now "Learn Shapes Circle Square Triangle [shapes]"

# Animals — Farm (2)
download tier2 V_F75acghoA "Farm Animals - Talking Flashcards [animals_farm]"
download tier2 bV8MSaYlSbc "Learn Farm Animals - Video Flash Cards [animals_farm]"

# Animals — Wild/Safari (1)
download tier2 CA6Mofzh7jo "Wild Animals - Vocabulary for Kids [animals_wild]"

# Animals — Sea/Ocean (2)
download tier2 KI685CvNUrw "Sea Animals Flashcards for Toddlers [animals_sea]"
download tier2 Oxw6FoUNeT4 "Sea Animals - Kids Vocabulary [animals_sea]"

# Objects / Household (2)
download tier2 vst1Rqf5oPU "Home Goods Flashcards for Toddlers [objects]"
download tier2 c-aiPRcTpao "Baby First Words - Flash Cards [objects]"

# Colors (2)
download tier2 qi3axJ9POnw "Learn Colors - Talking Flashcards [colors]"
download tier2 Ukbin1-HCtM "Colour Flashcards - Totcards 4K [colors]"

# Body Parts (2)
download tier2 SUt8q0EKbms "Body Parts - Kids Vocabulary [body_parts]"
download tier2 3MDw42FKO0w "Body Parts Name Flashcards for Kids [body_parts]"

# Action Verbs (2)
download tier2 4c6FyuetSVo "Action Verbs - Kids Vocabulary [actions]"
download tier2 MbnKa3_dOT4 "VERBS for Kids - Walk Jump Eat [actions]"

# Vehicles (3 — includes 2 originally planned)
download tier2 biX7NNxw_w8 "Means of Transport for Children [vehicles]"
download tier2 ACDZebcb2R0 "Learn Vehicles - Talking Flashcards [vehicles]"
download tier2 O_vcly1l3aU "100 Vehicle Names for Toddlers [vehicles]"

# Food (1 originally planned)
download tier2 LAqLBOh3Ric "Learning Fruit Names for Toddlers [food]"

# Numbers (1)
download tier2 D0Ajq682yrA "Number Song 1-20 - Singing Walrus [numbers]"

# Alphabet (1)
download tier2 _R0xRT2y7Uo "Learn The Alphabet Letters Phonics [alphabet]"

# ============================================================
# Summary
# ============================================================

echo ""
echo "=== Downloads complete ==="
echo ""
echo "TIER 1 (infant):"
ls -lh "$OUTPUT_DIR/tier1/"*.mp4 2>/dev/null || echo "  No tier1 videos"
echo ""
echo "TIER 2 (toddler):"
ls -lh "$OUTPUT_DIR/tier2/"*.mp4 2>/dev/null || echo "  No tier2 videos"
echo ""

TOTAL_T1=$(ls "$OUTPUT_DIR/tier1/"*.mp4 2>/dev/null | wc -l)
TOTAL_T2=$(ls "$OUTPUT_DIR/tier2/"*.mp4 2>/dev/null | wc -l)
echo "Tier 1: $TOTAL_T1 videos"
echo "Tier 2: $TOTAL_T2 videos"
echo "Total:  $((TOTAL_T1 + TOTAL_T2)) videos"
echo "Size:   $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)"

#!/usr/bin/env python3
"""Download, compress, and stage YouTube videos for brain training.

Downloads at 480p, compresses to brain-friendly format (128x128 gray, 10fps),
sanitizes filenames, and stages for upload to Hetzner.

Usage:
    # Download all categories
    python scripts/youtube_content.py download

    # Download one category
    python scripts/youtube_content.py download --category science

    # Compress already-downloaded raw files
    python scripts/youtube_content.py compress

    # Upload compressed files to Hetzner
    python scripts/youtube_content.py upload

    # Full pipeline
    python scripts/youtube_content.py all

Disk-aware: downloads one video at a time, compresses immediately,
deletes raw file before downloading next. Needs ~2 GB free.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ── Staging directories ──────────────────────────────────────────────
STAGING_ROOT = Path(
    os.environ.get("STAGING_ROOT", str(Path.home() / "engram-staging" / "staging_videos"))
)
RAW_DIR = STAGING_ROOT / "raw"
COMPRESSED_DIR = STAGING_ROOT / "compressed"

HETZNER_HOST = f"root@{os.environ.get('HETZNER_IP', '127.0.0.1')}"
HETZNER_KEY = f"~/.ssh/{os.environ.get('HETZNER_SSH_KEY', 'engram_hetzner')}"
HETZNER_VIDEO_DIR = "/data/videos"

# ── Content Catalog ──────────────────────────────────────────────────
# Each entry: (url, suggested_name, max_duration_seconds or None)
# max_duration: trim to this length. None = full video.
# Names must be ASCII, no spaces.
# ALL URLs verified working as of 2026-03-05.

CATALOG: dict[str, list[tuple[str, str, int | None]]] = {
    # ── SCIENCE & ENGINEERING EXPLAINERS ─────────────────────────────
    "science": [
        # Veritasium (verified)
        ("https://www.youtube.com/watch?v=bHIhgxav9LY", "veritasium_how_electricity_works", None),
        ("https://www.youtube.com/watch?v=t-_VPRCtiUg", "veritasium_synchronization", None),
        ("https://www.youtube.com/watch?v=XRr1kaXKBsU", "veritasium_gravity_misconceptions", None),
        ("https://www.youtube.com/watch?v=HeQX2HjkcNo", "veritasium_maths_fundamental_flaw", None),
        ("https://www.youtube.com/watch?v=MFzDaBzBlL0", "smarter_backwards_brain_bicycle", None),
        ("https://www.youtube.com/watch?v=QtxVdC7pBQM", "veritasium_jet_engines", None),
        ("https://www.youtube.com/watch?v=AGglJehON5g", "veritasium_perfect_battery", None),
        ("https://www.youtube.com/watch?v=cUzklzVXJwo", "veritasium_imaginary_numbers", None),
        # 3Blue1Brown (verified)
        ("https://www.youtube.com/watch?v=aircAruvnKk", "3b1b_neural_network_ch1", None),
        ("https://www.youtube.com/watch?v=IHZwWFHWa-w", "3b1b_neural_network_ch2", None),
        ("https://www.youtube.com/watch?v=Ilg3gGewQ5U", "3b1b_neural_network_ch3", None),
        ("https://www.youtube.com/watch?v=tIeHLnjs5U8", "3b1b_neural_network_ch4", None),
        ("https://www.youtube.com/watch?v=spUNpyF58BY", "3b1b_fourier_transform", None),
        ("https://www.youtube.com/watch?v=WUvTyaaNkzM", "3b1b_essence_of_calculus", None),
        ("https://www.youtube.com/watch?v=mH0oCDa74tE", "3b1b_group_theory_monster", None),
        # SmarterEveryDay (verified)
        ("https://www.youtube.com/watch?v=AnaASTBn_K4", "smarter_whip_sound_barrier", None),
        ("https://www.youtube.com/watch?v=7CUojMQgDpM", "smarter_curling_physics", None),
        ("https://www.youtube.com/watch?v=j2_dJY_mIys", "smarter_unmixing_color", None),
        ("https://www.youtube.com/watch?v=xe-f4gokRBs", "smarter_prince_ruperts_drop", None),
        # Kurzgesagt (verified)
        ("https://www.youtube.com/watch?v=lXfEK8G8CUI", "kurzgesagt_immune_system", None),
        ("https://www.youtube.com/watch?v=JyECrGp-Sw8", "kurzgesagt_all_nukes_at_once", None),
        ("https://www.youtube.com/watch?v=YbgnlkJPga4", "kurzgesagt_anthropocene_reviewed", None),
        ("https://www.youtube.com/watch?v=9P6rdqiybaw", "kurzgesagt_wormholes", None),
        ("https://www.youtube.com/watch?v=9FppammO1zk", "kurzgesagt_most_dangerous_weapon", None),
        ("https://www.youtube.com/watch?v=wwSzpaTHyS8", "kurzgesagt_paradox_of_time", None),
    ],
    # ── ROBOTICS & AI ────────────────────────────────────────────────
    "robotics": [
        # Boston Dynamics (verified)
        ("https://www.youtube.com/watch?v=tF4DML7FIWk", "bd_atlas", None),
        ("https://www.youtube.com/watch?v=fn3KWM1kuAw", "bd_atlas_parkour_2", None),
        ("https://www.youtube.com/watch?v=29ECwExc-_M", "bd_electric_atlas", None),
        ("https://www.youtube.com/watch?v=-e1_QhJ1EhQ", "bd_spot_launch", None),
        ("https://www.youtube.com/watch?v=I44_zbEwz_w", "bd_walk_run_crawl_rl", None),
        # Figure AI (verified)
        ("https://www.youtube.com/watch?v=0SRVJaOg9Co", "figure_02_demo", None),
        ("https://www.youtube.com/watch?v=Sq1QZB5baNw", "figure_openai_speech", None),
        ("https://www.youtube.com/watch?v=Eu5mYMavctM", "figure_03_intro", None),
        ("https://www.youtube.com/watch?v=jC3R4oxKd0M", "figure_03_whats_new", None),
        # Tesla Optimus (verified)
        ("https://www.youtube.com/watch?v=cpraXaw7dyc", "tesla_optimus_gen2", None),
        # Stuff Made Here (verified)
        ("https://www.youtube.com/watch?v=myO8fxhDRW0", "stuffmadehere_basketball_robot", None),
        ("https://www.youtube.com/watch?v=SjJulcvTA7Y", "stuffmadehere_17k_pounds_tools", None),
        ("https://www.youtube.com/watch?v=ix68oRfI5Gw", "stuffmadehere_chainsaw_robot_arm", None),
        # Unitree (verified)
        ("https://www.youtube.com/watch?v=xwgaMdHzW40", "unitree_g1_spin", None),
        ("https://www.youtube.com/watch?v=a5RKLjZs_ts", "unitree_g1_h2_robots", None),
    ],
    # ── HOW-TO / HANDS-ON ────────────────────────────────────────────
    "howto": [
        # Practical Engineering (verified)
        ("https://www.youtube.com/watch?v=OwS9aTE2Go4", "practical_eng_transistor", None),
        (
            "https://www.youtube.com/watch?v=cZINeaDjisY",
            "practical_eng_concrete_reinforcement",
            None,
        ),
        (
            "https://www.youtube.com/watch?v=VE1wM4oIh8Y",
            "practical_eng_infrastructure_hacked",
            None,
        ),
        ("https://www.youtube.com/watch?v=Y_729CQdG50", "practical_eng_mindblowing_infra", None),
        ("https://www.youtube.com/watch?v=qt2j2gn0yWc", "practical_eng_earthquake_isolation", None),
        # Technology Connections (verified)
        ("https://www.youtube.com/watch?v=OOK5xkFijPc", "techconnections_power_vs_energy", None),
        ("https://www.youtube.com/watch?v=RSTNhvDGbYI", "techconnections_rice_cookers", None),
        ("https://www.youtube.com/watch?v=udNXMAflbU8", "techconnections_holey_plugs", None),
        # Adam Savage / Tested (verified)
        ("https://www.youtube.com/watch?v=cwOXSXkW-uE", "tested_adam_nerf_rifle", None),
        ("https://www.youtube.com/watch?v=G7MDrUG4cws", "tested_adam_1000_nerf", None),
        # Cooking POV (verified)
        ("https://www.youtube.com/watch?v=hB42iztkzVQ", "kenji_french_toast_pizza", None),
        ("https://www.youtube.com/watch?v=9sfm9aqxygs", "kenji_tteokbokki_pov", None),
        ("https://www.youtube.com/watch?v=P6QWoOQMvE8", "kenji_cacio_e_pepe_pov", None),
        # Real Engineering (verified)
        ("https://www.youtube.com/watch?v=6LcGrLnzYuU", "real_eng_oceangate", None),
        ("https://www.youtube.com/watch?v=1lCOgFPtaZ4", "real_eng_f35b", None),
        # Primitive Technology (verified)
        ("https://www.youtube.com/watch?v=P73REgj-3UE", "primitive_tech_tiled_roof_hut", None),
        ("https://www.youtube.com/watch?v=nCKkHqlx9dE", "primitive_tech_wattle_daub_hut", None),
        ("https://www.youtube.com/watch?v=eesj3pJF3lA", "primitive_tech_wood_ash_cement", None),
    ],
    # ── NATURE & MOVEMENT ────────────────────────────────────────────
    "nature": [
        # BBC Earth (verified)
        ("https://www.youtube.com/watch?v=auSo1MyWf8g", "bbc_wonderful_world_attenborough", None),
        ("https://www.youtube.com/watch?v=83fE5YCpvbo", "bbc_earth_cutest_baby_animals", None),
        ("https://www.youtube.com/watch?v=uIKJgxbhAzM", "bbc_earth_spy_camera_animals", None),
        ("https://www.youtube.com/watch?v=zRuuRPM3X3o", "bbc_earth_enchanting_forests", None),
        ("https://www.youtube.com/watch?v=Q2ua2luJW00", "bbc_earth_pika_mountain", None),
        ("https://www.youtube.com/watch?v=ysa5OBhXz-Q", "bbc_wolves_change_rivers", None),
        # Netflix / Nature (verified)
        ("https://www.youtube.com/watch?v=qVJzQc9ELTE", "netflix_our_planet_walrus", None),
        # Wendover (verified)
        ("https://www.youtube.com/watch?v=AW3gaelBypY", "wendover_carbon_offset_problem", None),
        ("https://www.youtube.com/watch?v=y3qfeoqErtY", "wendover_overnight_shipping", None),
        ("https://www.youtube.com/watch?v=iIpPuJ_r8Xg", "wendover_us_military_transport", None),
        ("https://www.youtube.com/watch?v=JrO_tvMjqjo", "wendover_hawaii_logistics", None),
        ("https://www.youtube.com/watch?v=2qanMpnYsjk", "wendover_amazon_shipping", None),
    ],
    # ── CONVERSATIONS & INTERVIEWS ───────────────────────────────────
    "interviews": [
        # Lex Fridman (verified, trim to 30 min)
        ("https://www.youtube.com/watch?v=DxREm3s1scA", "lex_elon_musk", 1800),
        ("https://www.youtube.com/watch?v=jvqFAi7vkBc", "lex_sam_altman_gpt5", 1800),
        ("https://www.youtube.com/watch?v=e-gwvmhyU7A", "lex_aravind_srinivas_perplexity", 1800),
        ("https://www.youtube.com/watch?v=L_Guz73e6fw", "lex_sam_altman_future_ai", 1800),
        ("https://www.youtube.com/watch?v=NNr6gPelJ3E", "lex_roman_yampolskiy_ai_danger", 1800),
        ("https://www.youtube.com/watch?v=Ff4fRgnuFgQ", "lex_mark_zuckerberg", 1800),
        # Huberman Lab (verified, trim to 30 min)
        ("https://www.youtube.com/watch?v=QmOF0crdyRU", "huberman_dopamine_motivation", 1800),
        ("https://www.youtube.com/watch?v=SwQhKFMxmDY", "huberman_change_your_brain", 1800),
        # TED Talks (verified)
        ("https://www.youtube.com/watch?v=8S0FDjFBj8o", "ted_sound_smart_tedx", None),
        ("https://www.youtube.com/watch?v=UF8uR6Z6KLc", "ted_steve_jobs_stanford", None),
        ("https://www.youtube.com/watch?v=arj7oStGLkU", "ted_procrastinator_tim_urban", None),
        ("https://www.youtube.com/watch?v=iCvmsMzlF7o", "ted_vulnerability_brene_brown", None),
        ("https://www.youtube.com/watch?v=H14bBuluwB8", "ted_grit_angela_duckworth", None),
        # Diary of a CEO (verified, trim to 30 min)
        ("https://www.youtube.com/watch?v=rCtvAvZtJyE", "doac_neuroscientist_prediction", 1800),
        ("https://www.youtube.com/watch?v=hCW2NHbWNwA", "doac_neuroscientist_stress", 1800),
        # Joe Rogan (verified, trim to 30 min)
        ("https://www.youtube.com/watch?v=0pmviUS1Zac", "rogan_neil_degrasse_tyson_1347", 1800),
        # MKBHD (verified)
        ("https://www.youtube.com/watch?v=z19HM7ANZlo", "mkbhd_m4_mac_mini", None),
    ],
}

# Total: 80 videos across 5 categories — all URLs verified 2026-03-05


def sanitize_filename(name: str) -> str:
    """Remove non-ASCII, spaces, and special chars from filename."""
    name = re.sub(r"[^\w\-.]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def download_video(url: str, name: str, max_duration: int | None, output_dir: Path) -> Path | None:
    """Download a single video using yt-dlp. Returns path to downloaded file."""
    output_path = output_dir / f"{name}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        "--no-playlist",
        "--no-overwrites",
        "--socket-timeout",
        "30",
        "--retries",
        "3",
    ]

    if max_duration:
        cmd.extend(["--download-sections", f"*0-{max_duration}"])

    cmd.append(url)

    print(f"  Downloading: {name}")
    try:
        dl_timeout = 1200 if max_duration else 600
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=dl_timeout)
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("    TIMEOUT after 10 min")
        return None

    # Find the actual output file
    for ext in ["mp4", "mkv", "webm"]:
        p = output_dir / f"{name}.{ext}"
        if p.exists():
            return p

    # Try glob
    matches = list(output_dir.glob(f"{name}.*"))
    video_matches = [m for m in matches if m.suffix in (".mp4", ".mkv", ".webm", ".m4a")]
    if video_matches:
        return video_matches[0]

    print("    WARNING: Download succeeded but file not found")
    return None


def compress_video(input_path: Path, output_dir: Path) -> Path | None:
    """Compress video to brain-friendly format. Returns compressed path."""
    output_name = sanitize_filename(input_path.stem) + ".mp4"
    output_path = output_dir / output_name

    if output_path.exists():
        print(f"    Already compressed: {output_name}")
        return output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "scale=128:128,format=gray",
        "-c:v",
        "libx264",
        "-crf",
        "32",
        "-r",
        "10",
        "-c:a",
        "aac",
        "-b:a",
        "32k",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"    COMPRESS FAILED: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("    COMPRESS TIMEOUT")
        if output_path.exists():
            output_path.unlink()
        return None

    raw_size = input_path.stat().st_size / (1024 * 1024)
    comp_size = output_path.stat().st_size / (1024 * 1024)
    ratio = raw_size / comp_size if comp_size > 0 else 0
    print(f"    Compressed: {raw_size:.1f} MB → {comp_size:.1f} MB ({ratio:.0f}x)")
    return output_path


def download_category(category: str, entries: list) -> dict:
    """Download and compress all videos in a category."""
    cat_raw = RAW_DIR / category
    cat_compressed = COMPRESSED_DIR / category
    cat_raw.mkdir(parents=True, exist_ok=True)
    cat_compressed.mkdir(parents=True, exist_ok=True)

    stats = {"downloaded": 0, "compressed": 0, "failed": 0, "skipped": 0}

    for url, name, max_dur in entries:
        # Skip if already compressed
        comp_path = cat_compressed / f"{sanitize_filename(name)}.mp4"
        if comp_path.exists():
            print(f"  Skipping (already done): {name}")
            stats["skipped"] += 1
            continue

        # Download
        raw_path = download_video(url, name, max_dur, cat_raw)
        if raw_path is None:
            stats["failed"] += 1
            continue
        stats["downloaded"] += 1

        # Compress immediately (disk space conservation)
        result = compress_video(raw_path, cat_compressed)
        if result:
            stats["compressed"] += 1
        else:
            stats["failed"] += 1

        # Delete raw file to save disk space
        if raw_path.exists():
            raw_path.unlink()
            print("    Cleaned raw file")

    return stats


def cmd_download(args):
    """Download and compress videos."""
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    categories = [args.category] if args.category else list(CATALOG.keys())

    for cat in categories:
        if cat not in CATALOG:
            print(f"Unknown category: {cat}")
            print(f"Available: {', '.join(CATALOG.keys())}")
            sys.exit(1)

        entries = CATALOG[cat]
        print(f"\n{'='*60}")
        print(f"Category: {cat} ({len(entries)} videos)")
        print(f"{'='*60}")
        stats = download_category(cat, entries)
        print(
            f"\n  Results: {stats['downloaded']} downloaded, "
            f"{stats['compressed']} compressed, "
            f"{stats['skipped']} skipped, "
            f"{stats['failed']} failed"
        )

    # Summary
    total_compressed = (
        sum(1 for _ in COMPRESSED_DIR.rglob("*.mp4")) if COMPRESSED_DIR.exists() else 0
    )
    total_size_mb = (
        sum(f.stat().st_size for f in COMPRESSED_DIR.rglob("*.mp4")) / (1024 * 1024)
        if COMPRESSED_DIR.exists()
        else 0
    )
    print(f"\n{'='*60}")
    print(f"Total staged: {total_compressed} videos, {total_size_mb:.1f} MB")
    print(f"Location: {COMPRESSED_DIR}")


def cmd_upload(args):
    """Upload compressed videos to Hetzner."""
    if not COMPRESSED_DIR.exists():
        print("No compressed videos found. Run 'download' first.")
        sys.exit(1)

    categories = (
        [args.category]
        if args.category
        else [d.name for d in COMPRESSED_DIR.iterdir() if d.is_dir()]
    )

    for cat in categories:
        cat_dir = COMPRESSED_DIR / cat
        if not cat_dir.exists():
            continue

        videos = list(cat_dir.glob("*.mp4"))
        if not videos:
            continue

        remote_dir = f"{HETZNER_VIDEO_DIR}/youtube_{cat}"
        print(f"\nUploading {len(videos)} videos to {HETZNER_HOST}:{remote_dir}")

        # Create remote directory
        subprocess.run(
            ["ssh", "-i", os.path.expanduser(HETZNER_KEY), HETZNER_HOST, f"mkdir -p {remote_dir}"],
            check=True,
        )

        # Upload
        cmd = (
            [
                "scp",
                "-i",
                os.path.expanduser(HETZNER_KEY),
            ]
            + [str(v) for v in videos]
            + [f"{HETZNER_HOST}:{remote_dir}/"]
        )

        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"  Uploaded {len(videos)} files to {remote_dir}")
        else:
            print(f"  Upload FAILED for {cat}")

    print("\nTo activate: restart the gateway tmux session on Hetzner")
    print(f"  ssh -i {HETZNER_KEY} {HETZNER_HOST}")
    print("  tmux attach -t gateway")
    print("  # Ctrl-C, then restart with --batch-size 25 --batch-delay 3.0")


def cmd_status(args):
    """Show what's been downloaded/compressed."""
    print(f"Staging root: {STAGING_ROOT}")
    print()

    for cat in CATALOG:
        cat_compressed = COMPRESSED_DIR / cat
        done = len(list(cat_compressed.glob("*.mp4"))) if cat_compressed.exists() else 0
        total = len(CATALOG[cat])
        size_mb = (
            sum(f.stat().st_size for f in cat_compressed.glob("*.mp4")) / (1024 * 1024)
            if cat_compressed.exists()
            else 0
        )
        bar = f"[{'#' * done}{'.' * (total - done)}]"
        print(f"  {cat:15s} {bar} {done}/{total}  ({size_mb:.1f} MB)")

    if COMPRESSED_DIR.exists():
        total_files = sum(1 for _ in COMPRESSED_DIR.rglob("*.mp4"))
        total_mb = sum(f.stat().st_size for f in COMPRESSED_DIR.rglob("*.mp4")) / (1024 * 1024)
        print(f"\n  Total: {total_files} videos, {total_mb:.1f} MB compressed")


def main():
    parser = argparse.ArgumentParser(description="YouTube content pipeline for brain training")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Download and compress videos")
    dl.add_argument("--category", type=str, default=None, help="Single category to download")

    up = sub.add_parser("upload", help="Upload compressed videos to Hetzner")
    up.add_argument("--category", type=str, default=None, help="Single category to upload")

    sub.add_parser("status", help="Show download/compress progress")

    sub.add_parser("all", help="Download, compress, and upload everything")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "all":
        args.category = None
        cmd_download(args)
        cmd_upload(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

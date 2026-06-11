# Directory Structure Guide

The pipeline supports two organizational patterns:

## Pattern 1: Category-First (Recommended for pipeline.sh)

```
staging_videos/
├── juvenile_robotics/
│   ├── raw/          # Original downloads
│   └── compressed/   # Ready to upload
├── juvenile_science/
│   ├── raw/
│   └── compressed/
└── juvenile_nature/
    ├── raw/
    └── compressed/
```

**Usage:**
```bash
./pipeline.sh all urls_robotics.txt juvenile_robotics
./pipeline.sh upload juvenile_robotics
```

**Benefits:**
- Each category is self-contained
- Easy to process one category at a time
- Clear mapping: URL file → category → server directory

## Pattern 2: Stage-First (Existing structure)

```
staging_videos/
├── raw/
│   ├── robotics/
│   ├── science/
│   └── nature/
└── compressed/
    ├── robotics/
    ├── science/
    └── nature/
```

**Usage:**
```bash
# Manual workflow (outside pipeline.sh)
# 1. Download to raw/<category>/
# 2. Compress to compressed/<category>/
# 3. Upload with: scp compressed/robotics/* server:/data/videos/robotics/
```

**Benefits:**
- All raw files in one place
- All compressed files in one place
- Easier to see total storage by stage

## Converting Between Patterns

### Stage-First → Category-First

```bash
#!/bin/bash
# Move existing videos to category-first structure

for category in robotics science nature howto interviews; do
    # Create category directory
    mkdir -p "$category"

    # Move raw if exists
    if [ -d "raw/$category" ]; then
        mv "raw/$category" "$category/raw"
    fi

    # Move compressed if exists
    if [ -d "compressed/$category" ]; then
        mv "compressed/$category" "$category/compressed"
    fi
done

# Remove empty stage directories
rmdir raw compressed 2>/dev/null || true
```

### Category-First → Stage-First

```bash
#!/bin/bash
# Consolidate to stage-first structure

mkdir -p raw compressed

for category_dir in */; do
    category=$(basename "$category_dir")

    # Move raw if exists
    if [ -d "$category_dir/raw" ]; then
        mv "$category_dir/raw" "raw/$category"
    fi

    # Move compressed if exists
    if [ -d "$category_dir/compressed" ]; then
        mv "$category_dir/compressed" "compressed/$category"
    fi

    # Remove category directory if empty
    rmdir "$category_dir" 2>/dev/null || true
done
```

## Recommendations

**For new projects**: Use Category-First (Pattern 1)
- Works seamlessly with `pipeline.sh`
- Clearer organization
- Better for batch processing

**For existing structure**: Keep Stage-First if it works for you
- Already organized
- Can still use pipeline.sh with manual upload
- Or convert to Category-First with script above

## Pipeline Compatibility

The `pipeline.sh` script assumes **Category-First** structure:

```bash
# This works:
./pipeline.sh all urls_robotics.txt juvenile_robotics
# Creates: juvenile_robotics/raw/ and juvenile_robotics/compressed/

# This won't work with existing Stage-First:
./pipeline.sh upload robotics
# Expects: robotics/compressed/, not compressed/robotics/
```

## Workaround for Existing Stage-First Structure

Option 1: **Use manual upload**
```bash
# Instead of: ./pipeline.sh upload robotics
scp -i ~/.ssh/$HETZNER_SSH_KEY compressed/robotics/*.mp4 root@$HETZNER_IP:/data/videos/robotics/
```

Option 2: **Convert to Category-First** (see script above)

Option 3: **Create symlinks**
```bash
# Make pipeline.sh think it's Category-First
ln -s ../raw/robotics robotics_raw
ln -s ../compressed/robotics robotics_compressed

# Then use pipeline.sh (but watch out for symlink issues)
```

## Best Practice for New Videos

1. **Use pipeline.sh with Category-First structure**
   ```bash
   ./pipeline.sh all urls_new.txt juvenile_new_category
   ```

2. **Keep existing videos in Stage-First structure**
   - Don't need to reorganize
   - Already working with current setup
   - Upload manually when needed

3. **Or fully convert to Category-First**
   - Run conversion script above
   - All videos work with pipeline.sh
   - Consistent organization

## Storage Considerations

Both patterns use the same disk space, just organized differently.

**Category-First:**
- Pro: Easier to delete entire categories
- Pro: Self-contained units
- Con: More top-level directories

**Stage-First:**
- Pro: Easier to see total raw vs compressed
- Pro: Fewer top-level directories
- Con: Harder to manage individual categories

Choose based on your workflow preferences.

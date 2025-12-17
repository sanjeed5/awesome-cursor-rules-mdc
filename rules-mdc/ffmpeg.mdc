---
description: This guide defines best practices for integrating and utilizing FFmpeg within our applications, emphasizing API-first development, secure and reproducible builds, and optimal multimedia encoding techniques for quality and performance.
globs: **/*
---
# FFmpeg Best Practices

FFmpeg is the bedrock of our multimedia processing. Adhering to these guidelines ensures our implementations are secure, performant, and maintainable.

## 1. API-First for Application Integration

**Always interact with FFmpeg via its `libav*` APIs (e.g., `libavcodec`, `libavformat`, `libavfilter`) for application-level logic.** The `ffmpeg` command-line tool is for quick tests, scripting, or debugging, *never* for direct invocation within production application code.

❌ **BAD: Shelling out to `ffmpeg` CLI from application code**
```python
import subprocess

# Insecure, slow, hard to manage errors, resource leaks
def transcode_video_bad(input_path, output_path):
    cmd = f"ffmpeg -i {input_path} -c:v libx264 -crf 23 {output_path}"
    subprocess.run(cmd, shell=True, check=True)
```

✅ **GOOD: Using `libav*` APIs (conceptual C/C++)**
```c
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
// ... other libav* headers

// Proper API usage for transcoding (simplified for brevity)
int transcode_video_good(const char* input_path, const char* output_path) {
    AVFormatContext *input_ctx = NULL, *output_ctx = NULL;
    AVCodecContext *decoder_ctx = NULL, *encoder_ctx = NULL;
    // ... Initialize FFmpeg libraries
    // ... Open input file (avformat_open_input)
    // ... Find stream info (avformat_find_stream_info)
    // ... Find best video/audio stream (av_find_best_stream)
    // ... Allocate and open decoder context (avcodec_alloc_context3, avcodec_open2)
    // ... Allocate and open encoder context (avcodec_alloc_context3, avcodec_open2)
    // ... Open output file (avio_open, avformat_write_header)
    // ... Main loop: read packet, decode, filter, encode, write packet
    // ... Write trailer, cleanup (avformat_close_input, avio_closep, etc.)
    return 0; // Or error code
}
```
**Rationale:** Direct API calls offer fine-grained control, robust error handling, better resource management, and eliminate shell injection vulnerabilities.

## 2. Encoding for Quality & Performance

Optimize video encoding for target platforms and quality requirements.

### 2.1 Two-Pass Encoding for Bitrate Control

For any bitrate-constrained output (e.g., streaming, fixed file size targets), **always use two-pass encoding**. This delivers superior quality at a given bitrate compared to single-pass.

❌ **BAD: Single-pass encoding for bitrate-critical scenarios**
```bash
# Inconsistent quality, less efficient bitrate distribution
ffmpeg -i input.mp4 -c:v libx264 -b:v 1M output.mp4
```

✅ **GOOD: Two-pass encoding (using libx264 as an example)**
```bash
# Pass 1: Analyze video, no output
ffmpeg -i input.mp4 -c:v libx264 -b:v 1M -pass 1 -an -f null /dev/null && \
# Pass 2: Encode with optimized settings from pass 1
ffmpeg -i input.mp4 -c:v libx264 -b:v 1M -pass 2 -c:a aac -b:a 128k output.mp4
```
**API Equivalent (conceptual):** Set `AVCodecContext::flags |= AV_CODEC_FLAG_PASS1` for the first pass and `AV_CODEC_FLAG_PASS2` for the second, managing log files between passes.

### 2.2 Keyframe Interval Optimization

Set explicit keyframe intervals (`-g` and `-keyint_min`) to balance seekability and compression efficiency, especially for streaming or interactive playback.

❌ **BAD: Relying on default keyframe intervals for streaming**
```bash
# Suboptimal seeking performance, larger file size for given seekability
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4
```

✅ **GOOD: Explicit keyframe intervals (e.g., 2-second interval for 30fps)**
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -g 60 -keyint_min 60 output.mp4
```
**Rationale:** A consistent keyframe interval improves seeking and stream segmenting for ABR.

### 2.3 Adaptive Bitrate Streaming (ABR)

For web delivery, **always generate multiple renditions for ABR streaming (HLS/DASH)** using modern codecs like AV1 (`libaom-av1`, `libsvtav1`) or HEVC (`libx265`).

```bash
# Example for HLS ABR (simplified for multiple resolutions/bitrates)
# This requires multiple ffmpeg runs or a complex filtergraph
ffmpeg -i input.mp4 \
-map 0:v:0 -map 0:a:0 \
-c:v libx264 -preset medium -g 60 -keyint_min 60 -sc_threshold 0 -b:v 2000k -maxrate 2140k -bufsize 3000k -vf "scale=-2:1080" -c:a aac -b:a 128k \
-hls_time 10 -hls_playlist_type vod -hls_segment_filename "stream_1080p_%03d.ts" stream_1080p.m3u8 \
-map 0:v:0 -map 0:a:0 \
-c:v libx264 -preset medium -g 60 -keyint_min 60 -sc_threshold 0 -b:v 1000k -maxrate 1070k -bufsize 1500k -vf "scale=-2:720" -c:a aac -b:a 96k \
-hls_time 10 -hls_playlist_type vod -hls_segment_filename "stream_720p_%03d.ts" stream_720p.m3u8
# ... and so on for other resolutions, then generate master playlist
```
**Rationale:** ABR provides the best user experience by adapting to network conditions.

## 3. Audio Integrity

For audio-critical workflows (e.g., forensic, archival), **prioritize lossless formats and metadata preservation.**

✅ **GOOD: Lossless audio conversion for forensic/archival**
```bash
# Convert to FLAC, preserving original sample rate and metadata
ffmpeg -i input.mp3 -c:a flac -sample_fmt s32 -ar 48000 output.flac
```
**Rationale:** Lossless formats like FLAC or WAV (PCM) ensure no data degradation. Explicitly setting sample format and rate prevents accidental downsampling.

## 4. Reproducibility & Security

Integrate FFmpeg builds into CI pipelines and manage dependencies rigorously.

### 4.1 Pinning FFmpeg Versions

**Always pin a specific Git commit hash for FFmpeg builds in CI/CD** to ensure reproducibility. Periodically update to the latest master for security patches and new features.

❌ **BAD: Relying on system-installed `ffmpeg` or floating versions**
```dockerfile
# Unpredictable behavior across environments/builds
RUN apt-get update && apt-get install -y ffmpeg
```

✅ **GOOD: Building FFmpeg from a pinned Git commit**
```dockerfile
ARG FFMPEG_COMMIT="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0" # Example commit hash
RUN git clone https://git.ffmpeg.org/ffmpeg.git /usr/src/ffmpeg && \
    cd /usr/src/ffmpeg && \
    git checkout ${FFMPEG_COMMIT} && \
    ./configure --enable-shared --disable-static --enable-gpl --enable-libx264 --enable-libx265 && \
    make -j$(nproc) && make install && ldconfig
```
**Rationale:** Pinning commits guarantees that your build environment uses the exact FFmpeg version, preventing unexpected regressions.

### 4.2 FATE Test Suite Integration

**Run the FFmpeg FATE test suite (or a subset) as part of your CI pipeline** after any FFmpeg-related changes or updates.

**Rationale:** FATE is the upstream project's verification suite, catching regressions and ensuring correctness with untrusted media data.

## 5. Configuration Management

### 5.1 Using Preset Files

Store common or complex FFmpeg configurations in preset files (`.ffpreset` or `.avpreset`) for consistency and maintainability.

❌ **BAD: Long, repetitive command-line options**
```bash
ffmpeg -i input.mp4 -c:v libx264 -preset veryslow -crf 18 -profile:v high -level 4.1 -pix_fmt yuv420p -bf 2 -g 250 -keyint_min 25 -sc_threshold 40 -qcomp 0.6 -aq-mode 2 -aq-strength 1.0 -c:a aac -b:a 192k output.mp4
```

✅ **GOOD: Utilizing preset files**
```bash
# In your custom_x264_preset.ffpreset:
# preset=veryslow
# crf=18
# profile=high
# level=4.1
# pix_fmt=yuv420p
# bf=2
# g=250
# keyint_min=25
# sc_threshold=40
# qcomp=0.6
# aq-mode=2
# aq-strength=1.0

# Then use:
ffmpeg -i input.mp4 -c:v libx264 -fpre custom_x264_preset -c:a aac -b:a 192k output.mp4
```
**Rationale:** Presets centralize complex option sets, making commands cleaner and configurations reusable.

### 5.2 AVOption Naming Conventions

Adhere to `AVOption` naming conventions (e.g., `-c:v`, `-b:a`) for clarity and consistency when specifying stream-specific options.

```bash
# Explicitly target video codec and audio bitrate
ffmpeg -i input.mp4 -c:v libx264 -b:a 128k output.mp4
```
**Rationale:** Clear stream specifiers prevent ambiguity and improve readability.
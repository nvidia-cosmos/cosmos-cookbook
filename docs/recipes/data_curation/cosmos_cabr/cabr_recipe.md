# Content-Adaptive Video Compression for Cosmos Curator with Beamr CABR

**Authors:** [Michael Ozeryansky](https://www.linkedin.com/in/ozeryansky/), [Oded Shoval](https://www.linkedin.com/in/oded-shoval-91443736/)
**Organization:** [Beamr](https://beamr.com)

## Overview

This recipe shows how to replace Cosmos Curator's default CPU-based, fixed-bitrate video encoder with Beamr CABR (Content-Adaptive Bitrate) — a GPU-accelerated encoding solution that analyzes content complexity to deliver the minimum bitrate needed at any target quality level. The result: significantly smaller files, faster GPU-accelerated encoding, and validated invariance across Curator's downstream stages.

| Tool | Workload | Use Case |
|------|----------|----------|
| Beamr CABR + Cosmos Curator | Data Curation | Content-adaptive video compression for curation pipelines |

## Prerequisites

- NVIDIA GPU with NVENC support (L4/L40S or higher recommended)
- CUDA 12.5+ and NVIDIA driver 550+
- Beamr CABR-enabled FFmpeg executable
- Cosmos Curator installed and running

**Getting Beamr CABR:** If you are an existing Beamr NVENC CABR SDK user, refer to the "Build FFmpeg with CABR" section of the SDK documentation to build your CABR-enabled FFmpeg. If you are a new user, contact [beamr.com](https://beamr.com) to request access to the SDK. Once built, place the FFmpeg executable on your pipeline machines and note the path.

## Why Content-Adaptive Encoding?

Cosmos Curator's default encoder (`libopenh264`) allocates a fixed bitrate to every clip — 4 Mbps for the video pipeline, 10 Mbps for AV. A static parking lot scene gets the same bits-per-second as dense highway traffic with rain. Across thousands of hours, this wastes terabytes of storage. Meanwhile, `libopenh264` runs on CPU while your NVIDIA GPU sits idle during encoding.

More critically, encoded clips feed directly into captioning models, aesthetic filters, motion analysis, embedding generators, and training datasets. Encoding artifacts become noise that every downstream model must see through. CABR wraps around NVENC and adds per-frame content analysis — simple scenes compress aggressively, complex scenes get the bits they need — preserving the signal your pipeline depends on while eliminating wasted storage.

## Validated Pipeline Integrity

CABR has been validated directly against the Cosmos Curator AV pipeline. Beamr tested CABR optimization on 9 uncompressed AV source videos spanning diverse conditions, processed through the full Cosmos Curator AV pipeline. Comparing NVENC encoding with and without CABR optimization, CABR achieved 41–57% bitrate savings with equivalent downstream accuracy across all measured dimensions: caption embeddings (T5-XXL and SBERT) fell within the model's stochastic noise floor, t-SNE clusters confirmed compression variants are indistinguishable from source, and VRI structured classification showed ~95% average agreement across 45 fields — matching standard NVENC encoding. For full methodology and per-video results, see the [Cosmos Curator AV Pipeline Compression Analysis](assets/Beamr-Optimized-compression-for-Cosmos-Curate-AV-pipeline-23-Feb-2026.pdf).

In separate testing against NVIDIA's published Physical AI AV dataset (600 videos), Beamr validated CABR against the RF-DETR object detection model, achieving mAP of 0.96 with minimal bounding box deviation and no systematic confidence degradation. These results span both the curation pipeline and the perception pipeline — covering the full range of downstream model types that consume Curator output. For details, see the ML-Safe AV Testing series: [Part 1](https://blog.beamr.com/2025/08/13/beamr-is-pushing-the-boundaries-of-av-data-efficiency-accelerated-by-nvidia/), [Part 2](https://blog.beamr.com/2025/12/18/ml-safe-av-video-data-processing-achieves-up-to-50-storage-reduction/), [Part 3](https://blog.beamr.com/2026/01/05/deep-dive-managing-the-petabyte-scale-av-video-data-bottlenecks/).

As with any pipeline modification, run a sanity check on a representative subset of your data (100–500 clips) before deploying at scale. Compare captioning outputs, embedding cosine similarity, and filter accept/reject decisions between standard and CABR encoding. Based on the validation above, expect no differences beyond the model's inherent run-to-run variability.

## Integration Points Overview

Cosmos Curator encodes video at three independent locations. You can integrate CABR at any combination — they are not sequential steps.

| Integration Point | Location | What It Does | When to Use |
|-------------------|----------|--------------|-------------|
| Primary Clip Encoding | `ClipTranscodingStage` in video + AV pipelines | Encodes raw video segments into clips | Always — this is where most data volume flows |
| Window Encoding | `split_video_into_windows()` in windowing utils | Re-encodes each clip window for captioning/filtering | When your dataset produces multiple windows per clip (savings multiply) |
| Pre-Storage Compression | `ClipWriterStage` and `CosmosPredict2WriterStage` | Writes encoded bytes to storage | When ingesting externally-encoded content or optimizing existing datasets |

## Integration Point 1: Primary Clip Encoding (Highest Impact)

These are the main clip transcoding stages where raw video segments are encoded into clips. This is where most of your data volume flows through.

### Video Pipeline

File: `cosmos_curate/pipelines/video/clipping/clip_extraction_stages.py`

Define the path to Beamr's FFmpeg near the top of the file:

```python
BEAMR_FFMPEG_PATH = "/path/to/ffmpeg_beamr"
```

In `_extract_clips()`, change the ffmpeg command to use the Beamr executable:

```python
command = [
    BEAMR_FFMPEG_PATH,  # Was: "ffmpeg"
    "-hide_banner",
    "-loglevel",
    "warning" if self._verbose else "error",
]
```

Replace the encoder-specific parameter block with CABR settings:

```python
if self._encoder == "h264_nvenc":
    command.extend([
        "-rc", "maxq",      # CABR content-adaptive quality optimization
        "-preset", "p7",     # Highest quality preset
        "-tune", "hq",       # High quality tuning
    ])
    if force_pix_fmt:
        command.extend(["-pix_fmt", "yuv420p"])
else:
    command.extend(["-b:v", f"{self._openh264_bitrate}M"])
```

Ensure hardware acceleration flags are present (these already exist in the NVENC path):

```python
if self._use_hwaccel:
    command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
```

**Key parameter — `-rc maxq`:** This is the CABR mode. Unlike `-rc:v vbr -cq:v 21` (which targets a fixed quality level), `maxq` tells CABR to find the maximum compression achievable while preserving quality. Simple scenes compress aggressively; complex scenes get the bits they need.

### AV Pipeline

File: `cosmos_curate/pipelines/av/clipping/clip_extraction_stages.py`

Apply the same modifications — the AV pipeline's `ClipTranscodingStage` is structurally identical. AV datasets tend to be especially large and diverse, making them ideal candidates for CABR.

### Activating the Change

In your pipeline configuration, set the encoder to NVENC (which CABR wraps around):

```python
ClipTranscodingStage(
    encoder="h264_nvenc",
)
```

### Codec Options

CABR supports multiple codecs. Choose based on your GPU and compatibility needs:

```python
"-vcodec", "h264_nvenc"   # H.264 — broadest compatibility
"-vcodec", "hevc_nvenc"   # H.265 — ~30% better compression
"-vcodec", "av1_nvenc"    # AV1 — best compression, requires L4/L40S+
```

For most Curator users, `h264_nvenc` is the safe default. If your downstream pipeline supports HEVC decoding, `hevc_nvenc` gives additional savings.

## Integration Point 2: Window Encoding (Multiplied Savings)

File: `cosmos_curate/pipelines/video/utils/windowing_utils.py`

The `split_video_into_windows()` function splits each clip into frame-based windows for captioning, filtering, and preview generation. Each window is re-encoded via ffmpeg. If your dataset has 10,000 clips averaging 3 windows each, that's 30,000 encoding operations — savings here multiply across the entire dataset.

Define the Beamr FFmpeg path at the top of the file (or import it from your clip extraction stage), then modify the ffmpeg command inside the windowing function where the encode command is built:

```python
cmd = [
    BEAMR_FFMPEG_PATH,
    "-y",
    "-hwaccel", "cuda",
    "-hwaccel_output_format", "cuda",
    "-i", input_file.name,
    "-vf", f"select='between(n\\,{window.start}\\,{window.end})',setpts=PTS-STARTPTS",
    "-c:v", "h264_nvenc",
    "-rc", "maxq",
    "-preset", "p7",
    "-tune", "hq",
    "-c:a", "copy",
    "-f", "mp4",
    tmp_file.name,
]
```

## Integration Point 3: Pre-Storage Compression (Last-Mile)

This integration point adds a CABR compression pass before clips are written to storage. It's most valuable when processing pre-encoded clips from external sources or optimizing existing datasets retroactively.

**When to skip:** If you've already integrated CABR at Integration Points 1 and 2, data reaching storage is already optimized. A second pass yields minimal additional savings.

### Helper Function

Create a shared compression utility:

```python
import subprocess
from pathlib import Path
from loguru import logger

BEAMR_FFMPEG_PATH = "/path/to/ffmpeg_beamr"

def compress_with_cabr(
    encoded_data: bytes,
    tmp_dir: Path,
    codec: str = "h264_nvenc",
) -> bytes:
    """Compress video bytes using Beamr CABR before storage."""
    input_file = tmp_dir / "cabr_input.mp4"
    output_file = tmp_dir / "cabr_output.mp4"
    input_file.write_bytes(encoded_data)

    cmd = [
        BEAMR_FFMPEG_PATH,
        "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", str(input_file),
        "-c:a", "copy",
        "-vcodec", codec,
        "-rc", "maxq",
        "-preset", "p7",
        "-tune", "hq",
        "-f", "mp4",
        str(output_file),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_file.read_bytes()
    except subprocess.CalledProcessError as e:
        logger.error(f"CABR compression failed: {e.returncode}")
        if e.stderr:
            logger.warning(f"FFmpeg error: {e.stderr.decode('utf-8')}")
        return encoded_data  # Fall back to original on failure
```

### Apply to AV ClipWriterStage

File: `cosmos_curate/pipelines/av/writers/clip_writer_stage.py`

Ensure `make_pipeline_temporary_dir` and `compress_with_cabr` are imported (or defined in this module). In `_upload_clips()`, add compression before `write_bytes()`:

```python
def _upload_clips(self, video: AvVideo, encoder: str) -> int:
    num_uploaded_clips = 0
    for clip in video.clips:
        if not clip.encoded_data:
            raise ValueError(...)

        # Compress with CABR before upload
        with make_pipeline_temporary_dir(sub_dir="cabr") as tmp_dir:
            clip.encoded_data = compress_with_cabr(clip.encoded_data, tmp_dir)

        dest = self._get_clip_url(clip.uuid, encoder)
        clip.url = str(dest)
        write_bytes(clip.encoded_data, dest, ...)
        num_uploaded_clips += 1
    return num_uploaded_clips
```

### Apply to CosmosPredict2WriterStage

File: `cosmos_curate/pipelines/av/writers/cosmos_predict2_writer_stage.py`

Ensure `make_pipeline_temporary_dir` and `compress_with_cabr` are available in scope. In `write_video_clip()`, compress before writing:

```python
def write_video_clip(clip, camera_view, s3_client_instance, dest_url, *, verbose=False):
    if clip.encoded_data is None:
        raise ValueError(...)

    # Compress with CABR before writing to training dataset
    with make_pipeline_temporary_dir(sub_dir="cabr") as tmp_dir:
        clip.encoded_data = compress_with_cabr(clip.encoded_data, tmp_dir)

    write_bytes(clip.encoded_data, dest_url, ...)
```

This is particularly impactful for Predict2 datasets — these are training data for Cosmos world foundation models. Smaller files with preserved quality mean faster data loading during training.

## What to Expect

Integrating CABR replaces Cosmos Curator's default encoding with a two-layer improvement: GPU-accelerated encoding via NVENC, and content-adaptive bitrate optimization via CABR. In testing against the Cosmos Curator AV pipeline, the CABR optimization layer alone achieved significant bitrate reductions (40–50%+ in tested scenarios) compared to standard NVENC encoding, with no measurable impact on downstream model accuracy. Total savings relative to the default `libopenh264` fixed-bitrate encoder may be even greater, as the baseline allocates a flat 4–10 Mbps regardless of content complexity. Actual savings will vary depending on your content mix — datasets with high scene diversity tend to benefit the most, as CABR has more room to optimize across varying complexity levels.

In practice, this means:

- **Storage savings at scale.** Lower bitrates translate directly to reduced storage footprint, I/O time, transfer costs, and downstream data loading — savings that compound across large curation runs.
- **GPU-accelerated encoding.** Moving from CPU-based `libopenh264` to NVENC+CABR shifts encoding to the GPU already present in your machine for inference, freeing CPU resources and improving pipeline throughput.
- **Content-adaptive bitrate allocation.** Instead of fixed 4–10 Mbps for every clip regardless of complexity, CABR allocates bits based on actual content needs — every byte is signal, not padding.
- **Multi-codec flexibility.** H.264, H.265, and AV1 support lets you choose the right compression-compatibility tradeoff for your pipeline.

## Troubleshooting

**GPU memory contention.** If CABR encoding runs alongside GPU inference (captioning, filtering), monitor GPU memory usage. Adjust `nb_streams_per_gpu` to balance encoding and inference workloads.

**10-bit color input.** The codebase already handles 10-bit input by forcing `yuv420p` when NVENC is selected. This works with CABR — the existing `force_pix_fmt` logic is preserved in the modified command.

**Verifying CABR is active.** Set `-loglevel info` in the ffmpeg command and look for CABR-specific analysis messages in the output. Standard NVENC output without CABR analysis lines means the CABR module isn't loaded — verify you're using the Beamr FFmpeg executable, not system FFmpeg.

**Graceful fallback.** The `compress_with_cabr()` helper includes a fallback: if CABR encoding fails for any reason, it returns the original bytes unchanged, ensuring pipeline stability during initial integration.

## Resources

1. [Beamr CABR](https://beamr.com/) — Product information and FFmpeg build requests
2. [Cosmos Curator AV Pipeline Compression Analysis](assets/Beamr-Optimized-compression-for-Cosmos-Curate-AV-pipeline-23-Feb-2026.pdf) — Full validation methodology and per-video results
3. ML-Safe AV Testing series: [Part 1](https://blog.beamr.com/2025/08/13/beamr-is-pushing-the-boundaries-of-av-data-efficiency-accelerated-by-nvidia/), [Part 2](https://blog.beamr.com/2025/12/18/ml-safe-av-video-data-processing-achieves-up-to-50-storage-reduction/), [Part 3](https://blog.beamr.com/2026/01/05/deep-dive-managing-the-petabyte-scale-av-video-data-bottlenecks/) — Object detection validation on NVIDIA's Physical AI dataset
4. [Cosmos Curator](https://github.com/NVIDIA/Cosmos-Curator) — Source repository

---

**Publication Date:** 2026

**Citation:**

Suggested text citation:

> Ozeryansky, M., & Shoval, O. (2026). Content-Adaptive Video Compression for Cosmos Curator with Beamr CABR. In NVIDIA Cosmos Cookbook. Beamr.

```bibtex
@misc{cosmos_cookbook_cabr_compression_2026,
    title={Content-Adaptive Video Compression for Cosmos Curator with Beamr CABR},
    author={Ozeryansky, Michael and Shoval, Oded},
    organization={Beamr},
    year={2026},
    howpublished={\url{https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/data_curation/cosmos_cabr/cabr_recipe.html}},
    note={NVIDIA Cosmos Cookbook}
}
```

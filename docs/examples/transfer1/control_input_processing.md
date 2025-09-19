# Processing Control Input Data for Training

This document provides detailed information about preparing control input data for training different Cosmos-Transfer1 models.

## DepthControl Training Data Format

**Requirements:**

- Depth videos in MP4 format
- Must be frame-wise aligned with corresponding RGB videos
- Must have same [H, W] dimensions as the input videos
- Place in `depth/` directory

## SegControl Training Data Format

The segmentation data is stored in pickle files, one per video. After loading a pickle file, the data structure is as follows:

```python
[
    {  # First detected object
        'phrase': str,  # Name/description of the detected object
        'segmentation_mask_rle': {
            'data': bytes,  # Run-length encoded binary mask data
            'mask_shape': tuple  # Shape of the mask (height, width)
        }
    },
    {  # Second detected object
        'phrase': str,
        'segmentation_mask_rle': {
            'data': bytes,
            'mask_shape': tuple
        }
    },
    # ... more detected objects
]
```

### Key Components

**Object Detection:**

- List of dictionaries, one per detected object
- Each detection contains:
    - `phrase`: String describing the object
    - `segmentation_mask_rle`: Dictionary containing:
        - `data`: RLE-encoded binary mask data
        - `mask_shape`: Tuple specifying mask dimensions (height, width)

**Mask Creation:**

- Reference implementation in `cosmos_transfer1/auxiliary/sam2/sam2_model.py`

## KeypointControl Training Data Format

For training KeypointControl models, you need to provide a pickle file containing 2D human keypoint annotations for each frame. The pickle file should follow this structure:

```python
{
    frame_id: [  # List of detected humans in this frame
        {  # Annotation for one human
            'human-bbox': np.array([x1, y1, x2, y2, confidence], dtype=np.float16),  # Normalized coordinates
            'human-bbox-abs': np.array([x1, y1, x2, y2, confidence], dtype=np.float16),  # Absolute coordinates
            'body-keypoints': np.array([[x, y, confidence], ...], dtype=np.float16),  # Shape: [133, 3], in the COCO-Wholebody format, normalized coordinates
            'body-keypoints-abs': np.array([[x, y, confidence], ...], dtype=np.float16),  # Shape: [133, 3], in the COCO-Wholebody format, absolute coordinates
            'hand-keypoints': np.array([[x, y, confidence], ...], dtype=np.float16),  # Shape: [42, 3], relative coordinates. It's a duplicate of the [91:133]-th keypoints of the 'body-keypoints'
            'face-bbox': np.array([x1, y1, width, height], dtype=np.float16),  # Normalized coordinates of the face bounding boxes of the humans detected
            'face-bbox-abs': np.array([x1, y1, width, height], dtype=np.int16)  # Absolute coordinates of the face bounding boxes of the humans detected
        },
        # ... more humans in this frame
    ],
    # ... more frames
}
```

### Key Components

**Frame ID:**

- Key in the dictionary
- Should match the corresponding video frame

**Per-Human Detection:**

- List of dictionaries, one per detected human
- Each detection contains:
    - Bounding boxes (normalized and absolute)
    - Body keypoints (133 points)
    - Hand keypoints (42 points)
    - Face bounding box

**Coordinate Systems:**

- **Normalized coordinates**: Values between 0 and 1
- **Absolute coordinates**: Pixel coordinates in the image
- All coordinates follow [x, y] format

**Confidence Scores:**

- Included for each keypoint and bounding box
- Values between 0 and 1
- Higher values indicate more reliable detections

### Data Preparation Tips

**Keypoint Detection:**

- We used [rtmlib](https://github.com/Tau-J/rtmlib) for human keypoint detection and output the COCO-Wholebody keypoint convention

**File Organization:**

- Name the pickle file to match the video name
- Place in the `keypoint/` directory
- Ensure frame IDs match video frames

## VisControl and EdgeControl

These are **self-supervised** control types:

- No separate data preparation needed
- Control inputs are generated on-the-fly during training
- Processing happens automatically from input videos during the training process

## Summary

| Control Type | Data Format | Directory | Alignment Requirement |
|--------------|-------------|-----------|----------------------|
| DepthControl | MP4 video | `depth/` | Frame-wise aligned with RGB |
| SegControl | Pickle file | `seg/` | Object detection per video |
| KeypointControl | Pickle file | `keypoint/` | Human keypoints per frame |
| VisControl | Self-supervised | N/A | Generated on-the-fly |
| EdgeControl | Self-supervised | N/A | Generated on-the-fly |

Ensure consistent file naming across all modalities - if your RGB video is named `example1.mp4`, the corresponding control data should be `depth/example1.mp4`, `seg/example1.pickle`, or `keypoint/example1.pickle` respectively.

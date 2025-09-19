# Data Curation

## Data Challenges

Initial evaluations of the pretrained Cosmos-Predict2 model revealed gaps in generating vehicle collision scenes:

- Unrealistic motion dynamics
- Oversized vehicles (likely due to dashcam bias in pretraining)
- Lack of incident-specific behavior
- Limited ability to maintain fixed-camera perspective

While the pretrained model excelled at routine traffic scenes, it struggled with collision scenarios when tested on ITS-specific prompts. This confirmed the need for targeted post-training with anomaly-rich data featuring accidents in-action from fixed CCTV perspectives.

## Data Sourcing and Filtering

To address these limitations, we developed a multi-source data pipeline that combines:

- ITS normal traffic scenes: 100 hours of traffic surveillance footage from different intersections at various times of the day, all captured from fixed CCTV viewpoints (no dashcam or moving camera perspectives)
- ITS accident scenes: Compilation of accident scenes from different intersections at various times of the day, all captured from fixed CCTV viewpoints (totaling approximately 3.5 hours of video)

**Disclaimer**: All data collected for this case study is for research proof of concept and demonstration purposes only. This data has not been merged into the pre-training dataset. This example serves solely to illustrate the data curation methodology and post-training workflow.

### Splitting and Captioning

**ITS accident scenes**: Original 5-10 minute compilations were split into individual clips using `cosmos-curate` with `transnetv2` scene detection and objective captioning.

```json
{
    "pipeline": "split",
    "args": {
        "input_video_path": "s3://your_bucket/raw_data/its_accident_scenes",
        "output_clip_path": "s3://your_bucket/processed_data/its_accident_scenes/v0",
        "generate_embeddings": true,
        "generate_previews": true,
        "generate_captions": true,
        "splitting_algorithm": "transnetv2",
        "captioning_algorithm": "qwen",
        "captioning_prompt_variant": "default",
        "captioning_prompt_text": "You are a video captioning expert trained to describe short CCTV footage of traffic collisions and abnormalities. Every input video contains either a visible traffic collision or a clear traffic abnormality such as a near miss, illegal turn, jaywalking, sudden braking, or swerving. Your task is to generate one concise and factual English paragraph that describes both the static environment and the dynamic physical event. For collision events, clearly describe how the collision unfolds — including the objects involved, their directions and relative speeds, the point of contact, and what happens immediately after. Begin every caption with: 'A traffic CCTV camera' Then describe: Environment: weather, Visible elements: vehicles, pedestrians, traffic lights, signs, road markings, Dynamic event: What vehicles or people are involved, How they move before the event, Where the impact occurs (e.g., front-left bumper hits right side of motorcycle), What happens afterward (e.g., rider falls, car swerves, vehicle spins, traffic halts). Use clear, physics-based verbs such as: collides, hits, swerves, brakes, accelerates, turns, merges, falls, flips, spins, crosses. Output Rules: Output must be one concise paragraph (1-3 small sentences), Focus on visible, physical actions - no speculation or emotional inference, Do not include: driver intentions, license plates, timestamps, brand names, street/building names, or text overlays, Assume all videos contain either a collision or an abnormal traffic event. Output Style Examples: A traffic CCTV camera shows a dry four-way intersection during the day. A red hatchback runs a red light and enters the intersection at moderate speed. From the right, a white SUV proceeds legally and collides into the hatchback's passenger-side door. The hatchback comes to rest near the opposite curb. A traffic CCTV camera captures a multi-lane road during daytime. Vehicles are moving slowly in moderate traffic. A black sedan abruptly slows down, and a silver pickup behind it fails to brake in time, crashing into the sedan's rear bumper. The front of the pickup crumples slightly while the sedan is pushed forward by a few meters. A traffic CCTV camera captures an intersection under clear skies. A motorcyclist enters the intersection diagonally from the left, crossing through oncoming traffic. A silver SUV traveling straight at moderate speed strikes the motorcycle's front wheel with its front-left bumper. The rider is thrown off and skids several feet across the road surface.",
        "limit": 0,
        "limit_clips": 0,
        "perf_profile": true
   }
}
```

**ITS normal traffic scenes**: 100 hours of continuous surveillance footage split into 10-second clips using `fixed-stride` algorithm. Captioning focused on general scene description since no incidents were detected.

```json
{
    "pipeline": "split",
    "args": {
        "input_video_path": "s3://your_bucket/raw_data/its_normal_traffic_scenes",
        "output_clip_path": "s3://your_bucket/processed_data/its_normal_traffic_scenes/v0",
        "generate_embeddings": true,
        "generate_previews": true,
        "generate_captions": true,
        "splitting_algorithm": "fixed-stride",
        "fixed_stride_split_duration": 10,
        "captioning_algorithm": "qwen",
        "captioning_prompt_variant": "default",
        "captioning_prompt": "You are a video captioning expert trained to describe short CCTV footage of traffic scenes. The input videos may contain normal traffic flow, traffic incidents, collisions, or other traffic-related events. Your task is to generate one concise and factual English paragraph that describes both the static environment and any dynamic events occurring in the video. If a collision or incident occurs, clearly describe how it unfolds — including the objects involved, their directions and relative speeds, the point of contact, and what happens immediately after. Begin every caption with 'A traffic CCTV camera' then describe: environment (weather conditions, time of day), visible elements (vehicles, pedestrians, traffic lights, signs, road markings), dynamic events (if any): what vehicles or people are involved, how they move and interact, where any impact occurs, what happens afterward. Use clear, physics-based verbs such as: travels, moves, stops, waits, collides, hits, swerves, brakes, accelerates, turns, merges, falls, flips, spins, crosses, proceeds, continues. Output must be one concise paragraph (1-3 sentences). Focus on visible, physical actions - no speculation or emotional inference. Do not include: driver intentions, license plates, timestamps, brand names, street/building names, or text overlays. Describe what actually happens in the video, whether normal or abnormal. Examples - Normal Traffic: A traffic CCTV camera shows a busy four-way intersection during daytime with clear weather. Multiple vehicles approach from different directions, stopping appropriately at red lights and proceeding when the signal turns green. Traffic flows smoothly with pedestrians crossing at designated crosswalks. Near-Miss Events: A traffic CCTV camera shows a busy intersection during evening hours. A motorcyclist makes a sharp left turn just as a white van approaches from the opposite direction at moderate speed. The van brakes hard and swerves slightly to avoid contact, while the motorcyclist completes the turn safely.",
        "limit": 0,
        "limit_clips": 0,
        "perf_profile": true
    }
}
```

Robust filtering techniques were applied:

- Removal of pillarboxes, borders, and overlays
- Manual review to ensure visual quality standards

### Dataset Composition

The final curated dataset composition is summarized below:

| Dataset | Quality | Incident Coverage | Artifacts | Clips |
|---------|---------|-------------------|-----------|-------|
| ITS normal traffic scenes (10 sec clips) | High | No | None | 44,000 |
| ITS accident scenes (5-15 sec clips) | Medium | Yes | None | 1,200 |

### Post-Training Dataset Sampling

For post-training, we selected 1,000 samples from each dataset (1:1 ratio):

- **Normal traffic scenes**: Diverse selection across intersections and times of day
- **Accident scenes**: 1,000 clips from available 1,200 to balance normal and anomaly learning

## Model Considerations for Post-Training

### Video Resolution Requirements

Supported resolutions for 720p video:

- **16:9**: 1280x720 (recommended for ITS footage)
- **1:1**: 960x960
- **4:3**: 960x720
- **3:4**: 720x960
- **9:16**: 720x1280

**Important**: Resize all videos to supported resolutions before training to avoid errors.

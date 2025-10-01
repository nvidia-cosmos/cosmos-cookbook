# Zetta Data Curation Tool

Zetta is a command-line tool designed for video dataset curation and processing, enabling efficient management and transformation of video data.
It consists of two parts:

1. an NVCF function (based on the Nemo Curator container for video), and
2. the Zetta CLI that can be used to interact with the NVCF function.

## Table of Contents

- [Access Requirements](#access-requirements)
- [Building the Nemo-Curator Container](#building-the-nemo-curator-container)
- [Building the Zetta CLI](#building-the-zetta-cli)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example Pipeline](#example-metropolis-its-data-pipeline)
- [Key Features and Limitations](#key-features-and-limitations)
- [Additional Information](#additional-information)
- [Important Notes](#important-notes)

## Access Requirements

Before using Zetta, you need to:

1. Request access to [nemo-curator](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Curator) and [zetta](https://gitlab-master.nvidia.com/ov-infra-vfm-data-workstream/zetta) repos
2. Request access to the `video_curator_lha_trial` organization by posting a request in the Slack channel #cosmos-lha-data-curation
3. Once access is granted, generate your personal API key at `https://org.ngc.nvidia.com/setup`
4. For detailed Zetta usage instructions, refer to: `https://drive.google.com/file/d/1CpIRYAz_NGblZTj1FB510zOAbzr9fGCB/view?usp=drive_link`
5. While Zetta can be used with any S3 bucket, for LHA engagements, we use the `s3://lha-datasets/` bucket. The credentials are stored in the vault `aws-058264432168-cosmos-dev`. If you cannot find the credentials, please contact a member of the Cosmos Engineering team to be added.

## Building the Nemo-Curator Container

The Zetta NVCF function uses the [nemo-curator](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Curator) repository's `video` branch.

To set up the environment and build the Zetta NVCF function's docker container :

```bash
# check out the pipeline code
git clone -b video ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/NeMo-Curator.git
cd NeMo-Curator
python -m venv .venv
source .venv/bin/activate
pip install cython
pip install -e '.[video]'
```

Once the video curator is installed in your local path, execute the following command to build the nvcf function

```bash
video_curator image build \
    --image-name nvcr.io/ebplj5vljyh4/nemo_video_curator \
    --image-tag latest \
    --envs model_download,unified,video_splitting,vllm,text_curator
```

Once the container is built, push it to the required team/org.

### Guide for Development of New Features

The following are the key files and folders of importance if you are developing new features for Zetta.

```text
NeMo-Curator
├── Dockerfile
├── ...
├── docs
│   ├── README.md
│   └── user-guide
│       ├── ...
│       └── video
├── nemo_curator
│   ├── ...
│   └── video
│       ├── config
│       ├── data_utils
│       ├── environments
│       ├── __init__.py
│       ├── internal
│       ├── models --> Contains the core filtering and captioning models, like QWEN
│       ├── monitoring
│       ├── nvcf_module
│       ├── pipelines --> Contains the Ray Pipelines and Stages for captioning, clipping, embedding etc.
│       ├── ray_utils
│       ├── README.md
│       ├── run --> Contains the CLI to build the zetta function's docker container
│       ├── scripts
│       │   ├── invoke-run-av.sh
│       │   ├── invoke-run-video.sh
│       │   ├── invoke-status.sh
│       │   └── launch-local-nvcf.sh --> This shows how to use the dockerfile to create a local NVCF function equivalent
│       ├── tests
│       └── utils
├── pyproject.toml --
├── README.md
├── ...
├── tutorials
│   ├── ...
│   ├── cosmos-vfm-curation -->  Demonstrates how to curate a small dataset for fine-tuning Cosmos foundation models.
│   │   ├── cosmos-curation.ipynb
│   │   ├── nemo_curator_local_workspace
│   │   └── README.md
│   ├── video-curation
│   │   └── README.md
└── video.Dockerfile --> Used by the video-curator CLI to build the image. Do NOT build from this directly
```

### Troubleshooting and Tips

1. Shortcut to container build :

    Container building takes a while. If you are not making any changes, it's faster to grab the latest `prod_video_curator` that is tested by Nemo-Curator from the `Video_Curator_LHA_trial` org

    ```bash
    docker tag nvcr.io/0624307083068081/prod_video_curator:2025-04-30_e355e142 nvcr.io/ebplj5vljyh4/dev/prod_video_curator:2025-04-30_e355e142
    docker push nvcr.io/ebplj5vljyh4/dev/prod_video_curator:2025-04-30_e355e142
    ```

2. Troubleshooting function builds :

    If you encounter issues with function builds or this doccument becomes obsolete by new changes, refer to the [Nemo Curator's gitlab-ci](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Curator/-/blob/video/.gitlab-ci.yml?ref_type=heads#L178) file as source of truth since it automates the steps above.
    It also triggers the Zetta CLI build and validation CI pipeline that does the [Zetta CLI build](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Curator/-/blob/video/.gitlab-ci.yml?ref_type=heads#L242) which we discuss next.

## Building the Zetta CLI

The Zetta CLI can be installed from [zetta](https://gitlab-master.nvidia.com/ov-infra-vfm-data-workstream/zetta) Repository as follows

```bash
# check out the pipeline code
git clone ssh://git@gitlab-master.nvidia.com:12051/ov-infra-vfm-data-workstream/zetta.git
cd zetta
python -m venv .venv
source .venv/bin/activate
./install.sh -e .
zetta
```

If your installation succeded, you will see the zetta CLI's info as follows

```bash
zetta: Version: 0.5.18

 Usage: zetta [OPTIONS] COMMAND [ARGS]...

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ nvcf    NVCF Functionalities                                                                                                                │
│ local   Local Functionalities                                                                                                               │
│ slurm   SLURM Functionalities                                                                                                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Configuration

1. To toggle between different Zetta enabled orgs, it is best to set up aliases as follows :

    ```bash
    alias zetta-dev='unset NGC_NVCF_ORG && unset NGC_NVCF_API_KEY && export NGC_NVCF_ORG=ebplj5vljyh4 && export NGC_NVCF_API_KEY=$EDIFY_DEV_KEY && zetta nvcf config set'
    alias zetta-lha='unset NGC_NVCF_ORG && unset NGC_NVCF_API_KEY && export NGC_NVCF_ORG=0624307083068081 && export NGC_NVCF_API_KEY=$LHA_TRIAL_KEY && zetta nvcf config set'
    ```

    - Note that you need to specify the NGC Personal keys generated for your dev org (in this case `$EDIFY_DEV_KEY` ) and the Video_Curator_LHA_Trial org (i.e `$LHA_TRIAL_KEY`) for the commands to work.

## Usage

1. To get the current zetta NVCF function in your org, do  `zetta nvcf function list-functions`

    - Note that you can ignore the `ai-*` functions which are NIM functions.
    - For cosmos, under the video curator function you are looking for is : `internal-cosmos-eng`. Reccord it's function ID and version id for the next step.

2. To get details of the current function, do `zetta nvcf function list-function-detail --name internal-cosmos-eng`

    - Note that this image is the container image, that is used in the [Troubleshooting and Tips](#troubleshooting-and-tips) section
    - One limitation with the list-function-detail feature of zetta is that it does not provide information of which NGC models are currently used by the function. Use the NGC UI  to get the full information

3. To download models from one org and upload to another, while maintaining the metadata set by Zetta, the command is as follows.

    ```bash
    zetta-lha # Sets org to the LHA org
    zetta nvcf model list-models | grep t5_efficient_xxl_11b_encoder
    zetta nvcf model download-model --mname t5_efficient_xxl_11b_encoder --save-path .
    zetta-dev # Sets org to your Dev org
    zetta nvcf model upload-model --src-path t5_efficient_xxl_11b_encoder_v0.2.0 --data-file $PLACE_YOU_CLONED_ZETTA_TO/launcher_nvcf/example_configs/model_upload_t5_xxl.json
    ```

    - This assumes you have set up `zetta_dev` and `zetta-lha` aliases as mentioned in the [configuration](#configuration) section

4. Creating the nemo-curator function via Zetta

    Once all the models for the Nemo Curator NVCF functions are uploaded, you can create the function using the Zetta CLI in your org as follows.

    ```bash
    zetta-dev # To make sure you are in the right org
    zetta nvcf function create-function --name ${USER}-zetta-test-function --image <your_pushed_docker_image> --data-file ./creation_template.json
    ```

    - Note that the creation template is checked in to the same folder as this README.

5. Deploying the nemo-curator function via Zetta

    To deploy the zetta function you created in step 4, use the NVCF UI or the [Picasso Admin](http://picasso-admin.nvidia.com:8080/#/) tool.
    - [ WIP ] Alternatively you can try to deploy via Zetta CLI as follows. Note that the deployment template is checked in to the same folder as this README.

    ```bash
    export NGC_NVCF_ORG=ebplj5vljyh4
    export NGC_NVCF_API_KEY=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
    export NVCF_BACKEND=nvcf-dgxc-k8s-aws-use1-prd1
    export NVCF_GPU_TYPE=H100
    export NVCF_INSTANCE_TYPE=AWS.GPU.H100_4x
    export NVCF_TRACE_FILE=./temp/logs
    zetta nvcf function deploy-function --data-file ./deployment_template.json
    ```

6. Importing the active nemo-curator function to use with Zetta.

    If your function is active after deployment ( Check using `zetta nvcf function get-deployment-detail --id <FUNCTION_ID> --version <VERSION_ID>` ), you can register it as the active function.

    ```bash
    zetta-dev
    zetta nvcf function import-function \
    --id <FUNCTION_ID> \
    --version <VERSION_ID> \
    --name ${USER}-zetta-test-function
    ```

    - Alternatively you can import the function ID from the `Video_Curator_LHA_Trial` org. You can get the latest function ID from colleagues who have recently used the Zetta tool.
    - To use the LHA function, run `zetta-lha && zetta nvcf function import-function --id 38fe8e6d-f38a-4f6f-88dd-230ed3825dd1 --version 15222951-f63a-4b88-ad7f-bae48eecab54 --name internal-cosmos-eng-0422`

7. Invoking the nemo-curator

    - If you want to test a small e2e invocation, you can upload a single video file to NVCF and invoke the pipeline using

        ```bash
        zetta nvcf asset upload-asset --src-path <filename> --description "test video"
        zetta nvcf function invoke-function --data-file ./split_template.json --assetid <asset_id_from_previous_step>
        ```

    - Then you can invoke the function to do data curation on your S3 Bucket

        ```bash
        zetta nvcf function invoke-function --data-file <path_to_your_json_file> --s3-config-file <path_to_file_containing_a_single_aws_cred>
        ```

    - The data file here specifies the pipeline and arguments to run your necessary curation. Some examples json data-files are shown in [Example: Metropolis ITS Data Pipeline](#example-metropolis-its-data-pipeline)
    - The AWS cred file has to be exactly as follows :

        ```toml
        [default]
        aws_access_key_id=<key id>
        aws_secret_access_key=<key>
        region=<region>
        ```

8. Vizualizing the curation output

    To verify the output, use the following command to sample and visualize the output data:

    ```bash
    zetta nvcf view clips --zip-file /tmp/<filename_of_your_pipeline_output>.zip
    ```

## Example: Metropolis ITS Data Pipeline

The following section describes the process of curating the `DBQ_20_Hr_Export` dataset, provided by our Metropolis ITS team.

### 1. Raw Data

- 130 hours of surveillance camera video recordings
- Each video segment is stored as 1-hour long recordings
- Original format: .mkv (converted to .mp4 for Zetta compatibility)
- Resolution: 1080p
- Frame rate: 30fps
- Files are organized in nested directories by:
  - Road intersection
  - Time of day (AM vs PM)

### 2. Video Splitting and Captioning

Use the following JSON configuration for video splitting and captioning:

```json
{
    "pipeline": "split",
    "args": {
        "input_video_path": "s3://lha-datasets/metropolis/ITS/DBQ_20_Hr_Export/raw_data",
        "output_clip_path": "s3://lha-datasets/metropolis/ITS/DBQ_20_Hr_Export/processed_data/v0_prompt1",
        "generate_embeddings": true,
        "generate_previews": true,
        "generate_captions": true,
        "splitting_algorithm": "transnetv2",
        "captioning_algorithm": "qwen",
        "captioning_prompt_variant": "default",
        "captioning_prompt_text": "You are a video caption expert. Please describe this video in two paragraphs: 1. Describe the static scene: What objects and people are present? What are their spatial relationships and the overall environment? 2. Describe the actions and motion: How do objects move and interact? Focus on object permanence, collisions, and physical interactions. Be specific and precise in your observations.",
        "limit": 0,
        "limit_clips": 0,
        "perf_profile": true
   }
}
```

### 3. Recaptioning

When caption quality needs improvement, you can iterate the prompt without recomputing embeddings or previews. Use the `fixed-stride` splitting algorithm with a large duration value to preserve the original clips:

```json
{
    "pipeline": "split",
    "args": {
        "input_video_path": "s3://lha-datasets/metropolis/ITS/DBQ_20_Hr_Export/processed_data/v0_prompt1",
        "output_clip_path": "s3://lha-datasets/metropolis/ITS/DBQ_20_Hr_Export/processed_data/v0_prompt2",
        "generate_embeddings": false,
        "generate_previews": false,
        "generate_captions": true,
        "splitting_algorithm": "fixed-stride",
        "fixed_stride_split_duration": 100000000,
        "captioning_algorithm": "qwen",
        "captioning_prompt_variant": "default",
        "captioning_prompt_text": "<your modified prompt here>",
        "limit": 0,
        "limit_clips": 0,
        "perf_profile": true
   }
}
```

### 4. Data Sharding

```json
{
    "pipeline": "shard",
    "args": {
        "input_clip_path": "s3://lha-datasets/metropolis/ITS/DBQ_20_Hr_Export/processed_data/v0_prompt2",
        "output_dataset_path": "s3://lha-datasets/metropolis/ITS/DBQ_20_Hr_Export/webdataset/v0_prompt2",
        "annotation_version": "v0",
        "verbose": true,
        "perf_profile": false
    },
    "pipeline_config": {
        "stages": [
            {
                "class": "nemo_curator.video.pipelines.video.captioning.captioning_stages.T5Stage",
                "params": {
                    "caption_fields": [
                        "qwen_caption"
                    ]
                }
            },
            {
                "class": "nemo_curator.video.pipelines.video.io.download_stages.DownloadPackUpload",
                "stage_spec": {
                    "num_workers_per_node": 4
                }
            }
        ]
    }
}
```

### 5. Verification

Visualize the data as mentioned in the [usage](#usage) section to verify your curation pipeline outputs

## Key Features and Limitations

### Advantages

- **Easy to Use**: Zetta is deployed as an NVCF function, allowing users to set up data processing pipelines through configuration files without writing custom code
- **Streamlined Workflow**: The tool provides a unified interface for video splitting, captioning, and dataset creation
- **Configurable Processing**: All processing steps can be customized through JSON configuration files
- **Scalable Processing**: Supports batch processing of large video datasets
- **Quality Control**: Built-in support for basic video quality checks

### Current Limitations

- **Limited Filtering Options**: The current version does not support arbitrary or customized filtering algorithms
- **Basic Quality Control**: For noisy datasets requiring filtering based on:
  - Resolution
  - Frame length
  - Motion detection
  - Watermark removal
  - Banner detection
  These features are currently under development (WIP)

### Workaround for Custom Filtering

When custom filtering is required, follow this recommended workflow:

1. First, process videos through Zetta's standard pipeline
2. Apply standalone filters to remove clips that don't meet quality criteria
3. Manually remove corresponding metadata and embeddings for consistency
4. Finally, apply sharding to create the final webdataset

Note: This workaround requires additional manual effort but ensures data consistency.

## Additional Information

- For detailed documentation, refer to the official documentation
- For technical support, share the question in the #cosmos-lha-data-curation channel
- Refer to curator documentation for additional info: <https://developer.nvidia.com/docs/nemo-curator-video-processing/index.html>

## Important Notes

1. Follow data privacy and security guidelines
2. Verify output quality after each processing stage
3. Keep your API keys and credentials secure
4. Monitor processing logs for any errors or warnings
5. Plan for additional processing time if custom filtering is required
6. Consider data consistency when applying custom filters
7. Document any manual filtering steps for reproducibility
8. Adhere to the directory structure of the LHA-Datasets bucket specified below

### Directory Structure of LHA-Datasets bucket

When processing datasets using our LHA bucket, please follow this directory structure:

```bash
project_root/
└── dataset_root/
    ├── raw_data/          # Raw data
    ├── processed_data/    # Processed data
        ├── v0/            # Version 0
            ├── clips/     # Processed clips
            ├── meta/      # Metadata
            ├── embeds/    # Calculated embeddings
            └── ...
        └── v1/            # Version 1
            └── ...
    └── webdataset/        # Curated and sharded webdataset
```

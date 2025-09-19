# nuScenes

## Compile

Download:

```shell
./cosmos-playbook/experimental/joallen/nuscenes/download_nuscenes.py --username <username> --password <password> --output_dir ~/datasets/nuscenes/data
```

Compile:

```shell
./cosmos-playbook/experimental/joallen/nuscenes/compile_nuscenes.py ~/datasets/nuscenes/data ~/datasets/nuscenes/cam_front/test --version 'v1.0-test' --cameras 'CAM_FRONT' --workers 16
./cosmos-playbook/experimental/joallen/nuscenes/compile_nuscenes.py ~/datasets/nuscenes/data ~/datasets/nuscenes/cam_front/train --version 'v1.0-trainval' --cameras 'CAM_FRONT' --workers 16

s5cmd cp --show-pogress ~/datasets/nuscenes/cam_front/ s3://lha-datasets/nuscenes/cam_front/
```

## Pre-Process

```shell
s5cmd cp --show-progress "s3://lha-datasets/nuscenes/cam_front/test/*" ~/datasets/nuscenes/v0/raw/

cd cosmos-playbook
source .venv/bin/activate
python scripts/curation/tools/data_standardization/file_renaming.py ~/datasets/nuscenes/v0/raw

s5cmd cp --show-progress ~/datasets/nuscenes/v0/raw/ "s3://lha-datasets/debug/joallen/nuscenes/v0/raw/"
```

## Process

```shell

# Process
source cosmos-curate/.venv/bin/activate
# HACK
# cosmos-curate nvcf function import-function \
#     --funcid 58891eb4-0eef-4a91-8177-5c272275a375 \
#     --version b109565a-e15e-492c-997d-3c35ed2c7a9e \
#     --name Cosmos-Curate-LHA-1_0_2
mkdir "~/.config/cosmos_curate"
echo '{"id":"58891eb4-0eef-4a91-8177-5c272275a375","version":"b109565a-e15e-492c-997d-3c35ed2c7a9e","name":"Cosmos-Curate-LHA-1_0_2"}' > "~/.config/cosmos_curate/funcid.json"

cosmos-curate nvcf function invoke-function --data-file v0/split_0.json --s3-config-file ~/.aws/credentials
cosmos-curate nvcf function invoke-function --data-file v0/shard.json --s3-config-file ~/.aws/credentials

s5cmd cp --show-progress "s3://lha-datasets/debug/joallen/nuscenes/v0/shard/v0/*" ~/datasets/nuscenes/v0/shard/
```

"""Utilties for reading the DIR config file.

See README.md for more info on what the DIR config file is.
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import attrs
import cattrs
import yaml

CONFIG_FILE_LOCATION = pathlib.Path("~/.config/dir/config.yaml")


# When running on Lepton, we need to use a different config file location.
# This is because we have to hack lepton secrets into the config file.
# Hopefully this will be fixed in the future.
def is_lepton_job() -> bool:
    return os.environ.get("LEPTON_JOB_WORKER_INDEX") is not None


LEPTON_CONFIG_FILE_LOCATION = pathlib.Path("~/.config/dir/lepton_config.yaml")
CONFIG_FILE_LOCATION = pathlib.Path("~/.config/dir/config.yaml") if not is_lepton_job() else LEPTON_CONFIG_FILE_LOCATION


@attrs.define
class Parseable:
    user: str = attrs.field(repr=False)
    password: str = attrs.field(repr=False)
    server_url: str = "http://35.84.100.32:8000"


@attrs.define
class S3Profile:
    """A class to represent and interact with S3 AWS configuration details."""

    user: str = attrs.field(repr=False)
    key: str = attrs.field(repr=False)
    endpoint: str = "https://pbss.s8k.io"
    region: str = "us-east-1"


@attrs.define
class Aws:
    """A class to represent and interact with AWS configuration details."""

    s3_profiles: dict[str, S3Profile] = attrs.Factory(dict)


@attrs.define
class OpenAI:
    """A class to represent and interact with OpenAI configuration details."""

    user: str = attrs.field(repr=False)
    api_key: str = attrs.field(repr=False)


@attrs.define
class Huggingface:
    """A class to represent and interact with Huggingface configuration details."""

    user: str = attrs.field(repr=False)
    api_key: str = attrs.field(repr=False)


@attrs.define
class HiveProfile:
    """A class to represent and interact with Hive database configuration details."""

    user: str
    password: str = attrs.field(repr=False)
    port: int = 10000
    endpoint: str = "vfm-hive.cosmos.nvidia.com"


@attrs.define
class KratosUser:
    """A class to represent and interact with Kratos database configuration details."""

    secret: str = attrs.field(repr=False)
    id: str = "nvssa-prd--WlHx7ApnmvNjTkexVZa-Cvvmr8nrTD2dhA2iKWgUpM"


@attrs.define
class PostgresUser:
    """A class to represent and interact with Postgres database configuration details."""

    user: str
    password: str = attrs.field(repr=False)
    endpoint: str = "videoprd20240530230024961300000001.cfj3pyamiol7.us-west-2.rds.amazonaws.com"


@attrs.define
class DatabricksUser:
    """A class to represent and interact with Databricks database configuration details."""

    access_token: str = attrs.field(repr=False)
    hostname: str = "nvidia-kratos-ca1-jdbc.cloud.databricks.com"
    http_path: str = "sql/protocolv1/o/4250550431795074/dir-explorer"


@attrs.define
class VisualSearch:
    """A class to represent and interact with visual search config details."""

    key: str = attrs.field(repr=False)
    user: str = attrs.field(default="$oauthtoken", repr=False)


@attrs.define
class NVCFConfig:
    """A class to represent and interact with NVCF configuration details."""

    ssa_client_id: str = attrs.field(repr=False)
    ssa_client_secret: str = attrs.field(repr=False)
    nvcf_api: str = attrs.field(repr=False)
    jwt_token_provider: str = attrs.field(repr=False)


@attrs.define
class Postgres:
    """A class to represent and interact with Postgres configuration details."""

    profiles: dict[str, PostgresUser] = attrs.Factory(dict)


@attrs.define
class Kratos:
    """A class to represent and interact with Kratos configuration details."""

    profiles: dict[str, KratosUser] = attrs.Factory(dict)


@attrs.define
class Databricks:
    """A class to represent and interact with Kratos configuration details."""

    profiles: dict[str, DatabricksUser] = attrs.Factory(dict)


@attrs.define
class Hive:
    """A class to represent and interact with Postgres configuration details."""

    profiles: dict[str, HiveProfile] = attrs.Factory(dict)


@attrs.define
class WAndDB:
    """A class to represent and interact with WAndB configuration details."""

    api_key: str = attrs.field(repr=False)


@attrs.define
class Gemini:
    """A class to represent and interact with Gemini configuration details."""

    api_key: str = attrs.field(repr=False)


@attrs.define
class Anthropic:
    """A class to represent and interact with Anthropic configuration details."""

    api_key: str = attrs.field(repr=False)


@attrs.define
class Ngc:
    """A class to represent and interact with NGC configuration details."""

    api_key: str = attrs.field(repr=False)


@attrs.define
class Lepton:
    """A class to represent and interact with Lepton configuration details."""

    workspace_token: str = attrs.field(repr=False)
    workspace_id: str = "b5k2m9x7"


@attrs.define
class ConfigFileData:
    """A class to handle the configuration data for Yotta.

    This class supports loading from and saving to a configuration file.

    The config file is stored at ~/.config/dir/config.yaml. An example config file is:

    ``` yaml
    user: jhuffman
    aws:
        s3_profiles:
            team-dir:
                user: "team-dir"
                key: "xx"
            deep-imagination-testing:
                user: "deep-imagination-testing"
                key: "xx"
            team-dir-pdx:
                user: "team-dir"
                key: "xx"
                endpoint: "https://pdx.s8k.io"
    open_ai:
        user: "jhuffman_captioning"
        api_key: "xx"
    visual_search:
        key: "xx"
    huggingface:
        user: "jhuffman"
        api_key: "xx"
    parseable:
        user: "xx"
        password: "xx"
    postgres:
        profiles:
           team-vfm:
               user: team_vfm
               password: xx
    kratos:
        profiles:
           team-vfm:
               secret: "xx"
    databricks:
        profiles:
            team-vfm:
                access_token: "xx"
    gemini:
        api_key: "xx"
    anthropic:
        api_key: "xx"
    ngc:
        api_key: "xx"
    ```

    Attributes:
        open_ai (Optional[OpenAI]): Configuration data for OpenAI.
        huggingface (Optional[Huggingface]): Configuration data for Huggingface.
    """

    # An optional user attribute. If this is present, Yotta will use it to get the current user. Otherwise, it uses
    # getpass.getuser
    user: Optional[str] = None
    aws: Optional[Aws] = None
    open_ai: Optional[OpenAI] = None
    huggingface: Optional[Huggingface] = None
    # Optional information about which parseable server to connect to. If not present, logs will not be pushed to
    # parseable.
    parseable: Optional[Parseable] = None
    postgres: Optional[Postgres] = None
    kratos: Optional[Kratos] = None
    databricks: Optional[Databricks] = None
    hive: Optional[Hive] = None
    nvcf: Optional[NVCFConfig] = None
    wandb: Optional[WAndDB] = None
    gemini: Optional[Gemini] = None
    anthropic: Optional[Anthropic] = None
    ngc: Optional[Ngc] = None
    visual_search: Optional[VisualSearch] = None
    lepton: Optional[Lepton] = None

    @classmethod
    def from_dict(cls, data: dict) -> ConfigFileData:
        return cattrs.structure(data, ConfigFileData)

    @classmethod
    def from_file(cls) -> Optional[ConfigFileData]:
        file = None
        if CONFIG_FILE_LOCATION.expanduser().exists():
            file = CONFIG_FILE_LOCATION.expanduser()
        # elif environment.CONTAINER_PATHS_CONFIG.exists():
        #     file = environment.CONTAINER_PATHS_CONFIG
        if file is None:
            return None

        with open(file, "r") as f:
            if is_lepton_job():
                # In Lepton environment, replace environment variables
                content = f.read()
                content = os.path.expandvars(content)
                dict_ = yaml.safe_load(content)
            else:
                # In normal environment, load directly
                dict_ = yaml.safe_load(f)
        return cls.from_dict(dict_)

    def to_dict(self) -> dict:
        return cattrs.unstructure(self, ConfigFileData)

    def to_file(self, dest: Optional[pathlib.Path] = None) -> None:
        if dest is None:
            dest = CONFIG_FILE_LOCATION.expanduser()
        dict_ = self.to_dict()
        with open(dest, "w") as f:
            yaml.dump(dict_, f, default_flow_style=False, allow_unicode=True)

    def get_s3_profile(self, profile_name: str) -> S3Profile:
        if self.aws is None:
            raise ValueError("Config does not have any AWS data.")
        if profile_name not in self.aws.s3_profiles:
            raise ValueError(
                f"{profile_name=} not found in AWS config. Available profiles: {sorted(self.aws.s3_profiles.keys())}. "
                "See '/README.md#Add the yotta config file' for more info.",
            )
        return self.aws.s3_profiles[profile_name]

    def get_postgres_profile(self, profile_name: str) -> PostgresUser:
        if self.postgres is None:
            raise ValueError("Config does not have any Postgres data.")
        if profile_name not in self.postgres.profiles:
            raise ValueError(
                f"{profile_name=} not found in Postgres config. "
                f"Available profiles: {sorted(self.postgres.profiles.keys())}. "
                "See '/README.md#Add the yotta config file' for more info.",
            )
        return self.postgres.profiles[profile_name]

    def get_kratos_profile(self, profile_name: str) -> KratosUser:
        if self.kratos is None:
            raise ValueError("Config does not have any Kratos data.")
        if profile_name not in self.kratos.profiles:
            raise ValueError(
                f"{profile_name=} not found in Kratos config. "
                f"Available profiles: {sorted(self.kratos.profiles.keys())}. "
                "See '/README.md#Add the yotta config file' for more info.",
            )
        return self.kratos.profiles[profile_name]

    def get_databricks_profile(self, profile_name: str) -> DatabricksUser:
        if self.databricks is None:
            raise ValueError("Config does not have any Databricks data.")
        if profile_name not in self.databricks.profiles:
            raise ValueError(
                f"{profile_name=} not found in Databricks config. "
                f"Available profiles: {sorted(self.databricks.profiles.keys())}. "
                "See '/README.md#Add the yotta config file' for more info.",
            )
        return self.databricks.profiles[profile_name]

    def get_hive_profile(self, profile_name: str) -> HiveProfile:
        if self.hive is None:
            raise ValueError("Config does not have any Hive data.")
        if profile_name not in self.hive.profiles:
            raise ValueError(
                f"{profile_name=} not found in Hive config. "
                f"Available profiles: {sorted(self.hive.profiles.keys())}. "
                "See '/README.md#Add the yotta config file' for more info.",
            )
        return self.hive.profiles[profile_name]

    def get_parseable_info(self) -> Parseable:
        if self.parseable is None:
            raise ValueError(
                "Parseable info not found in DIR config. See '/README.md#Add the yotta config file' for more info.",
            )
        return self.parseable

    def get_wandb_api_key(self) -> str:
        if self.wandb is None:
            raise ValueError(
                "wandb info not found in DIR config. See '/README.md#Add the yotta config file' for more info.",
            )
        return self.wandb.api_key

    def get_gemini_api_key(self) -> str:
        if self.gemini is None:
            raise ValueError(
                "gemini api key not found in DIR config. See '/README.md#Add the yotta config file' for more info.",
            )
        return self.gemini.api_key

    def get_anthropic_api_key(self) -> str:
        if self.anthropic is None:
            raise ValueError(
                "anthropic api key not found in DIR config. See '/README.md#Add the yotta config file' for more info.",
            )
        return self.anthropic.api_key

    def get_ngc_api_key(self) -> str:
        if self.ngc is None:
            raise ValueError(
                "ngc user api key not found in DIR config. See '/README.md#Add the yotta config file' for more info.",
            )
        return self.ngc.api_key

    def get_visual_search_auth(self) -> tuple[str, str]:
        if self.visual_search is None:
            raise ValueError(
                "visual_search not found DIR config. See '/README.md#Add the yotta config file' for more info.",
            )
        return (self.visual_search.user, self.visual_search.key)


def maybe_load_config() -> Optional[ConfigFileData]:
    return ConfigFileData.from_file()


def load_config() -> ConfigFileData:
    config = maybe_load_config()
    if config is None:
        raise RuntimeError("DIR config file not found. See '/README.md#Add the yotta config file' for more info.")
    return config


def get_s3_profile_from_config(profile_name: str) -> S3Profile:
    config = load_config()
    return config.get_s3_profile(profile_name)

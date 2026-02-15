from typing import Dict

from dataclasses import dataclass
from pathlib import Path

from yaml import safe_load


@dataclass
class ProjectConfig:

    """
    Container object for project configuration paths.

    This dataclass represents configuration values loaded from the
    project's YAML configuration file. It centralizes filesystem paths
    used across the application.

    Attributes:
        project_home_path (Path):
            Root directory of the project workspace defined in
            `config.yaml` as `PROJECT_HOME_PATH`.

        project_config_path (Path):
            Path to the configuration YAML file used to initialize
            the project configuration.

        gcp_sa_path (Path):
            Path to the Google Cloud service account credential file,
            typically located at:
            `<PROJECT_HOME_PATH>/credentials/gcp/service_account.json`.
    """

    project_home_path: Path
    project_config_path: Path
    gcp_sa_path: Path

def load_project_config(
        project_root_path: Path
) -> ProjectConfig:

    """
    Load project configuration from a YAML file.

    This function reads `config.yaml` from the provided project root
    directory, extracts required configuration values, and constructs
    a `ProjectConfig` object.

    Expected YAML structure:
        PROJECT_HOME_PATH: /absolute/or/relative/path

    Args:
        project_root_path (Path):
            Root directory of the project containing `config.yaml`.

    Returns:
        ProjectConfig:
            Structured configuration object containing project paths.

    Raises:
        FileNotFoundError:
            If `config.yaml` does not exist.

        KeyError:
            If required configuration keys are missing.

        yaml.YAMLError:
            If the YAML file cannot be parsed.

    Notes:
        The GCP service account path is derived automatically from
        `PROJECT_HOME_PATH` using the following convention:

        credentials/gcp/service_account.json
    """

    config_path: Path = project_root_path / "config.yaml"

    with open(config_path, "r") as config_file:
        _config_content: Dict = safe_load(config_file)

    PROJECT_HOME_PATH: Path = Path(_config_content["PROJECT_HOME_PATH"])
    GCP_SA_PATH: Path = PROJECT_HOME_PATH / "credentials" / "gcp" / "service_account.json"

    return ProjectConfig(
        project_home_path = PROJECT_HOME_PATH,
        project_config_path = config_path,
        gcp_sa_path = GCP_SA_PATH
    )


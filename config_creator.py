from typing import Dict
from pathlib import Path
import os
from dataclasses import dataclass, asdict

import yaml

@dataclass(frozen=True)
class Config:
    PROJECT_HOME_PATH: Path
    CONFIG_YAML_PATH: Path

    def to_dict(
            self
    ) -> Dict:
        return {
            "PROJECT_HOME_PATH": str(self.PROJECT_HOME_PATH),
            "CONFIG_YAML_PATH": str(self.CONFIG_YAML_PATH),
        }

def write_config_yaml(
        cfg: Config,
        out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as hndl:
        yaml.safe_dump(
            cfg.to_dict(),
            hndl,
            sort_keys=False,
            default_flow_style=False,
        )

def main():
    pwd: Path = Path(os.getcwd())
    config_yaml_path: Path = pwd / "config.yaml"

    cfg: Config = Config(
        PROJECT_HOME_PATH = pwd,
        CONFIG_YAML_PATH = config_yaml_path
    )

    write_config_yaml(
        cfg = cfg,
        out_path = config_yaml_path
    )


if __name__ == "__main__":
    main()
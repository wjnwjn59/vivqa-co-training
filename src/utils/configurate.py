import yaml
from pathlib import Path

def load_yaml_config(config_dir = "configs"):
    """
    Load YAML configuration files from a specified directory.
    """
    config = {}
    config_path = Path(config_dir)

    for yaml_file in config_path.glob("*.yaml"):
        with open(yaml_file, "r") as f:
            file_key = yaml_file.stem  
            config[file_key] = yaml.safe_load(f)

    return config

# def validate_config(config: Dict):
#     """Add schema validation (e.g., with pydantic)"""
#     required_keys = ["model", "training", "data"]
#     for key in required_keys:
#         if key not in config:
#             raise ValueError(f"Missing required config key: {key}")
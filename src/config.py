from dataclasses import dataclass
from pathlib import Path
from constants import CONFIG_FILE, SCRIPT_DIR
import yaml
from typing import Optional

@dataclass
class Paths:
  mdc_instructions: Path
  rules_json: Path
  output_dir: Path
  exa_results_dir: Path

@dataclass
class ApiConfig:
  llm_model: str
  rate_limit_calls: int
  rate_limit_period: int
  max_retries: int
  retry_min_wait: int
  retry_max_wait: int

@dataclass
class ExaApiConfig:
  rate_limit_calls: int
  rate_limit_period: int
  max_retries: int
  retry_min_wait: int
  retry_max_wait: int

@dataclass
class ProcessingConfig:
  max_workers: int
  chunk_size: int
  retry_failed_only: bool

@dataclass
class Config:
  paths: Paths
  api: ApiConfig
  exa_api: ExaApiConfig
  processing: ProcessingConfig

  @classmethod
  def load(cls, config_path: str = CONFIG_FILE) -> "Config":
    with open(config_path) as f:
      data = yaml.safe_load(f)
      
    return cls(
      paths=Paths(**{k: Path(v) for k,v in data["paths"].items()}),
      api=ApiConfig(**data["api"]),
      exa_api=ExaApiConfig(**data["exa_api"]),
      processing=ProcessingConfig(**data["processing"])
    )

# Global CONFIG instance
_config_instance: Optional[Config] = None

def get_config() -> Config:
    """Get the global CONFIG instance, initializing it if necessary."""
    global _config_instance
    if _config_instance is None:
        config_path = SCRIPT_DIR / CONFIG_FILE
        _config_instance = Config.load(str(config_path))
    return _config_instance

# Initialize the global CONFIG instance
CONFIG = get_config() 
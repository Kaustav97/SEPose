"""Configuration loader for CARLA pose data generation.

Loads YAML configuration and provides access to simulation parameters.
CLI arguments override config file values.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Default config path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default.
    
    Returns:
        Dictionary containing all configuration values.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested config value.
    
    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to traverse
        default: Default value if key path doesn't exist
    
    Returns:
        Value at key path or default
    
    Example:
        get_nested(config, 'camera', 'dvs', 'positive_threshold')
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def apply_cli_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Apply CLI argument overrides to config.
    
    CLI arguments take precedence over config file values.
    Only non-None CLI values override config.
    
    Args:
        config: Configuration dictionary from YAML
        args: Parsed argparse namespace
    
    Returns:
        Updated configuration dictionary
    """
    # CARLA connection
    if hasattr(args, 'host') and args.host is not None:
        config['carla']['host'] = args.host
    if hasattr(args, 'port') and args.port is not None:
        config['carla']['port'] = args.port
    if hasattr(args, 'tm_port') and args.tm_port is not None:
        config['carla']['tm_port'] = args.tm_port
    if hasattr(args, 'asynch') and args.asynch:
        config['carla']['asynch'] = True
    if hasattr(args, 'hybrid') and args.hybrid:
        config['carla']['hybrid'] = True
    if hasattr(args, 'no_rendering') and args.no_rendering:
        config['carla']['no_rendering'] = True
    
    # Vehicles
    if hasattr(args, 'number_of_vehicles') and args.number_of_vehicles is not None:
        config['actors']['vehicles']['count'] = args.number_of_vehicles
    if hasattr(args, 'filterv') and args.filterv is not None:
        config['actors']['vehicles']['filter'] = args.filterv
    if hasattr(args, 'generationv') and args.generationv is not None:
        config['actors']['vehicles']['generation'] = args.generationv
    if hasattr(args, 'safe') and args.safe:
        config['actors']['vehicles']['safe'] = True
    if hasattr(args, 'car_lights_on') and args.car_lights_on:
        config['actors']['vehicles']['car_lights_on'] = True
    if hasattr(args, 'hero') and args.hero:
        config['actors']['vehicles']['hero'] = True
    if hasattr(args, 'respawn') and args.respawn:
        config['actors']['vehicles']['respawn'] = True
    
    # Walkers
    if hasattr(args, 'number_of_walkers') and args.number_of_walkers is not None:
        config['actors']['walkers']['count'] = args.number_of_walkers
    if hasattr(args, 'filterw') and args.filterw is not None:
        config['actors']['walkers']['filter'] = args.filterw
    if hasattr(args, 'generationw') and args.generationw is not None:
        config['actors']['walkers']['generation'] = args.generationw
    if hasattr(args, 'seedw') and args.seedw is not None:
        config['actors']['walkers']['seed'] = args.seedw
    
    # Output
    if hasattr(args, 'out_dir') and args.out_dir is not None:
        config['output']['directory'] = args.out_dir
    
    # Seed for Traffic Manager
    if hasattr(args, 'seed') and args.seed is not None:
        config['carla']['seed'] = args.seed
    
    return config

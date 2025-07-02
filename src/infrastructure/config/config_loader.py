"""
Configuration loader with support for multiple sources.

This module provides a flexible configuration loading system that supports:
- Environment variables
- Configuration files (YAML, JSON, TOML)
- Command line arguments
- Remote configuration (e.g., Consul, etcd)
- Dynamic reloading
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

import yaml
try:
    import toml
except ImportError:
    toml = None

from src.infrastructure.config.settings import Settings, Environment
from src.shared.protocols import Logger


class ConfigSource(ABC):
    """Abstract base class for configuration sources."""
    
    @abstractmethod
    async def load(self) -> Dict[str, Any]:
        """Load configuration from source."""
        pass
    
    @abstractmethod
    def supports_reload(self) -> bool:
        """Check if source supports reloading."""
        pass


class EnvironmentConfigSource(ConfigSource):
    """Load configuration from environment variables."""
    
    def __init__(self, prefix: str = "HR_MATCHER_"):
        self.prefix = prefix
    
    async def load(self) -> Dict[str, Any]:
        """Load configuration from environment."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.prefix):].lower()
                
                # Handle nested keys (e.g., DATABASE__HOST -> database.host)
                if "__" in config_key:
                    parts = config_key.split("__")
                    current = config
                    
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    current[parts[-1]] = self._parse_value(value)
                else:
                    config[config_key] = self._parse_value(value)
        
        return config
    
    def _parse_value(self, value: str) -> Any:
        """Parse environment variable value."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Handle booleans
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False
        
        # Handle numbers
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def supports_reload(self) -> bool:
        """Environment variables don't support hot reload."""
        return False


class FileConfigSource(ConfigSource):
    """Load configuration from file."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._last_modified = None
    
    async def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.file_path}")
        
        # Update last modified time
        self._last_modified = self.file_path.stat().st_mtime
        
        # Load based on file extension
        suffix = self.file_path.suffix.lower()
        
        with open(self.file_path, "r") as f:
            if suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            elif suffix == ".json":
                return json.load(f)
            elif suffix == ".toml" and toml:
                return toml.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
    
    def supports_reload(self) -> bool:
        """File sources support reloading."""
        return True
    
    def has_changed(self) -> bool:
        """Check if file has been modified."""
        if not self.file_path.exists():
            return False
        
        current_mtime = self.file_path.stat().st_mtime
        return current_mtime != self._last_modified


class CommandLineConfigSource(ConfigSource):
    """Load configuration from command line arguments."""
    
    def __init__(self, args: Optional[List[str]] = None):
        self.args = args
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="HR Matcher Application",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # General options
        parser.add_argument(
            "--environment",
            choices=["development", "testing", "staging", "production"],
            default="development",
            help="Application environment"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        
        # Database options
        parser.add_argument(
            "--db-host",
            help="Database host"
        )
        parser.add_argument(
            "--db-port",
            type=int,
            help="Database port"
        )
        
        # Cache options
        parser.add_argument(
            "--cache-provider",
            choices=["redis", "memory"],
            help="Cache provider"
        )
        
        # API options
        parser.add_argument(
            "--api-host",
            help="API host"
        )
        parser.add_argument(
            "--api-port",
            type=int,
            help="API port"
        )
        
        # Feature flags
        parser.add_argument(
            "--enable-linkedin",
            action="store_true",
            help="Enable LinkedIn integration"
        )
        parser.add_argument(
            "--enable-async",
            action="store_true",
            help="Enable async processing"
        )
        
        return parser
    
    async def load(self) -> Dict[str, Any]:
        """Load configuration from command line."""
        args = self.parser.parse_args(self.args)
        config = {}
        
        # Convert args to config dict
        for key, value in vars(args).items():
            if value is not None:
                # Handle nested keys
                if "__" in key:
                    parts = key.split("__")
                    current = config
                    
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    current[parts[-1]] = value
                else:
                    config[key] = value
        
        return config
    
    def supports_reload(self) -> bool:
        """Command line doesn't support reload."""
        return False


class RemoteConfigSource(ConfigSource):
    """Load configuration from remote source (e.g., Consul, etcd)."""
    
    def __init__(
        self,
        endpoint: str,
        key_prefix: str = "hr_matcher/",
        auth_token: Optional[str] = None
    ):
        self.endpoint = endpoint
        self.key_prefix = key_prefix
        self.auth_token = auth_token
    
    async def load(self) -> Dict[str, Any]:
        """Load configuration from remote source."""
        # This is a placeholder implementation
        # In production, you would integrate with Consul, etcd, etc.
        import aiohttp
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.endpoint}/v1/kv/{self.key_prefix}",
                headers=headers,
                params={"recurse": True}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_consul_data(data)
                else:
                    raise Exception(f"Failed to load remote config: {response.status}")
    
    def _parse_consul_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Parse Consul KV data."""
        config = {}
        
        for item in data:
            key = item["Key"].replace(self.key_prefix, "")
            value = item.get("Value", "")
            
            # Consul stores values as base64
            import base64
            decoded_value = base64.b64decode(value).decode("utf-8")
            
            # Parse JSON values
            try:
                parsed_value = json.loads(decoded_value)
            except json.JSONDecodeError:
                parsed_value = decoded_value
            
            # Handle nested keys
            parts = key.split("/")
            current = config
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = parsed_value
        
        return config
    
    def supports_reload(self) -> bool:
        """Remote sources support reloading."""
        return True


class ConfigLoader:
    """Main configuration loader that combines multiple sources."""
    
    def __init__(
        self,
        sources: Optional[List[ConfigSource]] = None,
        logger: Optional[Logger] = None
    ):
        self.sources = sources or []
        self.logger = logger
        self._config_cache: Optional[Settings] = None
        self._file_observer: Optional[Observer] = None
        self._reload_callbacks: List[callable] = []
    
    def add_source(self, source: ConfigSource) -> None:
        """Add configuration source."""
        self.sources.append(source)
    
    def add_reload_callback(self, callback: callable) -> None:
        """Add callback to be called on configuration reload."""
        self._reload_callbacks.append(callback)
    
    async def load(self) -> Settings:
        """Load configuration from all sources."""
        merged_config = {}
        
        # Load from each source in order (later sources override earlier ones)
        for source in self.sources:
            try:
                source_config = await source.load()
                merged_config = self._deep_merge(merged_config, source_config)
                
                if self.logger:
                    self.logger.debug(
                        f"Loaded configuration from {source.__class__.__name__}"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Failed to load config from {source.__class__.__name__}",
                        error=e
                    )
                # Continue with other sources
        
        # Create Settings instance
        self._config_cache = Settings(**merged_config)
        
        # Set up file watching if applicable
        self._setup_file_watching()
        
        return self._config_cache
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _setup_file_watching(self) -> None:
        """Set up file watching for hot reload."""
        file_sources = [s for s in self.sources if isinstance(s, FileConfigSource)]
        
        if not file_sources:
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, loader: ConfigLoader):
                self.loader = loader
            
            def on_modified(self, event: FileModifiedEvent):
                if not event.is_directory:
                    # Reload configuration
                    asyncio.create_task(self.loader._reload_config())
        
        self._file_observer = Observer()
        handler = ConfigFileHandler(self)
        
        for source in file_sources:
            self._file_observer.schedule(
                handler,
                path=str(source.file_path.parent),
                recursive=False
            )
        
        self._file_observer.start()
    
    async def _reload_config(self) -> None:
        """Reload configuration."""
        if self.logger:
            self.logger.info("Reloading configuration...")
        
        try:
            new_config = await self.load()
            
            # Call reload callbacks
            for callback in self._reload_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(new_config)
                    else:
                        callback(new_config)
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            f"Error in reload callback: {e}",
                            error=e
                        )
            
            if self.logger:
                self.logger.info("Configuration reloaded successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to reload configuration",
                    error=e
                )
    
    def stop_watching(self) -> None:
        """Stop file watching."""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self._file_observer = None


# Factory functions
def create_default_loader(logger: Optional[Logger] = None) -> ConfigLoader:
    """Create default configuration loader."""
    loader = ConfigLoader(logger=logger)
    
    # Add environment source (highest priority)
    loader.add_source(EnvironmentConfigSource())
    
    # Add file sources
    config_files = [
        Path("config.yaml"),
        Path("config.json"),
        Path(f"config.{os.getenv('ENVIRONMENT', 'development')}.yaml"),
    ]
    
    for config_file in config_files:
        if config_file.exists():
            loader.add_source(FileConfigSource(config_file))
    
    # Add command line source (lowest priority for defaults)
    loader.add_source(CommandLineConfigSource())
    
    return loader


async def load_config_with_validation(
    loader: Optional[ConfigLoader] = None,
    validators: Optional[List[callable]] = None
) -> Settings:
    """Load and validate configuration."""
    if loader is None:
        loader = create_default_loader()
    
    # Load configuration
    config = await loader.load()
    
    # Run validators
    if validators:
        for validator in validators:
            validator(config)
    
    # Built-in validation
    config.validate_weights()
    
    return config
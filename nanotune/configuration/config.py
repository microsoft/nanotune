import collections
import copy
import json
import jsonschema
import os
from typing import Dict, Tuple, Optional, Any, Mapping, Union

from pathlib import Path
import pkg_resources as pkgr

from os.path import expanduser


class Config:
    """
    Heavily inspired by qcodes config class
    """

    config_filename = "config.json"
    config_schema_filename = "config_schema.json"
    # userconfig_filename = 'user_config.yml

    """ Default configuration """
    default_config_path = pkgr.resource_filename(__name__, config_filename)
    default_schema_path = pkgr.resource_filename(__name__, config_schema_filename)
    _loaded_config_files = [default_config_path]

    # home_dir = expanduser('~')
    """ Golabl user specific configuration"""
    user_config_path = expanduser(os.path.join("~", config_filename))
    user_schema_path = user_config_path.replace(config_filename, config_schema_filename)

    """Local user specific configuration"""
    cwd_config_path = os.path.join(Path.cwd(), config_filename)
    cwd_schema_path = cwd_config_path.replace(config_filename, config_schema_filename)

    current_config: Dict[str, Any] = {}
    current_config_path: str = ""

    default_config: Dict[str, Any] = {}
    default_schema: Dict[str, Any] = {}

    def __init__(self, path: Optional[str] = None) -> None:
        """"""
        self.config_file_path = path
        self.default_config, self.default_schema = self.load_default()
        self.update_config()

    def load_config(self, path: str) -> Dict[str, Any]:
        """ """
        with open(path, "r") as fp:
            config = json.load(fp)
        return config

    def load_default(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ """
        default_config = self.load_config(self.default_config_path)
        default_schema = self.load_config(self.default_schema_path)
        self.validate(default_config, default_schema)
        return default_config, default_schema

    def update_config(self, path: Optional[str] = None) -> Dict[str, Any]:
        """"""
        config = copy.deepcopy(self.default_config)
        self.current_schema = copy.deepcopy(self.default_schema)

        self._loaded_config_files = [self.default_config_path]

        self._update_config_from_file(
            self.default_config_path, self.default_schema_path, config
        )
        self._update_config_from_file(
            self.user_config_path, self.user_schema_path, config
        )
        self._update_config_from_file(
            self.cwd_config_path, self.cwd_schema_path, config
        )
        if path is not None:
            self.config_file_path = path
        if self.config_file_path is not None:
            config_file = os.path.join(self.config_file_path, self.config_file_name)
            schema_file = os.path.join(self.config_file_path, self.schema_file_name)
            self._update_config_from_file(config_file, schema_file, config)
        if config is None:
            raise RuntimeError(
                "Could not load config from any of the " "expected locations."
            )
        self.current_config = config
        self.current_config_path = self._loaded_config_files[-1]

        return config

    def _update_config_from_file(
        self, file_path: str, schema: str, config: Dict[str, Any]
    ) -> None:
        """"""
        if os.path.isfile(file_path):
            self._loaded_config_files.append(file_path)
            my_config = self.load_config(file_path)
            config.update(my_config)
            self.validate(config, self.current_schema)  # , schema)

    def validate(
        self,
        json_config: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None,
        #  extra_schema_path: Optional[str] = None
    ) -> None:
        """
        Validate configuration;
        """
        if json_config is None and schema is None:
            jsonschema.validate(self.current_config, self.current_schema)
        else:
            jsonschema.validate(json_config, schema)

    def __getitem__(self, name):
        val = copy.deepcopy(self.current_config)
        for key in name.split("."):
            if not isinstance(key, str):
                key = str(key)
            val = val[key]
        return val

    def __getattr__(self, name):
        return getattr(self.current_config, name)

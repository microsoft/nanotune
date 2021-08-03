""" Set up nanotune namespace """
import nanotune.configuration as ntconfig

from .version import __version__

config: ntconfig.Config = ntconfig.Config()

from nanotune.data.databases import *
from nanotune.data.dataset import Dataset
from nanotune.device.device import Device

version = __version__
meta_tag = nt.config["core"]["meta_add_on"]

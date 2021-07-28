""" Set up nanotune namespace """
import nanotune.configuration as ntconfig

from .version import __version__

config: ntconfig.Config = ntconfig.Config()

from nanotune.data.databases import *
from nanotune.data.dataset import Dataset
from nanotune.device.device import Device

repo = git.Repo(path=os.path.dirname(nt.__file__), search_parent_directories=True)
sha = repo.head.object.hexsha
version = __version__  # repo.git.rev_parse(sha, short=8)
meta_tag = nt.config["core"]["meta_add_on"]

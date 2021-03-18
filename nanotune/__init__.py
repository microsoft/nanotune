""" Set up nanotune namespace """
from .version import __version__

import nanotune.configuration as ntconfig

config: ntconfig.Config = ntconfig.Config()

from nanotune.data.databases import *
from nanotune.data.dataset import Dataset
from nanotune.device.device import Device

import git

repo = git.Repo(path=os.path.dirname(nt.__file__),
                search_parent_directories=True)
sha = repo.head.object.hexsha
git_hash = repo.git.rev_parse(sha, short=8)
meta_tag = nt.config["core"]["meta_add_on"]

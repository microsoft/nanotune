from abc import ABC, abstractclassmethod
import importlib
from typing import Any, OrderedDict, Tuple

class ISerializable(ABC):

    @staticmethod
    def _get_module_and_class_names(typename: str) -> Tuple[str, str]:
        """ private helper to return the module name and class name by
            parsing the specified typename """
        module_name = ".".join(typename.split(".")[:-1])
        class_name = typename.split(".")[-1]
        return (module_name, class_name)

    @staticmethod
    def _get_class(typename: str) -> Any: # TODO: Should return ILoadFromYaml (or something like that)
        """ private helper to return the python class specified by the typename """
        module_name, class_name = ISerializable._get_module_and_class_names(typename)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls

    @staticmethod
    def create_from_type(**kwargs) -> Any:
        """ Creates an instance of type specified by the 'type' kwarg,
            using arguments defined in the 'init' kwarg """
        typename = kwargs["type"]
        init = kwargs["init"]
        cls = ISerializable._get_class(typename)
        return cls.make(**init)

    @staticmethod
    def create_from_named_node(node) -> Any:
        name, typeinfo = next(iter(node.items()))
        typeinfo["init"]["name"] = name
        return ISerializable.create_from_type(**typeinfo)

    @abstractclassmethod
    def make(cls, **kwargs) -> Any:
        """ Creates an instance of this class """
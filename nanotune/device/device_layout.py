# Copyright (c) 2021 Jana Darulova
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Tuple

class IDeviceLayout(ABC):

    @classmethod
    @abstractmethod
    def helper_gate(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def barriers(self) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def plungers(self) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def outer_barriers(self) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def central_barrier(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def plunger_barrier_pairs(self) ->List[Tuple[int, int]]:
        pass


@dataclass
class DataClassMixin:
    """A dataclass mixin."""


class DeviceLayout(DataClassMixin, IDeviceLayout):
    """An abstract data class."""


@dataclass
class DoubleDotLayout(DeviceLayout):
    top_barrier = 0
    left_barrier = 1
    left_plunger = 2
    central_barrier_ = 3
    right_plunger = 4
    right_barrier = 5

    @classmethod
    def barriers(self):
        main_barriers = [
            self.left_barrier, self.central_barrier_, self.right_barrier,
        ]
        return main_barriers

    @classmethod
    def plungers(self):
        return [self.left_plunger, self.right_plunger]

    @classmethod
    def outer_barriers(self):
        return [self.left_barrier, self.right_barrier]

    @classmethod
    def plunger_barrier_pairs(self) ->List[Tuple[int, int]]:
        p_b_pairs = [
            (self.left_plunger, self.left_barrier),
            (self.right_plunger, self.right_barrier)
        ]
        return p_b_pairs

    @classmethod
    def central_barrier(self) -> int:
        return self.central_barrier_

    @classmethod
    def helper_gate(self) -> int:
        return self.top_barrier

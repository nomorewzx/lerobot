import abc
from dataclasses import dataclass
from typing import List

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False

@MotorsBusConfig.register_subclass("feetech_group")
@dataclass
class FeetechMotorGroupsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, List[tuple[int, str]]]
    mock: bool = False

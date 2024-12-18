from dataclasses import dataclass
from typing import Dict, Any

from pydantic import BaseModel


@dataclass
class UserInput:
    text: str


@dataclass
class FinalResult:
    result: str


@dataclass
class PipelineResult:
    dataframe: Dict[str, Any]
    description_dict: Dict[str, Any]


@dataclass
class DecisionInfo:
    info: Dict[str, Any]


@dataclass
class FinalPipelineInput:
    dataframe: Dict[str, Any]
    info: Dict[str, Any]


@dataclass
class DescriptionDict(BaseModel):
    description: Dict[str, Any]


@dataclass
class OverviewInfo:
    overview: Dict[str, Any]

from dataclasses import dataclass
from typing import Dict, Any, List

from pydantic import BaseModel, Field


@dataclass
class WorkerInput:
    dataframe: Dict[str, Any]


@dataclass
class UserInput:
    text: str


@dataclass
class FinalResult:
    result: str


# TODO: no Dicts for Structured Outputs


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


# TODO: no BaseModel, @dataclass
class DescriptionDict(BaseModel):
    description: Dict[str, Any]


class OverviewInfo(BaseModel):
    overview: Dict[str, Any]


class FilterItem(BaseModel):
    key: str
    values: List[str]


class Filters(BaseModel):
    reason: str
    filters: List[FilterItem]
    successful: bool

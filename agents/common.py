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


class DescriptionDict(BaseModel):
    description: Dict[str, Any]


class OverviewInfo(BaseModel):
    overview: Dict[str, Any]


class FilterItem(BaseModel):
    key: str = Field(..., description="Filter key")
    values: List[str] = Field(..., description="List of filter values")


class Filters(BaseModel):
    reason: str = Field(..., description="Reason why such mapping was made")
    filters: List[FilterItem] = Field(..., description="User's filters")
    successful: bool = Field(..., description="Was the mapping successful and valid")

from dataclasses import dataclass
from typing import Dict


@dataclass
class UserInput:
    text: str


@dataclass
class FinalResult:
    result: str

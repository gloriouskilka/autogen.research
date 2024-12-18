# =============================================================================
# tools/math_tools.py
# =============================================================================
from autogen_core import CancellationToken
from pydantic import BaseModel

# Define some tools for testing.


class AddNumbersInput(BaseModel):
    x: float
    y: float


class AddNumbersResult(BaseModel):
    result: float


from autogen_core.tools import BaseTool


class AddNumbersTool(BaseTool[AddNumbersInput, AddNumbersResult]):
    def __init__(self):
        super().__init__(
            args_type=AddNumbersInput,
            return_type=AddNumbersResult,
            name="add_numbers",
            description="Adds two numbers together.",
        )

    async def run(self, args: AddNumbersInput, cancellation_token: CancellationToken) -> AddNumbersResult:
        return AddNumbersResult(result=args.x + args.y)

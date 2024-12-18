# =============================================================================
# agents/calculator_agent.py
# =============================================================================
from autogen_core.tool_agent import ToolAgent
from tools.math_tools import AddNumbersTool


async def register_calculator_agent(runtime):
    tool = AddNumbersTool()
    await ToolAgent.register(
        runtime=runtime,
        type="CalculatorAgent",
        factory=lambda: ToolAgent(
            description="An agent that can perform mathematical operations.",
            tools=[tool],
        ),
    )

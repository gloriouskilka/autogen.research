# =============================================================================
# agents/calculator_agent.py
# =============================================================================
from autogen_core.tool_agent import ToolAgent
from tools import AddNumbersTool

# An agent that uses a tool to perform calculations.

# from autogen_core import ToolAgent


async def register_calculator_agent(runtime):
    tool = AddNumbersTool()
    tool_agent = ToolAgent(
        description="An agent that can perform mathematical operations.",
        tools=[tool],
    )
    await ToolAgent.register(
        runtime=runtime,
        type="CalculatorAgent",
        factory=lambda: tool_agent,
    )

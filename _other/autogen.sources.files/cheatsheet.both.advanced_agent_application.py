# advanced_agent_application.py
# A complex example demonstrating advanced agent design patterns in the Autogen framework.
# This example showcases best practices in structuring an agent-based application,
# emphasizing testability, decomposability, and extensibility.

# ======================================================================
# Project Structure
# ======================================================================
# The proposed project structure follows best practices for Python applications:
#
# advanced_agent_application/
# ├── agents/
# │   ├── __init__.py
# │   ├── assistant_agent.py
# │   ├── planner_agent.py
# │   └── executor_agent.py
# ├── tools/
# │   ├── __init__.py
# │   └── custom_tools.py
# ├── main.py
# ├── runtime.py
# ├── tests/
# │   ├── __init__.py
# │   └── test_agents.py
# └── requirements.txt

# ======================================================================
# Overview
# ======================================================================
# This application simulates a multi-agent system where agents collaborate to solve tasks.
# - AssistantAgent: Interacts with the user and coordinates the workflow.
# - PlannerAgent: Generates plans or strategies to accomplish tasks.
# - ExecutorAgent: Executes tasks and reports results.
# The agents communicate via the AgentRuntime, enabling asynchronous messaging and collaboration.

# ======================================================================
# Best Practices Architecture
# ======================================================================
# - **Modular Design**: Agents, tools, and runtime are separated into modules.
# - **Extensibility**: New agents or tools can be added without modifying existing code.
# - **Testability**: Includes a `tests` directory with unit tests for agents.
# - **Readability**: Clear naming conventions and comments for maintainability.
# - **Reusability**: Components are designed to be reusable in similar applications.

# ======================================================================
# agents/assistant_agent.py
# ======================================================================
from autogen_core import (
    BaseAgent,
    AgentRuntime,
    AgentId,
    AgentType,
    MessageContext,
    CancellationToken,
)
from autogen_core.models import (
    ChatCompletionClient,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    LLMMessage,
)


class AssistantAgent(BaseAgent):
    """Agent that interacts with the user and coordinates tasks."""

    def __init__(self, model_client: ChatCompletionClient):
        super().__init__(description="User-facing assistant agent.")
        self.model_client = model_client
        self.task_id_counter = 0  # For assigning unique task IDs

    @classmethod
    async def register(cls, runtime: AgentRuntime, agent_type: str, model_client: ChatCompletionClient):
        await super().register(runtime, type=agent_type, factory=lambda: cls(model_client))

    async def on_message_impl(self, message: str, ctx: MessageContext) -> str:
        # Process user input and delegate tasks to PlannerAgent
        self.task_id_counter += 1
        task_id = f"task_{self.task_id_counter}"
        # Forward the task to PlannerAgent
        planner_agent_id = AgentId(type="planner_agent", key="default")
        planning_result = await self.runtime.send_message(
            message={"task_description": message, "task_id": task_id},
            recipient=planner_agent_id,
            sender=self.id,
            cancellation_token=ctx.cancellation_token,
        )
        # Return the final result to the user
        return f"Task '{task_id}' completed. Result:\n{planning_result}"


# ======================================================================
# agents/planner_agent.py
# ======================================================================
from autogen_core import (
    RoutedAgent,
    message_handler,
    AgentRuntime,
    AgentId,
    AgentType,
    MessageContext,
    CancellationToken,
)


class PlannerAgent(RoutedAgent):
    """Agent that generates plans for given tasks."""

    @classmethod
    async def register(cls, runtime: AgentRuntime, agent_type: str):
        await super().register(runtime, type=agent_type, factory=lambda: cls())

    @message_handler
    async def handle_task(self, message: dict, ctx: MessageContext) -> str:
        # Generates a plan for the task and forwards it to ExecutorAgent
        task_description = message["task_description"]
        task_id = message["task_id"]
        # Create a simple plan (for demonstration)
        plan = f"Plan for '{task_description}': Step 1 -> Step 2 -> Step 3"
        # Send plan to ExecutorAgent
        executor_agent_id = AgentId(type="executor_agent", key="default")
        execution_result = await self.runtime.send_message(
            message={"plan": plan, "task_id": task_id},
            recipient=executor_agent_id,
            sender=self.id,
            cancellation_token=ctx.cancellation_token,
        )
        # Return the execution result
        return execution_result


# ======================================================================
# agents/executor_agent.py
# ======================================================================
from autogen_core import (
    RoutedAgent,
    message_handler,
    AgentRuntime,
    AgentType,
    MessageContext,
    CancellationToken,
)


class ExecutorAgent(RoutedAgent):
    """Agent that executes plans and reports results."""

    @classmethod
    async def register(cls, runtime: AgentRuntime, agent_type: str):
        await super().register(runtime, type=agent_type, factory=lambda: cls())

    @message_handler
    async def execute_plan(self, message: dict, ctx: MessageContext) -> str:
        # Executes the given plan (simulation)
        plan = message["plan"]
        task_id = message["task_id"]
        # Simulate execution
        result = f"Executed {plan} for task '{task_id}'. Success!"
        return result


# ======================================================================
# tools/custom_tools.py
# ======================================================================
from autogen_core.tools import FunctionTool


def calculate(a: int, b: int) -> int:
    """Calculates the sum of two numbers."""
    return a + b


calculate_tool = FunctionTool(calculate, description="Calculate the sum of two numbers.")

# ======================================================================
# runtime.py
# ======================================================================
from autogen_core import SingleThreadedAgentRuntime


def create_runtime() -> SingleThreadedAgentRuntime:
    runtime = SingleThreadedAgentRuntime()
    return runtime


# ======================================================================
# main.py
# ======================================================================
from agents.assistant_agent import AssistantAgent
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from runtime import create_runtime
from autogen_core import AgentId, CancellationToken
from autogen_core.models import ChatCompletionClient

# Initialize the runtime
runtime = create_runtime()
runtime.start()

# Register agents
# Note: In a real application, you might pass actual model clients and configuration
mock_model_client = ChatCompletionClient()  # Replace with actual model client
await AssistantAgent.register(runtime, agent_type="assistant_agent", model_client=mock_model_client)
await PlannerAgent.register(runtime, agent_type="planner_agent")
await ExecutorAgent.register(runtime, agent_type="executor_agent")

# Communication with the AssistantAgent
assistant_agent_id = AgentId(type="assistant_agent", key="default")
user_message = "Please organize a meeting agenda for next week."

# Send a message to the AssistantAgent
response = await runtime.send_message(
    message=user_message,
    recipient=assistant_agent_id,
    sender=AgentId(type="user", key="default"),
    cancellation_token=CancellationToken(),
)
print("Final Response to User:")
print(response)

# Shutdown the runtime gracefully
await runtime.stop()

# ======================================================================
# tests/test_agents.py
# ======================================================================
import unittest
from agents.assistant_agent import AssistantAgent
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from runtime import create_runtime
from autogen_core import AgentId, CancellationToken


class AgentTest(unittest.TestCase):

    def setUp(self):
        self.runtime = create_runtime()
        self.runtime.start()
        # Register mock agents
        self.mock_model_client = ChatCompletionClient()

        async def register_agents():
            await AssistantAgent.register(
                self.runtime, agent_type="assistant_agent", model_client=self.mock_model_client
            )
            await PlannerAgent.register(self.runtime, agent_type="planner_agent")
            await ExecutorAgent.register(self.runtime, agent_type="executor_agent")

        import asyncio

        asyncio.run(register_agents())

    def tearDown(self):
        async def stop_runtime():
            await self.runtime.stop()

        import asyncio

        asyncio.run(stop_runtime())

    def test_agent_communication(self):
        # Test communication between agents
        assistant_agent_id = AgentId(type="assistant_agent", key="default")
        user_message = "Test task."
        cancellation_token = CancellationToken()

        async def send_message():
            response = await self.runtime.send_message(
                message=user_message,
                recipient=assistant_agent_id,
                sender=AgentId(type="test_user", key="default"),
                cancellation_token=cancellation_token,
            )
            self.assertIn("Success", response)

        import asyncio

        asyncio.run(send_message())


if __name__ == "__main__":
    unittest.main()

# ======================================================================
# Key Takeaways
# ======================================================================
# - **Agents are Modular**: Each agent is defined in its own module, making it easier to manage and extend.
# - **Runtime is Centralized**: A single runtime instance manages all agents and their communication.
# - **Asynchronous Communication**: Agents communicate asynchronously, allowing for concurrent processing.
# - **Tools are Separate**: Tools are defined separately and can be shared among agents.
# - **Testing is Emphasized**: Unit tests are provided to ensure agents function correctly.
# - **Graceful Shutdown**: The runtime is properly stopped to release resources.
# - **Extensibility**: New agents or tools can be added without altering existing codebase significantly.

# ======================================================================
# Extending the Application
# ======================================================================
# To extend this application:
# - **Add New Agents**: Create new agents in the `agents/` directory and register them in `main.py`.
# - **Enhance Agents**: Add new message handlers or capabilities to existing agents.
# - **Implement New Tools**: Define additional tools in `tools/` and integrate them with agents.
# - **Scale Out**: Adjust the runtime configuration to handle more complex workflows or distributed processing.
# - **Feature Flags**: Use configuration files or environment variables to enable/disable features without code changes.

# ======================================================================
# Conclusion
# ======================================================================
# This example demonstrates how to structure a complex agent-based application using the Autogen framework.
# By following best practices in project organization and design patterns, the application remains maintainable and scalable.

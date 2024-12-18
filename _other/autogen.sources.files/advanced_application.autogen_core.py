# advanced_application.py: Advanced Agent Design Patterns with Autogen Framework

# This example demonstrates how to build a complex, well-structured, and extensible agentic application
# using the Autogen framework. It focuses on advanced agent design patterns, project structure, and best practices
# for creating scalable and maintainable systems.

# =============================================================================
# Project Structure Overview
# =============================================================================

# The project is organized into several modules:

# - agents/
#   - __init__.py
#   - manager_agent.py
#   - worker_agent.py
#   - aggregator_agent.py
# - utils/
#   - __init__.py
#   - data_models.py
#   - helpers.py
# - tests/
#   - test_agents.py
# - main.py

# Each module serves a specific purpose, promoting separation of concerns and facilitating easier testing
# and extensibility.

# =============================================================================
# agents/__init__.py
# =============================================================================

# This file initializes the agents package and imports the agent classes for external use.

# agents/__init__.py
"""
from .manager_agent import ManagerAgent
from .worker_agent import WorkerAgent
from .aggregator_agent import AggregatorAgent
"""

# =============================================================================
# utils/__init__.py
# =============================================================================

# Initializes the utils package and imports utility functions and data models.

# utils/__init__.py
"""
from .data_models import TaskRequest, TaskResult, AggregatedResult
from .helpers import process_data, compute_aggregated_result
"""

# =============================================================================
# utils/data_models.py
# =============================================================================

# Contains data models used throughout the application.

# utils/data_models.py
"""
from pydantic import BaseModel
from typing import List

class TaskRequest(BaseModel):
    task_id: str
    data: List[int]

class TaskResult(BaseModel):
    task_id: str
    result: int

class AggregatedResult(BaseModel):
    total: int
    average: float
"""

# =============================================================================
# utils/helpers.py
# =============================================================================

# Contains helper functions used by agents.

# utils/helpers.py
"""
def process_data(data: List[int]) -> int:
    # Simulate data processing
    return sum(data)

def compute_aggregated_result(results: List[int]) -> dict:
    total = sum(results)
    average = total / len(results) if results else 0
    return {'total': total, 'average': average}
"""

# =============================================================================
# agents/manager_agent.py
# =============================================================================

# The ManagerAgent assigns tasks to WorkerAgents and collects their results.

# agents/manager_agent.py
from autogen_core import RoutedAgent, rpc, AgentRuntime, AgentId, MessageContext
from utils.data_models import TaskRequest, AggregatedResult
from agents.worker_agent import WorkerAgent
from agents.aggregator_agent import AggregatorAgent
from typing import List, Dict


class ManagerAgent(RoutedAgent):
    def __init__(self, description: str, runtime: AgentRuntime):
        super().__init__(description)
        self.runtime = runtime
        self.worker_agents: Dict[str, AgentId] = {}
        self.aggregator_agent_id = AgentId(type="AggregatorAgent", key="default")

    async def initialize_workers(self, num_workers: int):
        # Dynamically create WorkerAgents
        for i in range(num_workers):
            worker_key = f"worker_{i}"
            agent_id = AgentId(type="WorkerAgent", key=worker_key)
            await WorkerAgent.register(
                runtime=self.runtime,
                type="WorkerAgent",
                key=worker_key,
                factory=lambda: WorkerAgent(description=f"Worker {i}"),
            )
            self.worker_agents[worker_key] = agent_id

    @rpc
    async def handle_task_requests(self, tasks: List[TaskRequest], ctx: MessageContext) -> AggregatedResult:
        # Distribute tasks among WorkerAgents
        results = []
        for task in tasks:
            worker_key = task.task_id  # Simple mapping for this example
            worker_id = self.worker_agents.get(worker_key)
            if not worker_id:
                raise ValueError(f"No worker agent found for task {task.task_id}")

            result = await self.send_message(
                message=task,
                recipient=worker_id,
                cancellation_token=ctx.cancellation_token,
            )
            results.append(result.result)

        # Send results to AggregatorAgent
        aggregated_result = await self.send_message(
            message=results,
            recipient=self.aggregator_agent_id,
            cancellation_token=ctx.cancellation_token,
        )
        return AggregatedResult(**aggregated_result)


# =============================================================================
# agents/worker_agent.py
# =============================================================================

# WorkerAgents process individual tasks.

# agents/worker_agent.py
from autogen_core import RoutedAgent, rpc, MessageContext
from utils.data_models import TaskRequest, TaskResult
from utils.helpers import process_data


class WorkerAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def process_task(self, message: TaskRequest, ctx: MessageContext) -> TaskResult:
        # Perform data processing
        result = process_data(message.data)
        return TaskResult(task_id=message.task_id, result=result)


# =============================================================================
# agents/aggregator_agent.py
# =============================================================================

# The AggregatorAgent aggregates results from WorkerAgents.

# agents/aggregator_agent.py
from autogen_core import RoutedAgent, rpc, MessageContext
from utils.helpers import compute_aggregated_result
from typing import List


class AggregatorAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def aggregate_results(self, message: List[int], ctx: MessageContext) -> dict:
        # Aggregate results
        aggregated_result = compute_aggregated_result(message)
        return aggregated_result


# =============================================================================
# main.py
# =============================================================================

# The main entry point of the application.

# main.py
import asyncio
from autogen_core import SingleThreadedAgentRuntime, AgentId
from agents.manager_agent import ManagerAgent
from agents.aggregator_agent import AggregatorAgent
from utils.data_models import TaskRequest


async def main():
    # Initialize the agent runtime
    runtime = SingleThreadedAgentRuntime()

    # Register the AggregatorAgent
    await AggregatorAgent.register(
        runtime=runtime, type="AggregatorAgent", factory=lambda: AggregatorAgent(description="Aggregates results")
    )

    # Create and register the ManagerAgent
    manager_agent = ManagerAgent(description="Task Manager", runtime=runtime)
    await ManagerAgent.register(runtime=runtime, type="ManagerAgent", factory=lambda: manager_agent)

    # Start the runtime
    runtime.start()

    # Initialize WorkerAgents
    num_workers = 3
    await manager_agent.initialize_workers(num_workers)

    # Create a list of tasks
    tasks = [TaskRequest(task_id=f"worker_{i}", data=[i, i + 1, i + 2]) for i in range(num_workers)]

    # Send tasks to ManagerAgent and get the aggregated result
    result = await runtime.send_message(
        message=tasks,
        recipient=AgentId(type="ManagerAgent", key="default"),
        sender=AgentId(type="Client", key="default"),
    )

    print(f"Aggregated Result: {result}")

    # Stop the runtime
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())

# =============================================================================
# Explanation and Best Practices
# =============================================================================

# 1. **Project Structure and Modularity**

#    - **Separation of Concerns**: Each agent is defined in its own module within the agents package.
#      Utility functions and data models are placed in the utils package.
#    - **Reusability**: By decoupling agents and utilities, components can be reused in other applications.
#    - **Scalability**: New agents or functionalities can be added without modifying existing code.

# 2. **Agent Design Patterns**

#    - **Manager-Worker Pattern**: The ManagerAgent assigns tasks to WorkerAgents and aggregates results via the AggregatorAgent.
#    - **Dynamic Agent Initialization**: The ManagerAgent dynamically initializes WorkerAgents based on the number of tasks.
#    - **Message Passing**: Agents communicate through messages, promoting loose coupling.
#    - **Using Data Models**: Pydantic models (`TaskRequest`, `TaskResult`, etc.) enforce data validation and serialization.

# 3. **Extensibility**

#    - **Adding Features**: Additional processing steps or agents can be integrated by creating new modules and updating the ManagerAgent.
#    - **Plug-in Architecture**: Agents can be extended or replaced without affecting the overall system.

# 4. **Testability**

#    - **Unit Tests**: Agents and utility functions can be individually tested by importing them from their modules.
#    - **Mocking and Stubs**: Mock agent interactions by simulating message responses.

# Example Test Structure: `tests/test_agents.py`
"""
import pytest
from agents.worker_agent import WorkerAgent
from utils.data_models import TaskRequest

@pytest.mark.asyncio
async def test_worker_agent():
    agent = WorkerAgent(description='Test Worker')
    task_request = TaskRequest(task_id='test_task', data=[1, 2, 3])
    result = await agent.process_task(task_request, None)
    assert result.result == 6
"""

# 5. **Handling Future Enhancements**

#    - **Scalability**: The number of WorkerAgents can be easily scaled up or down.
#    - **Feature Flags**: Implement feature toggles to enable or disable features without code changes.
#    - **Configuration Management**: Use configuration files (e.g., YAML, JSON) to manage settings.

# 6. **Best Practices**

#    - **Asynchronous Programming**: Utilize async/await for all I/O-bound operations.
#    - **Avoid Blocking Calls**: Ensure that long-running operations do not block the event loop.
#    - **Error Handling**: Implement comprehensive try-except blocks and handle exceptions gracefully.
#    - **Logging**: Integrate logging for monitoring and debugging.
#    - **Documentation**: Provide docstrings and comments for all modules, classes, and functions.
#    - **Type Annotations**: Use type hints throughout the codebase to improve readability and facilitate static analysis.

# =============================================================================
# Conclusion
# =============================================================================

# This example demonstrates how to structure a complex agentic application using the Autogen framework.
# By following best practices in project organization, agent design patterns, and extensibility,
# you can create robust and maintainable systems that are ready for production and can evolve with future requirements.

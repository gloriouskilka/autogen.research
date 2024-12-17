# CheatSheet.py: A Comprehensive Guide to the Autogen Framework

# This cheat sheet provides examples and best practices for using the Autogen framework.
# It is intended for professional Python developers who want to quickly understand how to
# properly implement solutions using Autogen.

# Import necessary modules from the autogen_core package
from autogen_core import (
    AgentId,
    AgentRuntime,
    AgentInstantiationContext,
    AgentType,
    BaseAgent,
    CancellationToken,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
    event,
    rpc,
)
from autogen_core.tools import FunctionTool
from autogen_core.tool_agent import ToolAgent, tool_agent_caller_loop
from autogen_core.models import (
    ChatCompletionClient,
    ModelCapabilities,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage,
    FunctionExecutionResult,
    LLMMessage,
)
from autogen_core._types import FunctionCall
from typing import Any, List
import asyncio

# 1. Setting Up the Agent Runtime

# Create an instance of the AgentRuntime.
# For single-threaded applications, use SingleThreadedAgentRuntime.
runtime = SingleThreadedAgentRuntime()

# 2. Defining Agents

# Agents are the fundamental units of Autogen.
# You can create custom agents by subclassing BaseAgent or RoutedAgent.


# Example: Creating a simple agent that echoes received messages.
class EchoAgent(BaseAgent):
    def __init__(self, description: str):
        super().__init__(description)

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        # Echo the received message back to the sender
        return f"Echo: {message}"


# Register the EchoAgent with the runtime
async def register_echo_agent():
    await EchoAgent.register(
        runtime=runtime,
        type="EchoAgent",
        factory=lambda: EchoAgent(description="An agent that echoes messages."),
    )


# 3. Using RoutedAgent for Advanced Message Handling

# RoutedAgent allows you to define message handlers using decorators for cleaner code.


class MyRoutedAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @message_handler
    async def handle_string_message(self, message: str, ctx: MessageContext) -> str:
        # Handle messages of type str
        return f"Processed string message: {message}"

    @event
    async def handle_event_message(self, message: Any, ctx: MessageContext) -> None:
        # Handle event messages (no response expected)
        print(f"Received event message: {message}")

    @rpc
    async def handle_rpc_message(self, message: int, ctx: MessageContext) -> int:
        # Handle RPC calls and return a response
        return message * 2


# Register the MyRoutedAgent
async def register_my_routed_agent():
    await MyRoutedAgent.register(
        runtime=runtime,
        type="MyRoutedAgent",
        factory=lambda: MyRoutedAgent(description="A routed agent example."),
    )


# 4. Defining Subscriptions

# Subscriptions determine how agents receive messages published to topics.

# Example: Creating a TypeSubscription to subscribe an agent to a specific topic type.
my_subscription = DefaultTopicId(type="my_topic", source="default")
await runtime.add_subscription(my_subscription)


# Alternatively, use the @default_subscription decorator
@default_subscription()
class SubscribedAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @message_handler
    async def handle_default_subscription(self, message: Any, ctx: MessageContext) -> str:
        return f"Handled default subscription message: {message}"


# Register the SubscribedAgent
async def register_subscribed_agent():
    await SubscribedAgent.register(
        runtime=runtime,
        type="SubscribedAgent",
        factory=lambda: SubscribedAgent(description="An agent with a default subscription."),
    )


# 5. Sending Messages Between Agents


# Sending a direct message to an agent
async def send_direct_message():
    sender_id = AgentId(type="MyRoutedAgent", key="default")
    recipient_id = AgentId(type="EchoAgent", key="default")
    response = await runtime.send_message(
        message="Hello, EchoAgent!",
        recipient=recipient_id,
        sender=sender_id,
    )
    print(f"Received response: {response}")


# Publishing a message to a topic
async def publish_message():
    topic_id = DefaultTopicId(type="my_topic", source="default")
    await runtime.publish_message(
        message="Broadcast message to subscribers.",
        topic_id=topic_id,
    )


# 6. Using Tools and Code Execution

# Autogen provides tools for code execution, such as FunctionTool.


# Define a function to expose as a tool
async def get_stock_price(ticker: str) -> float:
    """Simulate fetching the stock price for a given ticker symbol."""
    import random

    return random.uniform(100.0, 500.0)


# Create a FunctionTool for the function
stock_price_tool = FunctionTool(
    func=get_stock_price,
    description="Fetch the current stock price for a given ticker symbol.",
)

# Implement a ToolAgent to handle function calls
tools = [stock_price_tool]


async def register_tool_agent():
    tool_agent = ToolAgent(
        description="An agent that executes tools.",
        tools=tools,
    )
    await ToolAgent.register(
        runtime=runtime,
        type="ToolAgent",
        factory=lambda: tool_agent,
    )


# 7. Integrating with LLMs


# Assume you have an implementation of ChatCompletionClient
class DummyChatCompletionClient(ChatCompletionClient):
    async def create(
        self,
        messages: List[LLMMessage],
        tools: List[ToolAgent] = [],
        json_output: bool = None,
        extra_create_args: dict = {},
        cancellation_token: CancellationToken = None,
    ) -> AssistantMessage:
        # Dummy implementation
        return AssistantMessage(content="This is a dummy response.", source="assistant")

    def actual_usage(self):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def total_usage(self):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def count_tokens(self, messages, tools=[]):
        return 0

    def remaining_tokens(self, messages, tools=[]):
        return 1000

    @property
    def capabilities(self):
        return ModelCapabilities(
            vision=False,
            function_calling=True,
            json_output=True,
        )


# Instantiate the chat client
chat_client = DummyChatCompletionClient()


# Start a conversation using the tool agent
async def have_conversation():
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the stock price of AAPL?", source="user"),
    ]

    tool_agent_id = AgentId(type="ToolAgent", key="default")

    # Run the tool agent caller loop
    generated_messages = await tool_agent_caller_loop(
        caller=runtime,
        tool_agent_id=tool_agent_id,
        model_client=chat_client,
        input_messages=messages,
        tool_schema=[tool.schema for tool in tools],
        cancellation_token=CancellationToken(),
        caller_source="assistant",
    )

    # Process the generated messages
    for msg in generated_messages:
        if isinstance(msg, AssistantMessage):
            print(f"Assistant: {msg.content}")
        elif isinstance(msg, FunctionExecutionResultMessage):
            for result in msg.content:
                print(f"Function Result: {result.content}")


# 8. Best Practices and Tips

# - Use `async` and `await` for all asynchronous operations.
# - Handle cancellation tokens to allow for graceful shutdowns.
# - Organize your agents logically with descriptive types and keys.
# - Use `RoutedAgent` and decorators for clean message handling.
# - Register custom serializers if you're using custom message types.
# - Document your agents and tools with clear docstrings and comments.
# - Ensure proper error handling, especially when integrating with external systems.
# - Test your agents individually to ensure they behave as expected.

# 9. Running the Runtime


# Start the runtime processing loop
def start_runtime():
    runtime.start()


# Stop the runtime when all tasks are complete
async def stop_runtime():
    await runtime.stop()


# 10. Main Execution


async def main():
    # Register agents
    await register_echo_agent()
    await register_my_routed_agent()
    await register_subscribed_agent()
    await register_tool_agent()

    # Start the runtime
    start_runtime()

    # Interact with agents
    await send_direct_message()
    await publish_message()
    await have_conversation()

    # Allow time for processing
    await asyncio.sleep(1)

    # Stop the runtime
    await stop_runtime()


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

# Final Notes:
# - Adjust and extend the code to fit your specific use cases.
# - The Autogen framework is powerful for building asynchronous, message-driven applications.
# - Stay updated with the Autogen documentation for the latest features and changes.


# A Comprehensive Guide to the Autogen Framework (Continued)

# This cheat sheet continues to provide examples and best practices for using the Autogen framework.
# It delves deeper into advanced topics to help you build complex, efficient, and extensible agentic systems,
# suitable for production environments where token efficiency and solution stability are critical.

# =============================================================================
# 11. Advanced Agent Design Patterns
# =============================================================================

# In complex systems, it's essential to design agents that are modular, maintainable, and scalable.

# 11.1. Composing Agents with Sub-Agents

# Agents can be composed of other agents to divide responsibilities and promote code reusability.

# Example: An Agent that delegates tasks to Sub-Agents


class ManagerAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)
        # Maintain references to sub-agents
        self.sub_agents = {
            "taskA": AgentId(type="TaskAgentA", key="default"),
            "taskB": AgentId(type="TaskAgentB", key="default"),
        }

    @rpc
    async def handle_task_request(self, message: str, ctx: MessageContext) -> str:
        # Delegate tasks based on message content
        if "taskA" in message:
            response = await self.send_message(
                message="Execute Task A",
                recipient=self.sub_agents["taskA"],
                cancellation_token=ctx.cancellation_token,
            )
        elif "taskB" in message:
            response = await self.send_message(
                message="Execute Task B",
                recipient=self.sub_agents["taskB"],
                cancellation_token=ctx.cancellation_token,
            )
        else:
            response = "Unknown task"
        return response


# Sub-Agent implementations
class TaskAgentA(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def execute_task_a(self, message: str, ctx: MessageContext) -> str:
        # Implementation of Task A
        return "Task A completed"


class TaskAgentB(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def execute_task_b(self, message: str, ctx: MessageContext) -> str:
        # Implementation of Task B
        return "Task B completed"


# Register the ManagerAgent and Sub-Agents
async def register_manager_and_sub_agents():
    await ManagerAgent.register(
        runtime=runtime,
        type="ManagerAgent",
        factory=lambda: ManagerAgent(description="Manages tasks by delegating to sub-agents."),
    )
    await TaskAgentA.register(
        runtime=runtime,
        type="TaskAgentA",
        factory=lambda: TaskAgentA(description="Handles Task A."),
    )
    await TaskAgentB.register(
        runtime=runtime,
        type="TaskAgentB",
        factory=lambda: TaskAgentB(description="Handles Task B."),
    )


# 11.2. Using Agent Factories for Dynamic Agent Creation

# Agent factories allow you to create agents dynamically, which is useful in scenarios where the number of agents
# is determined at runtime.

# Example: Dynamic creation of agents based on incoming messages


class DynamicAgentFactory:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.created_agents = {}

    async def create_agent(self, agent_type: str, agent_key: str, description: str) -> AgentId:
        agent_id = AgentId(type=agent_type, key=agent_key)
        if agent_id not in self.created_agents:
            await BaseAgent.register(
                runtime=self.runtime,
                type=agent_type,
                factory=lambda: BaseAgent(description=description),
            )
            self.created_agents[agent_id] = True
        return agent_id


# Usage within an agent:


class DynamicManagerAgent(RoutedAgent):
    def __init__(self, description: str, agent_factory: DynamicAgentFactory):
        super().__init__(description)
        self.agent_factory = agent_factory

    @rpc
    async def handle_dynamic_request(self, message: str, ctx: MessageContext) -> str:
        # Dynamically create an agent based on the message content
        agent_key = message  # Assume the message contains the agent key
        agent_id = await self.agent_factory.create_agent(
            agent_type="DynamicAgent",
            agent_key=agent_key,
            description=f"Dynamically created agent with key {agent_key}",
        )
        # Interact with the dynamically created agent
        response = await self.send_message(
            message="Hello, DynamicAgent!",
            recipient=agent_id,
            cancellation_token=ctx.cancellation_token,
        )
        return response


# =============================================================================
# 12. Best Practices for Token Efficiency
# =============================================================================

# In production environments, it's crucial to minimize token usage to reduce costs and improve system performance.

# 12.1. Use Efficient Message Formats

# - Avoid unnecessary verbosity in messages.
# - Use concise but clear language in prompts and responses.

# 12.2. Limit Context Size

# - Use context management strategies like buffering or head-and-tail contexts to limit the number of messages
#   sent to the LLM.

from autogen_core.model_context import BufferedChatCompletionContext

# Example: Using BufferedChatCompletionContext to limit message history

chat_context = BufferedChatCompletionContext(buffer_size=5)

# In your agent, use the chat context to manage messages


class TokenEfficientAgent(RoutedAgent):
    def __init__(self, description: str, chat_context: BufferedChatCompletionContext):
        super().__init__(description)
        self.chat_context = chat_context

    @rpc
    async def handle_request(self, message: str, ctx: MessageContext) -> str:
        # Add the incoming message to the context
        await self.chat_context.add_message(UserMessage(content=message, source="user"))

        # Get messages to send to the LLM
        messages = await self.chat_context.get_messages()

        # Interact with the LLM (assuming you have a chat client)
        response = await chat_client.create(messages)

        # Add the assistant's response to the context
        await self.chat_context.add_message(response)

        return response.content


# 12.3. Function Calling Instead of Textual Instructions

# - Use function calling capabilities to let the LLM invoke functions directly, reducing the need for verbose instructions.

# =============================================================================
# 13. Strategies for Solution Stability
# =============================================================================

# 13.1. Implement Robust Error Handling

# - Use try-except blocks to catch exceptions and handle them gracefully.
# - Define custom exceptions for different error scenarios.

# Example: Handling exceptions in message handlers


class RobustAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def handle_rpc(self, message: str, ctx: MessageContext) -> str:
        try:
            # Simulate a potential error
            if "error" in message:
                raise ValueError("An error occurred while processing the message.")
            return f"Processed message: {message}"
        except Exception as e:
            # Handle the exception and return an error message
            return f"Error: {str(e)}"


# 13.2. Use Cancellation Tokens Appropriately

# - Pass cancellation tokens to asynchronous operations to allow for graceful cancellation.
# - Check for cancellation in long-running tasks.

# Example: Checking for cancellation in a long-running task


class LongRunningAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def handle_long_task(self, message: str, ctx: MessageContext) -> str:
        # Simulate a long-running task
        for i in range(10):
            # Periodically check for cancellation
            if ctx.cancellation_token.is_cancelled():
                return "Operation cancelled."
            await asyncio.sleep(1)  # Simulate work
        return "Long-running task completed."


# 13.3. Implement Retries and Backoff Strategies

# - Implement retries for operations that might fail due to transient issues.
# - Use exponential backoff to avoid overwhelming the system.

# Example: Retrying a task with exponential backoff


async def retry_with_backoff(coro, max_retries=3, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await coro()
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise  # Re-raise the exception after max retries


# Usage within an agent


class RetryingAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def handle_task_with_retries(self, message: str, ctx: MessageContext) -> str:
        async def task():
            # Simulate a task that may fail
            if "fail" in message:
                raise ValueError("Simulated failure.")
            return "Task succeeded."

        try:
            result = await retry_with_backoff(task)
            return result
        except Exception as e:
            return f"Task failed after retries: {str(e)}"


# =============================================================================
# 14. Extensibility Tips
# =============================================================================

# 14.1. Use Inheritance and Mixin Classes

# - Create base classes or mixins with common functionality that can be reused across agents.


class LoggingMixin:
    async def log_message(self, message: str):
        print(f"[{self.id}] {message}")


class LoggingAgent(RoutedAgent, LoggingMixin):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def handle_request(self, message: str, ctx: MessageContext) -> str:
        await self.log_message(f"Received message: {message}")
        return "Message logged."


# 14.2. Define Protocols and Interfaces

# - Use Python's Protocols (from typing) to define interfaces for agents and tools.

from typing import Protocol


class Processor(Protocol):
    async def process(self, data: Any) -> Any: ...


class ProcessingAgent(RoutedAgent):
    def __init__(self, description: str):
        super().__init__(description)

    @rpc
    async def handle_process(self, data: Any, ctx: MessageContext) -> Any:
        processor: Processor = await self.get_processor()
        result = await processor.process(data)
        return result

    async def get_processor(self) -> Processor:
        # Return an instance of a class that implements the Processor protocol
        ...


# 14.3. Modularize Your Code

# - Organize your agents, tools, and utilities into separate modules and packages for better maintainability.

# =============================================================================
# 15. Saving and Loading State
# =============================================================================

# Agents can save and load their state, which is useful for persisting information across restarts.

# Example: Implementing state saving in an agent


class StatefulAgent(BaseAgent):
    def __init__(self, description: str):
        super().__init__(description)
        self.counter = 0

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        self.counter += 1
        return f"Message {self.counter}: {message}"

    async def save_state(self) -> dict:
        # Return a JSON-serializable representation of the state
        return {"counter": self.counter}

    async def load_state(self, state: dict) -> None:
        self.counter = state.get("counter", 0)


# Saving and loading the runtime state


async def save_runtime_state():
    state = await runtime.save_state()
    # Save 'state' to persistent storage (e.g., file, database)
    ...


async def load_runtime_state():
    # Load 'state' from persistent storage
    ...
    await runtime.load_state(state)


# =============================================================================
# 16. Custom Serialization for Complex Message Types
# =============================================================================

# If you use complex custom message types, you may need to register custom serializers.

from autogen_core._serialization import MessageSerializer


class CustomMessage:
    def __init__(self, data: Any):
        self.data = data


class CustomMessageSerializer(MessageSerializer[CustomMessage]):
    @property
    def data_content_type(self) -> str:
        return "application/json"

    @property
    def type_name(self) -> str:
        return "CustomMessage"

    def serialize(self, message: CustomMessage) -> bytes:
        import json

        return json.dumps({"data": message.data}).encode("utf-8")

    def deserialize(self, payload: bytes) -> CustomMessage:
        import json

        data = json.loads(payload.decode("utf-8"))
        return CustomMessage(data=data["data"])


# Register the custom serializer with the runtime

serializer = CustomMessageSerializer()
runtime.add_message_serializer(serializer)

# =============================================================================
# 17. Integrating External Systems
# =============================================================================

# Agents can interact with external systems, such as databases, APIs, or services.

# Example: Agent interacting with an external API


class ExternalApiAgent(RoutedAgent):
    def __init__(self, description: str, api_client: Any):
        super().__init__(description)
        self.api_client = api_client

    @rpc
    async def handle_api_request(self, endpoint: str, ctx: MessageContext) -> Any:
        response = await self.api_client.get(endpoint)
        return response


# Ensure that you handle exceptions and implement retries as necessary.

# =============================================================================
# 18. Security Considerations
# =============================================================================

# 18.1. Input Validation

# - Validate all inputs from external sources to prevent injection attacks.

# 18.2. Permissions and Access Control

# - Implement permissions checks if agents should have restricted access to certain operations.

# 18.3. Code Execution Safety

# - Be cautious when executing code generated by LLMs.
# - Use sandboxes or other isolation mechanisms if possible.

# =============================================================================
# 19. Profiling and Performance Optimization
# =============================================================================

# - Use profiling tools to identify bottlenecks in your agents.
# - Optimize critical sections of code.
# - Ensure that asynchronous operations are used effectively to prevent blocking the event loop.

# =============================================================================
# 20. Conclusion
# =============================================================================

# - The Autogen framework provides a powerful and flexible way to build agentic systems.
# - By following best practices and utilizing the advanced features of the framework,
#   you can create efficient, stable, and maintainable applications suitable for production environments.

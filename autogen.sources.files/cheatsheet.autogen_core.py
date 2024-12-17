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

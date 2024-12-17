# CheatSheet.py
# Comprehensive guide to using the Autogen framework for building complex AI applications.
# This cheat sheet demonstrates the key features, best practices, and tips for leveraging Autogen.

# ======================================================================
# Introduction
# ======================================================================
# Autogen is a powerful framework for building AI agents and applications that utilize Large Language Models (LLMs).
# It provides abstractions for agents, tools, conversations, and more, enabling developers to create complex, interactive AI systems.

# Install Autogen using pip:
# pip install autogen

# Import necessary modules from Autogen
from autogen_core import (
    AgentRuntime,
    BaseAgent,
    AgentId,
    AgentType,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    event,
    rpc,
    DefaultTopicId,
    DefaultSubscription,
    TypeSubscription,
    SingleThreadedAgentRuntime,
)
from autogen_core.models import (
    ChatCompletionClient,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    LLMMessage,
)
from autogen_core.tools import FunctionTool, Tool, CodeExecutionResult

# ======================================================================
# Setting Up the Runtime
# ======================================================================
# The AgentRuntime is responsible for managing agent communication and execution.
runtime = SingleThreadedAgentRuntime()

# Start the runtime to begin processing messages.
runtime.start()

# ======================================================================
# Creating Agents
# ======================================================================
# Agents are the core components that handle messages and perform actions.

# Example 1: Basic Agent


class EchoAgent(BaseAgent):
    @classmethod
    async def register(cls, runtime: AgentRuntime, agent_type: str):
        await super().register(runtime, type=agent_type, factory=lambda: cls())

    async def on_message_impl(self, message: str, ctx: MessageContext) -> str:
        # Simply echoes back the received message.
        return f"Echo: {message}"


# Register the EchoAgent with the runtime.
await EchoAgent.register(runtime, agent_type="echo_agent")

# Example 2: Routed Agent with Handlers


class MathAgent(RoutedAgent):
    @message_handler
    async def handle_addition(self, message: dict, ctx: MessageContext) -> str:
        # Handles messages with 'action' == 'add' and returns the sum.
        if message.get("action") == "add":
            result = message["a"] + message["b"]
            return f"Result: {result}"
        raise CantHandleException()

    @event
    async def handle_event(self, message: dict, ctx: MessageContext) -> None:
        # Handles events (messages without a response).
        print(f"Received event: {message}")


# Register the MathAgent.
await MathAgent.register(runtime, agent_type="math_agent", factory=lambda: MathAgent())

# ======================================================================
# Using Tools
# ======================================================================
# Tools are functionalities that agents can use or expose to other agents.

# Example: Simple Tool


def get_current_time() -> str:
    """Returns the current system time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Wrap the function as a FunctionTool for use in agents.
current_time_tool = FunctionTool(get_current_time, description="Get the current system time.")

# ======================================================================
# Creating a Tool Agent
# ======================================================================
# An agent that can execute tools upon request.


class ToolAgent(BaseAgent):
    def __init__(self, tools: list[Tool]):
        super().__init__(description="Tool Executor Agent")
        self.tools = {tool.name: tool for tool in tools}

    async def on_message_impl(self, message: dict, ctx: MessageContext) -> str:
        # Expects a message with 'tool_name' and 'args'.
        tool_name = message["tool_name"]
        args = message.get("args", {})
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            result = await tool.run_json(args, ctx.cancellation_token)
            return tool.return_value_as_string(result)
        else:
            return f"Tool '{tool_name}' not found."


# Register the ToolAgent with the available tools.
await ToolAgent.register(runtime, agent_type="tool_agent", factory=lambda: ToolAgent(tools=[current_time_tool]))

# ======================================================================
# Agent Communication
# ======================================================================
# Agents can send messages to each other using the runtime.

# Creating agent IDs
echo_agent_id = AgentId(type="echo_agent", key="default")
math_agent_id = AgentId(type="math_agent", key="default")
tool_agent_id = AgentId(type="tool_agent", key="default")

# Sending a message to EchoAgent
response = await runtime.send_message(
    message="Hello, EchoAgent!", recipient=echo_agent_id, sender=AgentId(type="sender", key="default")
)
print(response)  # Output: Echo: Hello, EchoAgent!

# Sending a message to MathAgent for addition
response = await runtime.send_message(
    message={"action": "add", "a": 5, "b": 7}, recipient=math_agent_id, sender=AgentId(type="sender", key="default")
)
print(response)  # Output: Result: 12

# Requesting current time from ToolAgent
response = await runtime.send_message(
    message={"tool_name": "get_current_time"}, recipient=tool_agent_id, sender=AgentId(type="sender", key="default")
)
print(response)  # Output: 2023-11-27 12:34:56 (example)

# ======================================================================
# Publishing Messages
# ======================================================================
# Agents can publish messages to topics, which are received by agents subscribed to those topics.

# Example: Subscribing agents to a topic

# Create a subscription for the EchoAgent to receive messages published to 'broadcast' topic.
await runtime.add_subscription(TypeSubscription(topic_type="broadcast_topic", agent_type="echo_agent"))

# Publishing a message to the 'broadcast' topic.
await runtime.publish_message(
    message="Broadcast message to all subscribers!",
    topic_id=DefaultTopicId(type="broadcast_topic"),
    sender=AgentId(type="sender", key="default"),
)

# ======================================================================
# Using ChatCompletion Models
# ======================================================================
# Agents can interact with LLMs using ChatCompletion clients.

# Example: AssistantAgent utilizing OpenAI's GPT model


class AssistantAgent(BaseAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__(description="LLM-powered Assistant")
        self.model_client = model_client

    async def on_message_impl(self, message: str, ctx: MessageContext) -> str:
        # Use the model client to generate a response.
        user_message = UserMessage(content=message, source="user")
        system_message = SystemMessage(content="You are an assistant.")
        response = await self.model_client.create([system_message, user_message])
        return response.content


# Instantiate the model client (requires proper API keys and setup).
# For demonstration purposes, we'll use a mock client.


class MockChatCompletionClient(ChatCompletionClient):
    async def create(self, messages: list[LLMMessage], **kwargs) -> AssistantMessage:
        # Simulate an assistant response
        return AssistantMessage(content="This is a mock response.", source="assistant")


# Register the AssistantAgent
assistant_model_client = MockChatCompletionClient()
await AssistantAgent.register(
    runtime, agent_type="assistant_agent", factory=lambda: AssistantAgent(assistant_model_client)
)

# Send a message to the AssistantAgent
assistant_agent_id = AgentId(type="assistant_agent", key="default")
response = await runtime.send_message(
    message="What is the capital of France?", recipient=assistant_agent_id, sender=AgentId(type="user", key="default")
)
print(response)  # Output: This is a mock response.

# ======================================================================
# Handling Code Execution
# ======================================================================
# Agents can execute code snippets using Code Executors.

from autogen_core.code_executor import CodeExecutor, CodeBlock, CodeResult


# Define a simple CodeExecutor
class PythonCodeExecutor(CodeExecutor):
    async def execute_code_blocks(
        self, code_blocks: list[CodeBlock], cancellation_token: CancellationToken
    ) -> CodeResult:
        # For demonstration, we'll just simulate code execution.
        outputs = []
        for block in code_blocks:
            outputs.append(f"Executed code in language: {block.language}")
        return CodeResult(exit_code=0, output="\n".join(outputs))


# An agent that executes code sent to it.


class CodeExecutionAgent(BaseAgent):
    def __init__(self, executor: CodeExecutor):
        super().__init__(description="Agent that executes code blocks")
        self.executor = executor

    async def on_message_impl(self, message: str, ctx: MessageContext) -> str:
        # Extract code blocks from the message (simplified).
        code_blocks = [CodeBlock(code=message, language="python")]
        result = await self.executor.execute_code_blocks(code_blocks, ctx.cancellation_token)
        return result.output


# Register the CodeExecutionAgent
code_executor = PythonCodeExecutor()
await CodeExecutionAgent.register(
    runtime, agent_type="code_executor_agent", factory=lambda: CodeExecutionAgent(code_executor)
)

# Send code to be executed
code_executor_agent_id = AgentId(type="code_executor_agent", key="default")
response = await runtime.send_message(
    message='print("Hello, World!")', recipient=code_executor_agent_id, sender=AgentId(type="user", key="default")
)
print(response)  # Output: Executed code in language: python

# ======================================================================
# Best Practices
# ======================================================================

# 1. Use clear and unique agent types and IDs.
#    - Helps in identifying and communicating with the correct agents.
# 2. Leverage RoutedAgent for complex message handling.
#    - Use @message_handler, @event, and @rpc decorators for clarity.
# 3. Handle exceptions gracefully.
#    - Catch and manage exceptions within agents to prevent crashes.
# 4. Utilize tools to encapsulate functionalities.
#    - Makes it easier to reuse and manage code.
# 5. Keep agent state minimal and save/load state when necessary.
#    - Use save_state and load_state for persistence.
# 6. Use cancellation tokens to manage long-running tasks.
#    - Allows tasks to be canceled, improving responsiveness.
# 7. Ensure proper shutdown of the runtime.
#    - Use runtime.stop() when done to clean up resources.

# ======================================================================
# Graceful Shutdown
# ======================================================================

# Stop the runtime when done.
await runtime.stop()

# ======================================================================
# Advanced Topics
# ======================================================================

# Implementing Agent Teams and Conversations (See Autogen documentation for details).

# ======================================================================
# Conclusion
# ======================================================================
# This cheat sheet provides a quick reference for using the Autogen framework to build complex AI applications.
# Refer to the official Autogen documentation for more in-depth explanations and additional features.

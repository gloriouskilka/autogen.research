Autogen is an advanced framework for building AI agents, emphasizing modularity, extensibility, and high-performance asynchronous operations. Below is a condensed overview of its main components, best practices, and code snippets tailored for expert Python developers.

---

### Project Structure

```
autogen_core/
├── __init__.py
├── _agent.py
├── _agent_id.py
├── _agent_instantiation.py
├── _agent_metadata.py
├── _agent_proxy.py
├── _agent_runtime.py
├── _agent_type.py
├── _base_agent.py
├── _cancellation_token.py
├── _closure_agent.py
├── _constants.py
├── _function_utils.py
├── _image.py
├── _message_context.py
├── _message_handler_context.py
├── _routed_agent.py
├── _serialization.py
├── _single_threaded_agent_runtime.py
├── _subscription.py
├── _subscription_context.py
├── _topic.py
├── _types.py
├── exceptions.py
├── logging.py
├── models/
│   ├── __init__.py
│   └── _types.py
├── model_context/
│   ├── __init__.py
│   ├── _buffered_chat_completion_context.py
│   ├── _chat_completion_context.py
│   └── _head_and_tail_chat_completion_context.py
├── tools/
│   ├── __init__.py
│   ├── _base.py
│   ├── _code_execution.py
│   └── _function_tool.py
└── tool_agent/
    ├── __init__.py
    ├── _caller_loop.py
    └── _tool_agent.py
```

---

### Core Concepts

#### Agents and Runtime

- **Agent** (`_agent.py`): Protocol defining the interface for all agents.
- **BaseAgent** (`_base_agent.py`): Abstract base class implementing the `Agent` protocol.
- **AgentRuntime** (`_agent_runtime.py`): Manages agent lifecycles and message delivery.
- **SingleThreadedAgentRuntime** (`_single_threaded_agent_runtime.py`): A concrete `AgentRuntime` suitable for single-threaded applications.

#### Messaging

- **MessageContext** (`_message_context.py`): Provides context for messages, including sender and cancellation tokens.
- **MessageHandlerContext** (`_message_handler_context.py`): Context manager for message handlers.
- **RoutedAgent** (`_routed_agent.py`): `BaseAgent` subclass with built-in message routing using decorators.

#### Identifiers and Metadata

- **AgentId** (`_agent_id.py`): Uniquely identifies an agent via `type` and `key`.
- **AgentMetadata** (`_agent_metadata.py`): Contains metadata for agents.
- **AgentType** (`_agent_type.py`): Represents an agent's type.

#### Subscription and Topics

- **Subscription** (`_subscription.py`): Protocol for defining topic subscriptions.
- **TopicId** (`_topic.py`): Identifies a message topic with `type` and `source`.
- **TypeSubscription** (`_type_subscription.py`): Subscribes to topics based on type.
- **DefaultSubscription** (`_default_subscription.py`): Simplifies subscription creation.

#### Utilities

- **CancellationToken** (`_cancellation_token.py`): For canceling asynchronous operations.
- **Serialization** (`_serialization.py`): Handles message serialization/deserialization.
- **Function Utilities** (`_function_utils.py`): Utilities for function introspection and schema generation.
- **Code Executors** (`code_executor/`): For executing code blocks securely.

---

### Best Practices

#### Agent Implementation

- **Inheritance**: Derive from `BaseAgent` or `RoutedAgent` for custom agents.
- **Message Handling**: Use `@message_handler`, `@event`, and `@rpc` decorators for routing.
- **Context Management**: Utilize `MessageContext` and `CancellationToken` for handling message processing and cancellation.
- **State Management**: Implement `save_state` and `load_state` for agent persistence.

#### Messaging Patterns

- **Direct Messaging**: Use `send_message` for RPC-style communication.
- **Publish/Subscribe**: Use `publish_message` and subscriptions for event-driven architectures.
- **Function Calls**: Leverage `FunctionCall` and `FunctionExecutionResult` for invoking functions within agents.

#### Serialization

- **Custom Messages**: Define message types using Pydantic models or dataclasses.
- **Serializers**: Implement `MessageSerializer` for custom serialization logic.
- **Registry**: Use `SerializationRegistry` to manage serializers.

---

### Code Snippets

#### Defining an Agent with Routed Message Handling

```python
from autogen_core import RoutedAgent, message_handler, rpc, MessageContext

class MyAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="My Custom Agent")

    @message_handler
    async def handle_text(self, message: str, ctx: MessageContext) -> None:
        # Handle plain text messages
        logger.debug(f"Received message: {message}")

    @rpc
    async def compute(self, message: dict, ctx: MessageContext) -> int:
        # Handle RPC calls
        return message.get('a', 0) + message.get('b', 0)
```

#### Registering an Agent

```python
from autogen_core import SingleThreadedAgentRuntime

runtime = SingleThreadedAgentRuntime()

# Register the agent
await MyAgent.register(runtime=runtime, type="my_agent_type", factory=MyAgent)
```

#### Sending Messages

```python
from autogen_core import AgentId

agent_id = AgentId(type="my_agent_type", key="agent_key")

# Send a direct message
await runtime.send_message(message="Hello, Agent!", recipient=agent_id)

# Make an RPC call
result = await runtime.send_message(message={'a': 5, 'b': 3}, recipient=agent_id)
logger.debug(f"Result of compute: {result}")
```

#### Publishing Messages

```python
from autogen_core import TopicId

topic = TopicId(type="my_topic", source="source_id")

# Publish a message to a topic
await runtime.publish_message(message="Event Data", topic_id=topic)
```

#### Subscribing to Topics

```python
from autogen_core import TypeSubscription

subscription = TypeSubscription(topic_type="my_topic", agent_type="my_agent_type")
await runtime.add_subscription(subscription)
```

#### Defining Custom Messages with Pydantic

```python
from pydantic import BaseModel
from autogen_core import MessageSerializer, SerializationRegistry

class CustomMessage(BaseModel):
    content: str
    timestamp: float

serializer = MessageSerializer[CustomMessage]()
runtime.add_message_serializer(serializer)
```

#### Implementing a Tool Agent

```python
from autogen_core import ToolAgent, FunctionTool
from autogen_core.tool_agent import tool_agent_caller_loop

async def add(a: int, b: int) -> int:
    return a + b

add_tool = FunctionTool(func=add, description="Adds two numbers")

tool_agent = ToolAgent(description="Tool Agent", tools=[add_tool])
await tool_agent.register(runtime=runtime, type="tool_agent_type", factory=lambda: tool_agent)

# Using the tool agent in a caller loop
input_messages = []  # Populate with initial messages
output_messages = await tool_agent_caller_loop(
    caller=runtime,
    tool_agent_id=AgentId("tool_agent_type", "default"),
    model_client=model_client_instance,  # Your ChatCompletionClient instance
    input_messages=input_messages,
    tool_schema=[add_tool.schema],
)
```

---

### Advanced Concepts

#### Closure Agents

Closure agents are defined using a function and can be useful for lightweight or one-off agents.

```python
from autogen_core import ClosureAgent, ClosureContext

async def agent_logic(agent: ClosureContext, message: Any, ctx: MessageContext) -> Any:
    # Agent's behavior implemented here
    return f"Processed: {message}"

agent_type = await ClosureAgent.register_closure(
    runtime=runtime,
    type="closure_agent",
    closure=agent_logic,
    description="A Closure Agent Example"
)
```

#### Cancellation Tokens

Ensure tasks can be cancelled to prevent resource leaks.

```python
from autogen_core import CancellationToken

cancellation_token = CancellationToken()

# Linking cancellation to a task
task = asyncio.create_task(agent_instance.some_long_running_task())
cancellation_token.link_future(task)

# Cancel the operation when needed
cancellation_token.cancel()
```

#### Custom Serialization

Implement custom serialization for complex message types.

```python
from autogen_core import MessageSerializer

class MyCustomSerializer(MessageSerializer[CustomType]):
    def serialize(self, message: CustomType) -> bytes:
        # Custom serialization logic
        pass

    def deserialize(self, payload: bytes) -> CustomType:
        # Custom deserialization logic
        pass

runtime.add_message_serializer(MyCustomSerializer())
```

---

### Logging and Telemetry

Utilize the built-in logging for debugging and monitoring.

```python
import logging
from autogen_core import ROOT_LOGGER_NAME

logger = logging.getLogger(ROOT_LOGGER_NAME)
logger.setLevel(logging.DEBUG)

# Log custom messages
logger.debug("Debug information")
logger.info("Informational message")
```

---

### Key Takeaways

- **Asynchronous Design**: Agents and runtimes are designed with `asyncio` for concurrency.
- **Modularity**: Use protocols and abstract base classes to define clear interfaces.
- **Extensibility**: Implement custom agents, serializers, and tools as needed.
- **Context Management**: Leverage context managers for message handling and agent instantiation.
- **Robust Error Handling**: Use exceptions like `CantHandleException` to manage unsupported messages.

---

Autogen's framework provides a powerful foundation for building sophisticated AI agents, allowing for fine-grained control over agent behavior, communication, and lifecycle management, all while promoting best practices in modern Python development.


# Q: Thank you! Please write an example of complex pipeline with rpc calls returning data to other Agents which process the data deterministically
# A:

Here's a concise example demonstrating a complex pipeline with agents communicating via RPC calls, processing data deterministically using Autogen's framework.

```python
# Define custom messages
from dataclasses import dataclass

@dataclass
class DataRequest:
    data_id: int

@dataclass
class DataResponse:
    data_id: int
    content: str

@dataclass
class ProcessedData:
    data_id: int
    result: str

# Define Agent A: Initiates the pipeline
from autogen_core import RoutedAgent, rpc, MessageContext

class AgentA(RoutedAgent):
    def __init__(self):
        super().__init__(description="Agent A")

    @rpc
    async def start_process(self, message: dict, ctx: MessageContext) -> str:
        data_id = message['data_id']
        # Make RPC call to AgentB to fetch data
        agent_b = await self.runtime.get('agent_b_type', key='agent_b')
        data_response = await self.send_message(
            DataRequest(data_id=data_id),
            recipient=agent_b,
            cancellation_token=ctx.cancellation_token
        )
        # Process the data (deterministically)
        processed_result = f"Processed {data_response.content}"
        # Send data to AgentC for further processing
        agent_c = await self.runtime.get('agent_c_type', key='agent_c')
        final_result = await self.send_message(
            ProcessedData(data_id=data_id, result=processed_result),
            recipient=agent_c,
            cancellation_token=ctx.cancellation_token
        )
        return final_result

# Define Agent B: Fetches data
class AgentB(RoutedAgent):
    def __init__(self):
        super().__init__(description="Agent B")
        self.data_store = {1: "Data for ID 1", 2: "Data for ID 2"}

    @rpc
    async def handle_data_request(self, message: DataRequest, ctx: MessageContext) -> DataResponse:
        content = self.data_store.get(message.data_id, "Default Data")
        return DataResponse(data_id=message.data_id, content=content)

# Define Agent C: Final processing
class AgentC(RoutedAgent):
    def __init__(self):
        super().__init__(description="Agent C")

    @rpc
    async def handle_processed_data(self, message: ProcessedData, ctx: MessageContext) -> str:
        # Deterministic processing
        final_result = f"Final result for ID {message.data_id}: {message.result.upper()}"
        return final_result

# Set up the runtime and register agents
from autogen_core import SingleThreadedAgentRuntime

runtime = SingleThreadedAgentRuntime()

await AgentA.register(runtime=runtime, type="agent_a_type", factory=AgentA)
await AgentB.register(runtime=runtime, type="agent_b_type", factory=AgentB)
await AgentC.register(runtime=runtime, type="agent_c_type", factory=AgentC)

# Start the runtime
runtime.start()

# Initialize Agent A and start the process
agent_a_id = await runtime.get('agent_a_type', key='agent_a')
result = await runtime.send_message(
    message={'data_id': 1},
    recipient=agent_a_id
)
logger.debug(result)  # Output: Final result for ID 1: PROCESSED DATA FOR ID 1

# Stop the runtime when done
await runtime.stop()
```

In this example:

- **AgentA** initiates the process by making an RPC call to **AgentB** to retrieve data.
- **AgentB** receives a `DataRequest`, fetches the data deterministically from its `data_store`, and returns a `DataResponse`.
- **AgentA** processes the received data deterministically and sends a `ProcessedData` message to **AgentC**.
- **AgentC** performs final deterministic processing on the data and returns the result.
- The `SingleThreadedAgentRuntime` manages the agents and their interactions.

This pipeline demonstrates agents communicating via RPC calls, processing data deterministically at each step.


# Q: Great example, thank you! Please suggest a proper rpc call scheme when I have the start of the application, I need to determine which pipeline to call based on user input (text), then run deterministic pipeline of the processing
# A:

Certainly! To handle user input and determine the appropriate processing pipeline, you can design a `CoordinatorAgent` that:

1. Receives the user's input.
2. Analyzes the input to determine which pipeline to execute.
3. Orchestrates a deterministic sequence of RPC calls to other agents representing different processing stages.

Below is an example illustrating this scheme using Autogen's framework:

```python
# Define custom messages for different processing stages
from dataclasses import dataclass

@dataclass
class UserInput:
    text: str

@dataclass
class PipelineAData:
    text: str
    processed_step: str

@dataclass
class PipelineBData:
    text: str
    processed_step: str

@dataclass
class FinalResult:
    result: str

# Define the CoordinatorAgent
from autogen_core import RoutedAgent, rpc, MessageContext

class CoordinatorAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Coordinator Agent")

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        user_text = message.text

        # Determine which pipeline to use based on user input
        if "pipeline A" in user_text.lower():
            # Execute Pipeline A
            agent_a_id = await self.runtime.get('pipeline_a_agent_type', key='pipeline_a_agent')
            pipeline_data = PipelineAData(text=user_text, processed_step="Start")
            final_result = await self.send_message(
                message=pipeline_data,
                recipient=agent_a_id,
                cancellation_token=ctx.cancellation_token
            )
        elif "pipeline B" in user_text.lower():
            # Execute Pipeline B
            agent_b_id = await self.runtime.get('pipeline_b_agent_type', key='pipeline_b_agent')
            pipeline_data = PipelineBData(text=user_text, processed_step="Start")
            final_result = await self.send_message(
                message=pipeline_data,
                recipient=agent_b_id,
                cancellation_token=ctx.cancellation_token
            )
        else:
            # Default action or error handling
            final_result = FinalResult(result="Error: No valid pipeline identified.")

        return final_result

# Define Pipeline A Agent
class PipelineAAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Pipeline A Agent")

    @rpc
    async def process_pipeline_a(self, message: PipelineAData, ctx: MessageContext) -> FinalResult:
        # Deterministic processing steps for Pipeline A
        step1_result = await self.step_one(message)
        step2_result = await self.step_two(step1_result)
        return FinalResult(result=step2_result.processed_step)

    async def step_one(self, data: PipelineAData) -> PipelineAData:
        # Deterministic processing step one
        data.processed_step += " -> Pipeline A Step 1"
        return data

    async def step_two(self, data: PipelineAData) -> PipelineAData:
        # Deterministic processing step two
        data.processed_step += " -> Pipeline A Step 2"
        return data

# Define Pipeline B Agent
class PipelineBAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Pipeline B Agent")

    @rpc
    async def process_pipeline_b(self, message: PipelineBData, ctx: MessageContext) -> FinalResult:
        # Deterministic processing steps for Pipeline B
        step1_result = await self.other_step_one(message)
        step2_result = await self.other_step_two(step1_result)
        return FinalResult(result=step2_result.processed_step)

    async def other_step_one(self, data: PipelineBData) -> PipelineBData:
        # Deterministic processing step one
        data.processed_step += " -> Pipeline B Step 1"
        return data

    async def other_step_two(self, data: PipelineBData) -> PipelineBData:
        # Deterministic processing step two
        data.processed_step += " -> Pipeline B Step 2"
        return data

# Set up the runtime and register agents
from autogen_core import SingleThreadedAgentRuntime

runtime = SingleThreadedAgentRuntime()

# Register CoordinatorAgent
await CoordinatorAgent.register(runtime=runtime, type="coordinator_agent_type", factory=CoordinatorAgent)

# Register PipelineAAgent
await PipelineAAgent.register(runtime=runtime, type="pipeline_a_agent_type", factory=PipelineAAgent)

# Register PipelineBAgent
await PipelineBAgent.register(runtime=runtime, type="pipeline_b_agent_type", factory=PipelineBAgent)

# Start the runtime
runtime.start()

# Simulate user input and initiate processing
coordinator_agent_id = await runtime.get('coordinator_agent_type', key='coordinator_agent')

# Example user input that triggers Pipeline A
user_input = UserInput(text="Please run pipeline A with this data.")

final_result = await runtime.send_message(
    message=user_input,
    recipient=coordinator_agent_id
)

logger.debug(final_result.result)
# Output: Start -> Pipeline A Step 1 -> Pipeline A Step 2

# Example user input that triggers Pipeline B
user_input_b = UserInput(text="Execute pipeline B with different data.")

final_result_b = await runtime.send_message(
    message=user_input_b,
    recipient=coordinator_agent_id
)

logger.debug(final_result_b.result)
# Output: Start -> Pipeline B Step 1 -> Pipeline B Step 2

# Stop the runtime when done
await runtime.stop()
```

**Explanation:**

- **CoordinatorAgent**:
  - Receives `UserInput` messages.
  - Analyzes `message.text` to determine which pipeline to execute.
  - Makes an RPC call to either `PipelineAAgent` or `PipelineBAgent` based on the input.
  - Returns the final result to the caller.

- **PipelineAAgent** and **PipelineBAgent**:
  - Implement specific processing pipelines (`Pipeline A` and `Pipeline B` respectively).
  - Each defines deterministic processing steps (`step_one`, `step_two`).
  - After processing, they return a `FinalResult`.

- **Runtime Setup**:
  - Uses `SingleThreadedAgentRuntime` to manage agents.
  - All agents are registered with unique types.
  - Agents are instantiated and identified via agent IDs.

- **Processing Flow**:
  1. User sends input to `CoordinatorAgent`.
  2. `CoordinatorAgent` determines the appropriate pipeline.
  3. `CoordinatorAgent` sends an RPC message to the selected pipeline agent.
  4. The pipeline agent processes the data through deterministic steps.
  5. The final result is returned back through the `CoordinatorAgent` to the user.

**Best Practices Demonstrated:**

- **Deterministic Processing**: Each processing step in the pipeline manipulates the data in a predictable way.
- **RPC Calls**: Agents communicate via `send_message` with `cancellation_token` support.
- **Dynamic Pipeline Selection**: The `CoordinatorAgent` uses input analysis to decide which pipeline to run.
- **Separation of Concerns**: Different agents handle specific responsibilities, promoting modularity.
- **Agent Registration and Initialization**: Agents are properly registered with the runtime, ensuring they can be retrieved when needed.

**Extending the Example:**

- **Adding More Pipelines**: You can introduce additional pipelines by creating new agents similar to `PipelineAAgent` and `PipelineBAgent`.
- **Complex Input Analysis**: Implement more sophisticated logic in `CoordinatorAgent` to decide on pipelines, potentially using NLP techniques.
- **Error Handling**: Enhance the agents to handle exceptions and invalid inputs gracefully.
- **State Management**: If processing steps require maintaining state, implement `save_state` and `load_state` methods in agents.

**Note on Deterministic Processing:**

All processing steps are deterministic, meaning given the same input, they will always produce the same output without randomness or side effects. This is crucial for predictable behavior in pipelines.

**Running the Code:**

Remember to have the Autogen framework (`autogen_core`) properly installed and set up in your environment before running the code. The agents use asynchronous operations (`async`/`await`), so ensure your main execution context supports async execution (e.g., using `asyncio.run` in an async main function).

**Example of Async Main Function:**

```python
import asyncio

async def main():
    # Setup and agent registration code here...
    # (Use the code provided above)

# Run the async main function
asyncio.run(main())
```

---

By structuring your application with a coordinator agent that dynamically chooses processing pipelines based on user input, you achieve flexibility and scalability. This design allows you to easily add or modify pipelines without changing the core logic of input handling.

# Q: How to have a LLM-decided (tool call?) in the beginning selection of the pipeline? I don't want to manually write a deterministic pipeline selector like it is in CoordinatorAgent handle_user_input. I know there is a possibility to Autogen to decide the pipeline to run
# A:

Certainly! To allow the LLM to decide which pipeline to execute based on user input, you can utilize Autogen's function calling capabilities, where the LLM generates a `FunctionCall` to select the pipeline. This involves:

- **Defining tools** (functions) representing each pipeline.
- **Setting up a `ToolAgent`** capable of executing these tools.
- **Modifying the `CoordinatorAgent`** to interact with the LLM and process `FunctionCall`s.
- **Using the `tool_agent_caller_loop`** to manage the interaction between the `CoordinatorAgent`, the LLM, and the `ToolAgent`.

Here's how you can achieve this:

---

### **1. Define Pipeline Functions as Tools**

First, define each pipeline as a separate tool using `FunctionTool`.

```python
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

async def pipeline_a(text: str, cancellation_token: CancellationToken) -> str:
    # Deterministic processing steps for Pipeline A
    result = f"{text} -> Pipeline A Step 1 -> Pipeline A Step 2"
    return result

async def pipeline_b(text: str, cancellation_token: CancellationToken) -> str:
    # Deterministic processing steps for Pipeline B
    result = f"{text} -> Pipeline B Step 1 -> Pipeline B Step 2"
    return result

# Wrap the functions as tools
pipeline_a_tool = FunctionTool(func=pipeline_a, description="Process data using Pipeline A")
pipeline_b_tool = FunctionTool(func=pipeline_b, description="Process data using Pipeline B")
```

---

### **2. Set Up the Tool Agent**

Create a `ToolAgent` that can execute these tools upon the LLM's request.

```python
from autogen_core.tool_agent import ToolAgent

tool_agent = ToolAgent(description="Pipeline Tool Agent", tools=[pipeline_a_tool, pipeline_b_tool])
```

---

### **3. Modify the `CoordinatorAgent` to Use the LLM for Pipeline Selection**

In the `CoordinatorAgent`, we will:

- Send the user's input to the LLM.
- The LLM decides which pipeline to use by generating a `FunctionCall`.
- The `CoordinatorAgent` handles the `FunctionCall`, dispatching it to the `ToolAgent`.
- Collect the result and return it to the user.

```python
from autogen_core.models import ChatCompletionClient, AssistantMessage, UserMessage, LLMMessage, SystemMessage
from autogen_core import FunctionExecutionResultMessage, FunctionCall, CancellationToken
from autogen_core.tool_agent import tool_agent_caller_loop
from autogen_core import RoutedAgent, rpc, MessageContext

class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        user_text = message.text

        # Prepare the initial messages for the LLM
        input_messages: List[LLMMessage] = [
            SystemMessage(content="""
You are an assistant that decides which processing pipeline to use based on user input.
Available functions are:
- pipeline_a: Use this when the user mentions 'pipeline A' or wants detailed processing.
- pipeline_b: Use this for quick processing or when the user mentions 'pipeline B'.
"""),
            UserMessage(content=user_text, source="user")
        ]

        # The LLM will decide which pipeline to use by generating a FunctionCall.
        tool_agent_id = await self.runtime.get('tool_agent_type', key='pipeline_tool_agent')

        # Use the Tool Agent caller loop to interact with the LLM and execute the function calls
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[pipeline_a_tool.schema, pipeline_b_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant"
        )

        # Extract the final result from the generated messages
        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    return FinalResult(result=msg.content)
                elif isinstance(msg.content, list):
                    # Handle potential FunctionCalls in the assistant's response
                    continue
        return FinalResult(result="Error: No valid output from LLM")
```

**Explanation:**

- The `CoordinatorAgent` uses the LLM to interpret the user's input and decide which pipeline to execute.
- It sends a `SystemMessage` to guide the LLM on available functions (pipelines).
- The `tool_agent_caller_loop` manages the interaction, including handling `FunctionCall`s and executing the tools via the `ToolAgent`.

---

### **4. Implement a `ChatCompletionClient`**

You'll need a `ChatCompletionClient` to interface with the LLM. Below is an example using OpenAI's GPT-4 API with function calling capabilities.

```python
import openai
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, RequestUsage
from autogen_core.tools import ToolSchema
from autogen_core import FunctionCall
from typing import List, Mapping, Any

class OpenAIChatCompletionClient(ChatCompletionClient):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        openai.api_key = api_key

    async def create(
        self,
        messages: List[LLMMessage],
        tools: List[Tool | ToolSchema] = [],
        json_output: bool = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken = None,
    ) -> CreateResult:
        # Convert messages to OpenAI API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    api_messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg.content, list):
                    # Handle FunctionCalls
                    for fc in msg.content:
                        api_messages.append({
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": fc.name,
                                "arguments": fc.arguments
                            }
                        })
            elif isinstance(msg, FunctionExecutionResultMessage):
                for result in msg.content:
                    api_messages.append({
                        "role": "function",
                        "name": result.call_id,  # Use call_id as the function name
                        "content": result.content
                    })
        # Prepare functions for the API
        api_functions = [tool.schema for tool in tools]
        # Call the OpenAI API
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=api_messages,
            functions=api_functions,
            function_call="auto",
            **extra_create_args
        )
        # Extract the response
        choice = response.choices[0]
        message = choice.message
        if 'function_call' in message:
            function_call = FunctionCall(
                id="unique_call_id",  # Generate a unique ID here
                name=message.function_call['name'],
                arguments=message.function_call['arguments']
            )
            content = [function_call]
        else:
            content = message.content

        return CreateResult(
            finish_reason=choice.finish_reason,
            content=content,
            usage=RequestUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            ),
            cached=False
        )
```

**Note:**

- Replace `openai.ChatCompletion.create` with `openai.ChatCompletion.acreate` for async operation.
- Ensure that your OpenAI API key is securely stored and not hard-coded.

---

### **5. Register and Start the Agents**

```python
# Set up the runtime and register agents
runtime = SingleThreadedAgentRuntime()

# Instantiate the model client with your OpenAI API key
model_client = OpenAIChatCompletionClient(model_name="gpt-4-0613", api_key="your-openai-api-key")

# Register the CoordinatorAgent
coordinator_factory = lambda: CoordinatorAgent(model_client=model_client)
await CoordinatorAgent.register(runtime=runtime, type="coordinator_agent_type", factory=coordinator_factory)

# Register the ToolAgent
tool_agent_factory = lambda: tool_agent
await tool_agent.register(runtime=runtime, type="tool_agent_type", factory=tool_agent_factory)

# Start the runtime
runtime.start()

# Simulate user input and initiate processing
coordinator_agent_id = await runtime.get('coordinator_agent_type', key='coordinator_agent')
user_input = UserInput(text="Please process this data using the best pipeline.")

# Send the user input to the CoordinatorAgent
final_result = await runtime.send_message(
    message=user_input,
    recipient=coordinator_agent_id
)

logger.debug(final_result.result)
# Output will depend on the LLM's decision and processing result
```

---

### **6. Ensure Correct OpenAI API Usage**

When using OpenAI's APIs:

- **API Key Security**: Keep your API key secure and do not expose it in code repositories.
- **Usage Policies**: Comply with OpenAI's usage policies, including those regarding data privacy, disallowed content, and rate limits.
- **Error Handling**: Implement appropriate error handling for API calls, considering network issues, API errors, and timeouts.

---

### **Explanation and Benefits**

**LLM-Decided Pipeline Selection:**

- By incorporating the LLM into the decision-making process, you allow it to interpret complex or nuanced user inputs.
- The LLM decides which pipeline to execute by generating a `FunctionCall` corresponding to the appropriate tool.

**Function Calling Mechanism:**

- The OpenAI LLM supports function calling where it can generate structured outputs (`FunctionCall`s) that specify which function (tool) to invoke and with what arguments.
- The `tool_agent_caller_loop` utility facilitates the flow between your `CoordinatorAgent`, the LLM, and the `ToolAgent`.

**Deterministic Processing:**

- The actual processing within each pipeline remains deterministic.
- Only the selection of the pipeline is influenced by the LLM, which can be made more consistent by crafting careful prompts and providing clear function descriptions.

**Prompt Design:**

- The `SystemMessage` sent to the LLM helps guide its decision-making.
- Clearly describe each available function (pipeline) and when it should be used.

---

### **Considerations**

- **Consistency and Determinism**: While the LLM can understand and interpret user input flexibly, its outputs may vary. To enhance consistency:
  - Use deterministic LLM models or settings.
  - Provide explicit and clear instructions in the system prompt.
  - Test with a variety of inputs to refine prompts.

- **Error Handling**: Ensure that your agents handle situations where the LLM might not produce a valid `FunctionCall` or when the `ToolAgent` encounters errors.

- **Scalability**: You can easily add more pipelines by defining new tools and updating the system prompt accordingly.

- **Privacy and Compliance**: Be cautious with user data passed to the LLM and adhere to privacy regulations as well as OpenAI's policies.

---

### **Full Example Code**

Here's how all the pieces fit together in a concise example:

```python
import asyncio
from dataclasses import dataclass
from typing import List

# Custom message classes
@dataclass
class UserInput:
    text: str

@dataclass
class FinalResult:
    result: str

# Import Autogen components
from autogen_core import RoutedAgent, rpc, MessageContext, SingleThreadedAgentRuntime, CancellationToken
from autogen_core.tools import FunctionTool, Tool
from autogen_core.tool_agent import ToolAgent, tool_agent_caller_loop
from autogen_core.models import ChatCompletionClient, AssistantMessage, UserMessage, LLMMessage, SystemMessage

# Define pipeline functions
async def pipeline_a(text: str, cancellation_token: CancellationToken) -> str:
    result = f"{text} -> Pipeline A Step 1 -> Pipeline A Step 2"
    return result

async def pipeline_b(text: str, cancellation_token: CancellationToken) -> str:
    result = f"{text} -> Pipeline B Step 1 -> Pipeline B Step 2"
    return result

# Wrap functions as tools
pipeline_a_tool = FunctionTool(func=pipeline_a, description="Process data using Pipeline A")
pipeline_b_tool = FunctionTool(func=pipeline_b, description="Process data using Pipeline B")

# Set up the ToolAgent
tool_agent = ToolAgent(description="Pipeline Tool Agent", tools=[pipeline_a_tool, pipeline_b_tool])

# Implement ChatCompletionClient with OpenAI
# (Use the OpenAIChatCompletionClient implementation provided above)

# CoordinatorAgent using the LLM to decide the pipeline
class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        user_text = message.text

        input_messages: List[LLMMessage] = [
            SystemMessage(content="""
You are an assistant that decides which processing pipeline to use based on user input.
Available functions are:
- pipeline_a: Use this when the user mentions 'pipeline A' or wants detailed processing.
- pipeline_b: Use this for quick processing or when the user mentions 'pipeline B'.
"""),
            UserMessage(content=user_text, source="user")
        ]

        tool_agent_id = await self.runtime.get('tool_agent_type', key='pipeline_tool_agent')

        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[pipeline_a_tool.schema, pipeline_b_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant"
        )

        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    return FinalResult(result=msg.content)
        return FinalResult(result="Error: No valid output from LLM")

# Main function to run the example
async def main():
    runtime = SingleThreadedAgentRuntime()

    # Replace with your actual OpenAI API key
    model_client = OpenAIChatCompletionClient(model_name="gpt-4-0613", api_key="your-openai-api-key")

    coordinator_factory = lambda: CoordinatorAgent(model_client=model_client)
    await CoordinatorAgent.register(runtime=runtime, type="coordinator_agent_type", factory=coordinator_factory)

    tool_agent_factory = lambda: tool_agent
    await tool_agent.register(runtime=runtime, type="tool_agent_type", factory=tool_agent_factory)

    runtime.start()

    coordinator_agent_id = await runtime.get('coordinator_agent_type', key='coordinator_agent')
    user_input = UserInput(text="Please process this data using Pipeline B.")
    final_result = await runtime.send_message(
        message=user_input,
        recipient=coordinator_agent_id
    )

    print(final_result.result)

    await runtime.stop()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
```

---

### **Key Takeaways**

- **LLM Integration**: Leveraging the LLM for decision-making enhances flexibility in interpreting user inputs.
- **Function Calling Paradigm**: Utilizing function calling enables structured interactions between the LLM and your agents.
- **Modular Design**: Separating concerns into different agents and tools promotes maintainability and scalability.
- **Prompt Engineering**: Effective prompts are crucial for guiding the LLM's behavior consistently.

---

By integrating the LLM into your pipeline selection process, you offload the decision-making to the model, making your system more dynamic and adaptable to complex user inputs. This approach aligns well with Autogen's capabilities, enabling you to build sophisticated agent-based applications.


# Q: Great, thank you! Please write a proposed architecture (file structure) of the project which has this LLM-decided Router logic in the beginning of the processing of the user input + deterministic pipelines (returning pandas dataframe as dict + <small description dict>, for example) + LLM decider in the middle (NOT consuming the whole dataframe to save tokens, only to consume <small description dict> to decide which pipeline to run next) + Deterministic pipeline in the end, which should run some pandas analysis of the dataframe generating <another small dict with overview info> and write LLM-analysis in the end.

# A:

Certainly! Below is a proposed architecture for your project that incorporates:

- **LLM-decided Router logic** at the beginning to process user input.
- **Deterministic pipelines** that return pandas DataFrames and small description dictionaries.
- An **LLM decider in the middle**, which consumes only the small description dict to decide the next pipeline.
- A final **deterministic pipeline**, performing pandas analysis and generating another small dict with overview info.
- A final **LLM analysis**, which generates the end report based on the overview info.

---

### **Project File Structure**

```
project_root/
├── main.py
├── config.py
├── agents/
│   ├── __init__.py
│   ├── coordinator_agent.py
│   ├── pipeline_a_agent.py
│   ├── pipeline_b_agent.py
│   ├── middle_decider_agent.py
│   ├── final_pipeline_agent.py
│   └── analysis_agent.py
├── tools/
│   ├── __init__.py
│   └── function_tools.py
├── models/
│   ├── __init__.py
│   └── openai_client.py
├── utils/
│   ├── __init__.py
│   └── data_utils.py
└── requirements.txt
```

---

### **Explanation of Components**

#### **1. main.py**

- Entry point of the application.
- Sets up the agent runtime and registers agents.
- Handles user input and initiates the processing pipeline.

#### **2. config.py**

- Contains configuration settings like API keys, model names, and other constants.

#### **3. agents/**

- **coordinator_agent.py**: Receives user input and uses an LLM to decide the initial pipeline (Router logic).
- **pipeline_a_agent.py & pipeline_b_agent.py**: Deterministic pipelines processing data and returning a pandas DataFrame along with a small description dict.
- **middle_decider_agent.py**: Uses an LLM to decide the next pipeline based on the small description dict from previous pipelines.
- **final_pipeline_agent.py**: Performs deterministic pandas analysis on the DataFrame and generates an overview info dict.
- **analysis_agent.py**: Uses an LLM to produce a final analysis report based on the overview info dict.

#### **4. tools/function_tools.py**

- Defines functions (wrapped as tools) that represent the pipelines, using `FunctionTool`.

#### **5. models/openai_client.py**

- Implements the `ChatCompletionClient` interface using OpenAI's API for interactions with the LLM.

#### **6. utils/data_utils.py**

- Contains utility functions for data processing, DataFrame handling, and dict conversion.

#### **7. requirements.txt**

- Lists required Python packages (e.g., `autogen-core`, `pandas`, `openai`).

---

### **Implementation Details**

#### **main.py**

```python
import asyncio
from config import OPENAI_API_KEY, MODEL_NAME
from autogen_core import SingleThreadedAgentRuntime
from models.openai_client import OpenAIChatCompletionClient
from agents.coordinator_agent import CoordinatorAgent
from agents.middle_decider_agent import MiddleDeciderAgent
from agents.analysis_agent import AnalysisAgent
from agents.pipeline_a_agent import PipelineAAgent
from agents.pipeline_b_agent import PipelineBAgent
from agents.final_pipeline_agent import FinalPipelineAgent
from autogen_core.tool_agent import ToolAgent
from tools.function_tools import pipeline_a_tool, pipeline_b_tool, final_pipeline_tool
from agents.common import UserInput

async def main():
    runtime = SingleThreadedAgentRuntime()
    model_client = OpenAIChatCompletionClient(model_name=MODEL_NAME, api_key=OPENAI_API_KEY)

    # Register agents
    await CoordinatorAgent.register(runtime=runtime, type="coordinator_agent_type", factory=lambda: CoordinatorAgent(model_client))
    await MiddleDeciderAgent.register(runtime=runtime, type="middle_decider_agent_type", factory=lambda: MiddleDeciderAgent(model_client))
    await AnalysisAgent.register(runtime=runtime, type="analysis_agent_type", factory=lambda: AnalysisAgent(model_client))

    # Register deterministic pipeline agents
    await PipelineAAgent.register(runtime=runtime, type="pipeline_a_agent_type", factory=PipelineAAgent)
    await PipelineBAgent.register(runtime=runtime, type="pipeline_b_agent_type", factory=PipelineBAgent)
    await FinalPipelineAgent.register(runtime=runtime, type="final_pipeline_agent_type", factory=FinalPipelineAgent)

    # Set up ToolAgent with pipeline tools
    pipeline_tools = [pipeline_a_tool, pipeline_b_tool]
    tool_agent = ToolAgent(description="Pipeline Tool Agent", tools=pipeline_tools)
    await tool_agent.register(runtime=runtime, type="tool_agent_type", factory=lambda: tool_agent)

    runtime.start()

    # Simulate user input and initiate processing
    coordinator_agent_id = await runtime.get('coordinator_agent_type', key='coordinator_agent')
    user_input_text = input("Enter your request: ")
    user_input = UserInput(text=user_input_text)

    final_result = await runtime.send_message(
        message=user_input,
        recipient=coordinator_agent_id
    )

    print("Final Analysis Report:")
    print(final_result.result)

    await runtime.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

#### **config.py**

```python
OPENAI_API_KEY = 'your-openai-api-key'
MODEL_NAME = 'gpt-4-0613'
```

---

#### **agents/common.py**

```python
from dataclasses import dataclass

@dataclass
class UserInput:
    text: str

@dataclass
class FinalResult:
    result: str
```

---

#### **agents/coordinator_agent.py**

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage
from autogen_core.tools import ToolSchema
from autogen_core.tool_agent import tool_agent_caller_loop
from agents.common import UserInput, FinalResult
from typing import List
from tools.function_tools import pipeline_a_tool, pipeline_b_tool

class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        user_text = message.text

        input_messages: List[LLMMessage] = [
            SystemMessage(content="""You are an assistant that decides which initial processing function to call based on user input.
            Available functions are pipeline_a and pipeline_b."""),
            UserMessage(content=user_text, source="user")
        ]

        tool_agent_id = await self.runtime.get('tool_agent_type', key='tool_agent')

        # Use the caller loop to decide initial pipeline
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[pipeline_a_tool.schema, pipeline_b_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant"
        )

        # Extract result data
        last_message_content = None
        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
                last_message_content = msg.content
                break

        if last_message_content:
            # Deserialize the result (assuming JSON format)
            import json
            result_data = json.loads(last_message_content)
            description_dict = result_data['description_dict']

            # Proceed to the middle decider agent
            middle_decider_agent_id = await self.runtime.get('middle_decider_agent_type', key='middle_decider_agent')
            decision_info = await self.send_message(
                message=description_dict,
                recipient=middle_decider_agent_id,
                cancellation_token=ctx.cancellation_token
            )

            # Proceed to final pipeline
            final_pipeline_agent_id = await self.runtime.get('final_pipeline_agent_type', key='final_pipeline_agent')
            final_result = await self.send_message(
                message={'dataframe': result_data['dataframe'], 'info': decision_info},
                recipient=final_pipeline_agent_id,
                cancellation_token=ctx.cancellation_token
            )

            return final_result

        return FinalResult(result="Error: Unable to process input.")
```

---

#### **agents/pipeline_a_agent.py**

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from utils.data_utils import process_data_a
from typing import Dict

class PipelineAAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Pipeline A Agent")

    @rpc
    async def run_pipeline_a(self, message: Dict, ctx: MessageContext) -> Dict:
        data = message['data']
        dataframe, description_dict = process_data_a(data)
        result = {
            'dataframe': dataframe.to_dict(),
            'description_dict': description_dict
        }
        return result
```

**Note**: `PipelineBAgent` would be similar but using `process_data_b`.

---

#### **agents/middle_decider_agent.py**

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from agents.common import FinalResult
from typing import List
from tools.function_tools import final_pipeline_tool

class MiddleDeciderAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Middle Decider Agent")
        self.model_client = model_client

    @rpc
    async def decide_next_step(self, message: Dict, ctx: MessageContext) -> Dict:
        description_dict = message  # Small description dict

        input_messages: List[LLMMessage] = [
            SystemMessage(content="""Based on the provided data description, decide the next processing function to call.
            Available function: final_pipeline."""),
            UserMessage(content=str(description_dict), source="user")
        ]

        tool_agent_id = await self.runtime.get('tool_agent_type', key='tool_agent')

        # Use the caller loop to decide the next pipeline
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[final_pipeline_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant"
        )

        # Extract decision info
        last_message_content = None
        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
                last_message_content = msg.content
                break

        if last_message_content:
            import json
            decision_info = json.loads(last_message_content)
            return decision_info

        return {}
```

---

#### **agents/final_pipeline_agent.py**

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from pandas import DataFrame
from agents.common import FinalResult
from utils.data_utils import analyze_data

class FinalPipelineAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Final Pipeline Agent")

    @rpc
    async def run_final_pipeline(self, message: Dict, ctx: MessageContext) -> FinalResult:
        dataframe_dict = message['dataframe']
        dataframe = DataFrame.from_dict(dataframe_dict)
        info = message.get('info', {})

        overview_info = analyze_data(dataframe)

        # Proceed to the analysis agent
        analysis_agent_id = await self.runtime.get('analysis_agent_type', key='analysis_agent')
        analysis_result = await self.send_message(
            message=overview_info,
            recipient=analysis_agent_id,
            cancellation_token=ctx.cancellation_token
        )

        return FinalResult(result=analysis_result)
```

---

#### **agents/analysis_agent.py**

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage
from agents.common import FinalResult
from typing import List

class AnalysisAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Analysis Agent")
        self.model_client = model_client

    @rpc
    async def generate_report(self, message: Dict, ctx: MessageContext) -> str:
        overview_info = message  # The small overview dict

        input_messages: List[LLMMessage] = [
            SystemMessage(content="""Generate a detailed analysis report based on the following overview info."""),
            UserMessage(content=str(overview_info), source="user")
        ]

        response = await self.model_client.create(
            messages=input_messages,
            cancellation_token=ctx.cancellation_token
        )

        if isinstance(response.content, str):
            return response.content

        return "Error: Failed to generate report."
```

---

#### **tools/function_tools.py**

```python
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from utils.data_utils import process_data_a, process_data_b
from typing import Dict

async def pipeline_a(data: str, cancellation_token: CancellationToken) -> Dict:
    dataframe, description_dict = process_data_a(data)
    return {
        'dataframe': dataframe.to_dict(),
        'description_dict': description_dict
    }

async def pipeline_b(data: str, cancellation_token: CancellationToken) -> Dict:
    dataframe, description_dict = process_data_b(data)
    return {
        'dataframe': dataframe.to_dict(),
        'description_dict': description_dict
    }

async def final_pipeline(dataframe_dict: Dict, cancellation_token: CancellationToken) -> Dict:
    # Placeholder function representing the final pipeline
    return {'status': 'Final pipeline executed'}

# Wrap functions as tools
pipeline_a_tool = FunctionTool(func=pipeline_a, description="Process data using Pipeline A")
pipeline_b_tool = FunctionTool(func=pipeline_b, description="Process data using Pipeline B")
final_pipeline_tool = FunctionTool(func=final_pipeline, description="Execute the final data processing pipeline")
```

---

#### **models/openai_client.py**

```python
import openai
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, RequestUsage
from autogen_core import CancellationToken
from typing import List, Mapping, Any

class OpenAIChatCompletionClient(ChatCompletionClient):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        openai.api_key = api_key

    async def create(
        self,
        messages: List[LLMMessage],
        tools: List[Any] = [],
        json_output: bool = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken = None,
    ) -> CreateResult:
        # Implementation similar to the one provided earlier
        # Ensure that you handle the messages and function calls appropriately
        pass

    # Implement other methods as required by the ChatCompletionClient interface
```

---

#### **utils/data_utils.py**

```python
import pandas as pd
from typing import Tuple, Dict

def process_data_a(data: str) -> Tuple[pd.DataFrame, Dict]:
    # Deterministic processing logic for Pipeline A
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    description_dict = {'summary': 'Pipeline A processed the data.'}
    return df, description_dict

def process_data_b(data: str) -> Tuple[pd.DataFrame, Dict]:
    # Deterministic processing logic for Pipeline B
    df = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
    description_dict = {'summary': 'Pipeline B processed the data.'}
    return df, description_dict

def analyze_data(dataframe: pd.DataFrame) -> Dict:
    # Deterministic analysis logic
    overview_info = {'mean': dataframe.mean().to_dict(), 'sum': dataframe.sum().to_dict()}
    return overview_info
```

---

### **Key Takeaways**

- **Modular Architecture**: Separating agents, tools, models, and utility functions enhances maintainability and scalability.
- **LLM for Decision-Making**: The LLM decides which pipeline to execute at both the beginning and middle stages, based on small description dicts.
- **Deterministic Data Processing**: All data manipulation and processing are deterministic, ensuring consistent and predictable results.
- **Token Efficiency**: By only passing small description dicts to the LLM, you minimize token usage and optimize performance.
- **Asynchronous Operations**: Utilizing `async`/`await` and `asyncio` ensures non-blocking operations and better resource utilization.

---

### **Final Notes**

- **Error Handling**: Implement robust error handling in your agents to manage unforeseen issues gracefully.
- **Data Serialization**: Ensure that DataFrames are properly serialized (e.g., using `df.to_dict()`) when sending between agents.
- **Testing**: Thoroughly test each component individually and then as part of the whole pipeline to ensure correct functionality.
- **Security and Privacy**: Be cautious with the data sent to the LLM, ensuring compliance with data privacy regulations and OpenAI's policies.

By following this architecture, your project will have a clear structure that aligns with your specified requirements, enabling efficient collaboration between deterministic processing and LLM-driven decision-making.

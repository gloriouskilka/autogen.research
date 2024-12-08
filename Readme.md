# Q: I have these tools I'd like to run via Autogen. What is the best approach? To create an Agent with tool calls and to run agent? To wrap Agent somehow? Please write clear code examples how to implement such Agent which should run functions and return result without LLM-post processing (Deterministic DB query and analysis via message passing, IIUC)
# A: import asyncio
from typing import Dict
from pydantic import BaseModel
from autogen_core import (
    RoutedAgent,
    message_handler,
    MessageContext,
    AgentId,
    SingleThreadedAgentRuntime,
    CancellationToken,
    AgentRuntime,
)

# Define your function outputs
class PipelineResult(BaseModel):
    dataframe: Dict
    description_dict: Dict

class OverviewInfo(BaseModel):
    overview: Dict

# Assume these are your processing functions
async def pipeline_a(data: str, cancellation_token: CancellationToken = None) -> PipelineResult:
    dataframe, description_dict = process_data_pipeline_a(data)
    return PipelineResult(dataframe=dataframe.to_dict(), description_dict=description_dict)

async def pipeline_b(data: str, cancellation_token: CancellationToken = None) -> PipelineResult:
    dataframe, description_dict = process_data_pipeline_b(data)
    return PipelineResult(dataframe=dataframe.to_dict(), description_dict=description_dict)

async def final_pipeline(dataframe: Dict, info: Dict, cancellation_token: CancellationToken = None) -> OverviewInfo:
    overview_info = analyze_full_data(dataframe)
    return OverviewInfo(overview=overview_info)

# Define message types for your agent
class PipelineARequest(BaseModel):
    data: str

class PipelineBRequest(BaseModel):
    data: str

class FinalPipelineRequest(BaseModel):
    dataframe: Dict
    info: Dict

# Create the agent that handles these messages
class DataPipelineAgent(RoutedAgent):
    def __init__(self, description: str = "Data Pipeline Agent"):
        super().__init__(description)

    @message_handler
    async def handle_pipeline_a(self, message: PipelineARequest, ctx: MessageContext):
        result = await pipeline_a(message.data, cancellation_token=ctx.cancellation_token)
        return result

    @message_handler
    async def handle_pipeline_b(self, message: PipelineBRequest, ctx: MessageContext):
        result = await pipeline_b(message.data, cancellation_token=ctx.cancellation_token)
        return result

    @message_handler
    async def handle_final_pipeline(self, message: FinalPipelineRequest, ctx: MessageContext):
        result = await final_pipeline(message.dataframe, message.info, cancellation_token=ctx.cancellation_token)
        return result

# Set up the runtime and interact with the agent
async def main():
    # Create the data pipeline agent
    data_pipeline_agent = DataPipelineAgent()

    # Create the runtime
    runtime = SingleThreadedAgentRuntime()
    # Register the agent
    await data_pipeline_agent.register(runtime, type="data_pipeline_agent", factory=lambda: data_pipeline_agent)
    agent_id = AgentId(type="data_pipeline_agent", key="default")

    # Send messages to the agent
    request_a = PipelineARequest(data="some input data")
    result_a = await runtime.send_message(
        message=request_a,
        recipient=agent_id
    )
    print(f"Pipeline A result: {result_a}")

    request_b = PipelineBRequest(data="some other input data")
    result_b = await runtime.send_message(
        message=request_b,
        recipient=agent_id
    )
    print(f"Pipeline B result: {result_b}")

    # For the final pipeline
    request_final = FinalPipelineRequest(
        dataframe=result_a.dataframe,  # Assuming result_a is of type PipelineResult
        info=result_a.description_dict
    )
    result_final = await runtime.send_message(
        message=request_final,
        recipient=agent_id
    )
    print(f"Final Pipeline result: {result_final}")

# Run the main function
asyncio.run(main())
 


# Q: Please help to fix sending the message to tool_agent_id in coordinator_agent.py. I think it is not correct code, please help to fix
# A: Certainly! The issue you're facing with sending the message to `tool_agent_id` in your `coordinator_agent.py` is likely due to inconsistencies in registering and retrieving the `ToolAgent`. Specifically, it's related to how you're specifying the `key` during agent registration and retrieval.

---

### **Problem Explanation**

In your `main.py`, when registering the `ToolAgent`, you **did not** specify a `key`. By default, if no `key` is provided during registration, it uses `'default'`.

```python
# main.py
# Register the ToolAgent without specifying a key
tool_agent_factory = lambda: tool_agent
await tool_agent.register(runtime=runtime, type="tool_agent_type", factory=tool_agent_factory)
```

However, in your `coordinator_agent.py`, when you attempt to retrieve the `ToolAgent`, you **do specify** a `key`:

```python
# coordinator_agent.py
# Retrieve the ToolAgent using a key
tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")
```

Since you didn't register the `ToolAgent` with `key="tool_agent"`, retrieving it with this key will fail because the `AgentRuntime` can't find an agent with the type `'tool_agent_type'` and the key `'tool_agent'`.

---

### **Solution**

To fix this issue, you need to ensure consistency in the `key` used during both registration and retrieval of agents. You have two options:

---

#### **Option 1: Specify a Key During Registration and Retrieval**

**Update `main.py` to include a `key` when registering the `ToolAgent`:**

```python
# main.py
# Register the ToolAgent with key="tool_agent"
tool_agent_factory = lambda: tool_agent
await tool_agent.register(runtime=runtime, type="tool_agent_type", key="tool_agent", factory=tool_agent_factory)
```

**Update `coordinator_agent.py` to retrieve the `ToolAgent` using the same `key`:**

```python
# coordinator_agent.py
# Retrieve the ToolAgent using key="tool_agent"
tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")
```

This ensures that the `AgentRuntime` can match the agent's type and key during retrieval, successfully returning the correct `tool_agent_id`.

---

#### **Option 2: Use the Default Key in Both Registration and Retrieval**

Since the default `key` is `'default'` when none is specified, you can register and retrieve the agent without specifying the `key`.

**Ensure `main.py` registers the `ToolAgent` without a key (already done):**

```python
# main.py
# Register the ToolAgent without specifying a key
tool_agent_factory = lambda: tool_agent
await tool_agent.register(runtime=runtime, type="tool_agent_type", factory=tool_agent_factory)
```

**Update `coordinator_agent.py` to retrieve the `ToolAgent` without specifying a `key`:**

```python
# coordinator_agent.py
# Retrieve the ToolAgent using the default key
tool_agent_id = await self.runtime.get("tool_agent_type")
```

---

### **Recommendation**

For clarity and to avoid potential conflicts, I recommend **Option 1**, explicitly specifying a `key` during both registration and retrieval. This approach is especially useful if you have multiple agents of the same type but with different keys.

---

### **Corrected Code**

#### **main.py**

```python
# main.py
import asyncio
from autogen_core import SingleThreadedAgentRuntime
from models.openai_client import OpenAIChatCompletionClient
from agents.coordinator_agent import CoordinatorAgent
from agents.middle_decider_agent import MiddleDeciderAgent
from agents.analysis_agent import AnalysisAgent
from agents.final_pipeline_agent import FinalPipelineAgent
from autogen_core.tool_agent import ToolAgent
from tools.function_tools import pipeline_a_tool, pipeline_b_tool, final_pipeline_tool
from agents.common import UserInput

async def main():
    runtime = SingleThreadedAgentRuntime()
    model_client = OpenAIChatCompletionClient(model_name=MODEL_NAME, api_key=OPENAI_API_KEY)

    # Register agents with specific keys
    await CoordinatorAgent.register(
        runtime=runtime,
        type="coordinator_agent_type",
        key="coordinator_agent",
        factory=lambda: CoordinatorAgent(model_client)
    )
    await MiddleDeciderAgent.register(
        runtime=runtime,
        type="middle_decider_agent_type",
        key="middle_decider_agent",
        factory=lambda: MiddleDeciderAgent(model_client)
    )
    await AnalysisAgent.register(
        runtime=runtime,
        type="analysis_agent_type",
        key="analysis_agent",
        factory=lambda: AnalysisAgent(model_client)
    )
    await FinalPipelineAgent.register(
        runtime=runtime,
        type="final_pipeline_agent_type",
        key="final_pipeline_agent",
        factory=FinalPipelineAgent
    )

    # Set up ToolAgent with pipeline tools, registering with key="tool_agent"
    pipeline_tools = [pipeline_a_tool, pipeline_b_tool, final_pipeline_tool]
    tool_agent = ToolAgent(description="Pipeline Tool Agent", tools=pipeline_tools)
    await tool_agent.register(
        runtime=runtime,
        type="tool_agent_type",
        key="tool_agent",
        factory=lambda: tool_agent
    )

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

#### **coordinator_agent.py**

```python
# coordinator_agent.py

from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop

from agents.common import UserInput, FinalResult, PipelineResult, DecisionInfo, FinalPipelineInput
from typing import List
from tools.function_tools import pipeline_a_tool, pipeline_b_tool
import json

class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        user_text = message.text

        input_messages: List[LLMMessage] = [
            SystemMessage(
                content="""
You are an assistant that decides which initial processing function to call based on user input.
Available functions are pipeline_a and pipeline_b.
"""
            ),
            UserMessage(content=user_text, source="user"),
        ]

        # Retrieve the ToolAgent using key="tool_agent"
        tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")

        # Use the caller loop to decide initial pipeline
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[pipeline_a_tool.schema, pipeline_b_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant",
        )

        # Extract result data
        last_message_content = None
        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    last_message_content = msg.content
                    break
                elif isinstance(msg.content, list):
                    continue  # Skip function calls

        if last_message_content:
            # Deserialize the result (assuming JSON format)
            result_data = json.loads(last_message_content)
            pipeline_result = PipelineResult(
                dataframe=result_data["dataframe"],
                description_dict=result_data["description_dict"]
            )

            # Proceed to the middle decider agent
            middle_decider_agent_id = await self.runtime.get("middle_decider_agent_type", key="middle_decider_agent")
            decision_info = await self.send_message(
                message=pipeline_result.description_dict,
                recipient=middle_decider_agent_id,
                cancellation_token=ctx.cancellation_token
            )

            # Proceed to final pipeline
            final_pipeline_agent_id = await self.runtime.get("final_pipeline_agent_type", key="final_pipeline_agent")
            final_input = FinalPipelineInput(
                dataframe=pipeline_result.dataframe,
                info=decision_info.info
            )

            final_result = await self.send_message(
                message=final_input,
                recipient=final_pipeline_agent_id,
                cancellation_token=ctx.cancellation_token
            )

            return FinalResult(result=final_result.result)

        return FinalResult(result="Error: Unable to process input.")
```

---

### **Explanation**

- **Registration with Key:**
  - In `main.py`, when you register the `ToolAgent`, you provide `key="tool_agent"`.
  - This ensures the agent is registered under the type `'tool_agent_type'` and key `'tool_agent'`.
- **Retrieval with Key:**
  - In `coordinator_agent.py`, when retrieving `tool_agent_id`, you specify the same key using `key="tool_agent"`.
  - This matches the registered agent, allowing `self.runtime.get` to successfully find and return the correct `tool_agent_id`.

**Note:** Using keys helps differentiate between multiple agents of the same type. If you have multiple agents registered under the same type but different keys, specifying the key ensures you're interacting with the intended agent instance.

---

### **Additional Check**

Ensure that all other agents are also registered and retrieved consistently. For example, if you register other agents with specific keys, make sure you use the same keys when retrieving them.

---

### **Summary**

By ensuring consistency in the use of `key` during both agent registration and retrieval, you resolve the issue of sending the message to `tool_agent_id`. This guarantees that `self.runtime.get` retrieves the correct agent instance, and the subsequent calls to `tool_agent_caller_loop` function as intended.

---

Feel free to let me know if you have any further questions or need additional assistance!
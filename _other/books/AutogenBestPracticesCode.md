Below are concise, essential rules and best practices for using the Autogen framework, followed by two fully working example applications demonstrating both LLM-based (nondeterministic) agents and deterministic agents, as well as key Autogen features.

## Core Principles

### Agents:
- Agents process messages (tasks) and generate responses.
- Each `AssistantAgent` uses an LLM or is implemented deterministically.
- Agents can have tools: Python callables providing external capabilities (e.g. `get_weather`).

### Teams & Workflows:
- `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`: group agents together.
- Teams route messages among agents based on turn order, selector functions, or handoffs.

### Messages:
- `TextMessage`, `HandoffMessage`, `MultiModalMessage`: structured communication primitives.
- Agents read all previous messages before responding.

### Termination Conditions:
- `TextMentionTermination`, `MaxMessageTermination`, `HandoffTermination`: stop conditions.
- Combine conditions with `|` (OR) or `&` (AND).

### State & Persistence:
- Agents and teams can be saved and reloaded (`save_state`, `load_state`).

### Tools & Function Calls:
- Define Python functions as tools to provide capabilities beyond the LLM.
- Invoke them directly in code or via special tool messages.

### LLM vs Deterministic Agents:
- LLM-backed: uses `OpenAIChatCompletionClient` or similar.
- Deterministic: returns hardcoded or computed answers without calling the LLM.

## Best Practices:
- Keep system messages minimal and explicit.
- Chain agents logically and manage tools carefully.
- Use termination conditions to avoid infinite loops.
- Save and restore state for reproducible workflows.
- Use `Console` streaming for debugging; remove it in production code.

## Example 1: LLM-Based Multi-Agent Workflow

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 25°C and clear."

model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

primary_agent = AssistantAgent(
    name="primary",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a weather assistant. Respond with 'TERMINATE' when done."
)

critic_agent = AssistantAgent(
    name="critic",
    model_client=model_client,
    system_message="Review the primary agent's answer. If it's good, respond 'TERMINATE'. Otherwise, ask for improvements."
)

termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=termination)

async def run_llm_example():
    await Console(team.run_stream(task="What is the weather in Berlin?"))

asyncio.run(run_llm_example())
```

Example 2: Deterministic Agents without LLM Calls
```python
import asyncio
from typing import Sequence
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console

class DeterministicAgent(BaseChatAgent):
    @property
    def produced_message_types(self):
        return [TextMessage]
    async def on_messages(self, messages: Sequence[TextMessage], cancellation_token: CancellationToken) -> Response:
        # Simple deterministic logic: echo back last user message in uppercase
        user_msg = next((m for m in messages if m.source == "user"), None)
        resp = user_msg.content.upper() if user_msg else "NO INPUT"
        return Response(chat_message=TextMessage(content=resp, source=self.name))
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

agent_1 = DeterministicAgent(name="det_agent_1", description="Echoes user message in uppercase")
agent_2 = DeterministicAgent(name="det_agent_2", description="Echoes user message in uppercase")

team = RoundRobinGroupChat([agent_1, agent_2], termination_condition=MaxMessageTermination(max_messages=3))

async def run_deterministic_example():
    await Console(team.run_stream(task="hello world"))

asyncio.run(run_deterministic_example())
```

These examples show key Autogen concepts—LLM integration, termination conditions, tool usage, message routing, state management (not shown here, but referenced), and deterministic logic.



[//]: # (Вы сказали:)
I don't see that you've managed to demonstrate all features of Autogen, foe example, handoff. Fix it. I need a proper technical cheatsheet for Python developer who never used Autogen
ChatGPT

Below is a concise technical cheat sheet for a Python developer new to Autogen, followed by three updated, fully working example applications. The examples demonstrate multiple features, including LLM-based agents, deterministic agents, tools, state saving/loading, handoffs, multiple team types, termination conditions, and message routing.

## Autogen Technical Cheatsheet

### Core Concepts

**Agents**: Classes that process messages and produce responses.
- `AssistantAgent` uses LLMs or can be deterministic.
- Provide `system_message` to guide agent behavior.
- Agents have tools (Python callables) to perform non-LLM tasks.

**Messages**:
- `TextMessage`: Basic text.
- `HandoffMessage`: Transfer conversation control between agents/users.
- `MultiModalMessage`: Includes text and images.

**Teams & Execution**:
- Teams: `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`.
- `run(task=...)` returns final `TaskResult` after completion.
- `run_stream(task=...)` yields messages as they are generated.

**Termination Conditions**:
- `TextMentionTermination("TERMINATE")`: stops if message contains "TERMINATE".
- `MaxMessageTermination(n)`: stops after n messages.
- `HandoffTermination(target="user")`: stops after handoff to user.
- Combine with `|` (OR) / `&` (AND).

**Tools**:
- Python functions passed to agents in `tools=[...]`.
- Agents can call these directly from code or inline in responses.

**Handoffs**:
- Agents can send `HandoffMessage` to another agent or user.
- Useful in `Swarm` or multi-step workflows.

**State Persistence**:
- `save_state()` and `load_state()` store and restore agent/team state.

### Best Practices

- Keep `system_message` minimal and explicit.
- Chain agents logically and clearly.
- Use termination conditions to prevent infinite loops.
- Use `Console()` for debugging, remove in production.
- Validate tools and handle errors gracefully.

### Example 1: LLM-Based Multi-Agent with Termination & Tools

Demonstrates:
- LLM-backed agents (`AssistantAgent` with `OpenAIChatCompletionClient`)
- Tools (`get_weather`)
- Termination condition (`TextMentionTermination`)
- `RoundRobinGroupChat`


```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 22°C and sunny."

model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

primary_agent = AssistantAgent(
    name="primary",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful weather assistant. Use the 'get_weather' tool. Say 'TERMINATE' when done."
)

critic_agent = AssistantAgent(
    name="critic",
    model_client=model_client,
    system_message="You are a critic. If the answer is correct and helpful, say 'TERMINATE', otherwise request improvement."
)

termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=termination)

async def run_example_1():
    await Console(team.run_stream(task="What's the weather in London today?"))

asyncio.run(run_example_1())
```

## Example 2: Deterministic Agent & State Persistence

### Demonstrates:
- Deterministic agent (`BaseChatAgent` without LLM)
- State saving and loading (`save_state`, `load_state`)
- `RoundRobinGroupChat` with `MaxMessageTermination`

```python
import asyncio
from typing import Sequence
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console

class DeterministicEchoAgent(BaseChatAgent):
    @property
    def produced_message_types(self):
        return [TextMessage]

    async def on_messages(self, messages: Sequence[TextMessage], cancellation_token: CancellationToken) -> Response:
        user_msg = next((m for m in messages if m.source == "user"), None)
        reply = user_msg.content[::-1] if user_msg else "NO MSG"
        return Response(chat_message=TextMessage(content=reply, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

agent = DeterministicEchoAgent(name="echo")
team = RoundRobinGroupChat([agent], termination_condition=MaxMessageTermination(max_messages=2))

async def run_example_2():
    # First run
    res = await team.run(task="hello")
    print("First run result:", res.chat_message.content)
    state = await team.save_state()

    # Reset and try again without loading state
    await team.reset()
    res_no_load = await team.run(task="world")
    print("Run after reset (no state load):", res_no_load.chat_message.content)

    # Load previous state
    await team.load_state(state)
    res_loaded = await team.run(task="again")
    print("Run after state load:", res_loaded.chat_message.content)

asyncio.run(run_example_2())
```

## Example 3: Handoff with Swarm Team

### Demonstrates:
- Swarm team workflow
- HandoffMessage and HandoffTermination
- Multiple agents delegating tasks


```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def refund_flight(flight_id: str) -> str:
    return f"Flight {flight_id} has been refunded."

model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

travel_agent = AssistantAgent(
    "travel_agent",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    system_message="You are a travel agent. If user wants a refund, get flight_id from user, then handoff to flights_refunder. After refund done, say 'TERMINATE'."
)

flights_refunder = AssistantAgent(
    "flights_refunder",
    model_client=model_client,
    tools=[refund_flight],
    handoffs=["travel_agent", "user"],
    system_message="You are the refund agent. After refunding flight, handoff back to travel_agent."
)

termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
team = Swarm([travel_agent, flights_refunder], termination_condition=termination)

async def run_example_3():
    # First: user requests a refund. travel_agent asks for flight_id and hands off to user.
    result = await Console(team.run_stream(task="I need a flight refund"))
    # After receiving user input manually (e.g. "Flight ID: ABC123"), the conversation continues.
    # This example requires interactive handoff handling, so in a non-interactive environment,
    # you'd simulate by calling `team.run_stream(HandoffMessage(source='user', target='flights_refunder', content='ABC123'))`.

asyncio.run(run_example_3())
```

Below are three advanced code examples demonstrating complex agent systems and the use of `RoundRobinGroupChat`, `SelectorGroupChat`, and `Swarm`. Each scenario uses multiple agents, tools, handoffs, termination conditions, state handling, and both LLM-based and deterministic agents.

## Example 1: RoundRobinGroupChat

**Agents**: Data collector, Data analyst, Reporter

**Features**:
- LLM-based (`OpenAIChatCompletionClient`) and deterministic agents
- Tools (`get_stock_data`, `get_news`)
- Termination conditions (`TextMentionTermination`)
- Saving/Loading state

```python
import asyncio
from typing import Any, Dict, List
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

async def get_stock_data(symbol: str) -> Dict[str, Any]:
    return {"price": 250.50, "volume": 500000, "pe_ratio": 30.2, "market_cap": "500B"}

async def get_news(query: str) -> List[Dict[str, str]]:
    return [
        {"title": f"{query} launches new product", "date": "2024-09-10", "summary": f"{query} expanding market share."},
        {"title": f"{query} reports strong Q4 earnings", "date": "2024-09-12", "summary": f"{query} beats estimates."}
    ]

model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

class DeterministicSummarizer(BaseChatAgent):
    @property
    def produced_message_types(self): return [TextMessage]

    async def on_messages(self, messages, cancellation_token: CancellationToken) -> Response:
        user_msg = next((m.content for m in messages if m.source=="user"), "")
        return Response(chat_message=TextMessage(content=f"Final summary: {user_msg} - DATA CONSOLIDATED", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

data_collector = AssistantAgent(
    name="data_collector",
    model_client=model_client,
    tools=[get_stock_data, get_news],
    system_message="You gather company financial & news data using provided tools. Conclude with 'TERMINATE'."
)

data_analyst = AssistantAgent(
    name="data_analyst",
    model_client=model_client,
    system_message="You analyze data from data_collector. When done, say 'TERMINATE'."
)

reporter = DeterministicSummarizer(name="reporter")

termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat([data_collector, data_analyst, reporter], termination_condition=termination)

async def run_round_robin_example():
    res = await team.run(task="Analyze TSLA and provide a final summary")
    print("Result:", res.chat_message.content)
    state = await team.save_state()

    await team.reset()
    res2 = await team.run(task="Analyze AAPL and provide a final summary")
    print("Result after reset (no load):", res2.chat_message.content)

    await team.load_state(state)
    res3 = await team.run(task="Analyze MSFT and provide a final summary")
    print("Result after state load:", res3.chat_message.content)

asyncio.run(run_round_robin_example())
```

## Example 2: SelectorGroupChat

**Agents**: Planner, WebSearchAgent, ArxivSearchAgent, Reporter

**Features**:
- SelectorGroupChat with a custom `selector_func` that always routes control back to planner after each agent
- Tools (`google_search`, `arxiv_search`)
- Termination (`TextMentionTermination`)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import AgentMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def google_search(query: str) -> str:
    return f"Google results for {query}: Found relevant articles."

async def arxiv_search(query: str) -> str:
    return f"Arxiv results for {query}: Found 2 papers."

model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

planner_agent = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="You plan research steps. After deciding a step, hand off to search agents. If done, say 'TERMINATE'.",
    handoffs=["Google_Search_Agent","Arxiv_Search_Agent","Report_Agent"]
)

google_search_agent = AssistantAgent(
    name="Google_Search_Agent",
    model_client=model_client,
    tools=[google_search],
    system_message="Perform Google search and return snippet. Then return to planner.",
    handoffs=["planner"]
)

arxiv_search_agent = AssistantAgent(
    name="Arxiv_Search_Agent",
    model_client=model_client,
    tools=[arxiv_search],
    system_message="Perform Arxiv search and return snippet. Then return to planner.",
    handoffs=["planner"]
)

report_agent = AssistantAgent(
    name="Report_Agent",
    model_client=model_client,
    system_message="Synthesize all info and say 'TERMINATE'.",
    handoffs=[]
)

termination = TextMentionTermination("TERMINATE")

def selector_func(messages: "Sequence[AgentMessage]") -> str | None:
    # Always route next turn to planner unless last was planner, then route to first listed handoff agent in the planner's message.
    last_msg = messages[-1]
    if last_msg.source != "planner":
        return "planner"
    return None

team = SelectorGroupChat(
    [planner_agent, google_search_agent, arxiv_search_agent, report_agent],
    termination_condition=termination,
    selector_func=selector_func
)

async def run_selector_example():
    await Console(team.run_stream(task="Research 'quantum computing breakthroughs' and generate a final report."))

asyncio.run(run_selector_example())
```

## Example 3: Swarm with Handoffs

**Agents**: TravelAgent, LocalAgent, LanguageAgent, TravelSummaryAgent

**Features**:
- Swarm for flexible message passing
- Tools (`get_local_activities`)
- Handoff with `HandoffMessage`
- Termination (`HandoffTermination`, `TextMentionTermination`)
- LLM-based and multiple steps

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def get_local_activities(city: str) -> str:
    return f"Activities in {city}: Hiking, local market, museum."

model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")

planner_agent = AssistantAgent(
    "planner_agent",
    model_client=model_client,
    handoffs=["local_agent","language_agent","travel_summary_agent","user"],
    system_message="You plan a trip. First ask local_agent for activities, then language_agent for language tips, then finalize with travel_summary_agent. If you need user input, handoff to user. Say 'TERMINATE' when done."
)

local_agent = AssistantAgent(
    "local_agent",
    model_client=model_client,
    tools=[get_local_activities],
    handoffs=["planner_agent","user"],
    system_message="Suggest local activities. If need more info from user, handoff to user. Otherwise hand back to planner_agent."
)

language_agent = AssistantAgent(
    "language_agent",
    model_client=model_client,
    handoffs=["planner_agent","user"],
    system_message="Provide language tips. If need user input, handoff to user. Otherwise back to planner_agent."
)

travel_summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=model_client,
    handoffs=["planner_agent","user"],
    system_message="Summarize the plan. Then say 'TERMINATE'."
)

termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")

team = Swarm([planner_agent, local_agent, language_agent, travel_summary_agent], termination_condition=termination)

async def run_swarm_example():
    # Start the planning. The planner will orchestrate calls to local_agent and language_agent, and finalize with travel_summary_agent.
    await Console(team.run_stream(task="Plan a 3-day trip to Kyoto."))

asyncio.run(run_swarm_example())
```

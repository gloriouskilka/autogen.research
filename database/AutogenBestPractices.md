Below are essential rules and best practices for working with the Autogen framework, assuming your team is proficient in Python but new to Autogen. The framework simplifies building multi-agent conversational applications, where agents (LLMs) cooperate or criticize each other, execute tools, and produce final results. Key features include Agents, Tools, Teams (group chats, swarms), Termination Conditions, and State Management.

## Core Concepts

### Agents

- Represent AI participants (e.g. “assistant\_agent”) that respond to messages.
- Defined by `AssistantAgent` (LLM-driven), `UserProxyAgent` (simulate user), or custom classes.
- Each agent has a `system\_message` setting its role and behavior.
- Agents use tools (Python functions) to perform tasks like web searches, data analysis, or querying APIs.

Example:
```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 72 degrees and sunny."

weather_agent = AssistantAgent(
    name="weather_agent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    tools=[get_weather],
    system_message="You are a weather expert."
)
```

## Tools

Python callables that agents can invoke.
Used for actions beyond pure text generation (e.g. queries, math ops).
Return structured data for the agent to incorporate.

Example:
    
```python
async def search_web(query: str) -> str:
    return "Mocked search result for: " + query
```

## Messages

Agents produce `TextMessage` or `MultiModalMessage`.
Agents communicate by passing messages in a conversation loop.
`HandoffMessage` can transfer control between agents or user.

Example:
```python

from autogen_agentchat.messages import TextMessage

user_msg = TextMessage(content="What is the weather?", source="user")
```

## Teams (Group Chats)

Combine multiple agents into a conversation.

- **RoundRobinGroupChat**: Agents take turns responding.
- **SelectorGroupChat**: A selector decides which agent responds next.
- **Swarm**: Multiple agents collaborate or delegate tasks.

### Example:
```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

termination = TextMentionTermination("TERMINATE")
team = RoundRobinGroupChat([weather_agent], termination_condition=termination)
```
Run the team:
```python
result = await team.run(task="What is the weather in New York?")
```

## Termination Conditions

Define when a conversation ends (e.g. max messages, a keyword "TERMINATE", or a handoff event).
Combine conditions with logical operators (|, &).

### Example:
```python
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

max_termination = MaxMessageTermination(max_messages=5)
keyword_termination = TextMentionTermination("TERMINATE")
combined = max_termination | keyword_termination
```

## State Management

Save/load agent or team states to resume conversations. This is useful for long-running sessions, storing context, or caching.

### Example:
```python
state = await weather_agent.save_state()
# ... Later ...
await weather_agent.load_state(state)
```

## Core Concepts

### Agents

- Represent AI participants (e.g. `assistant_agent`) that respond to messages.
- Defined by `AssistantAgent` (LLM-driven), `UserProxyAgent` (simulate user), or custom classes.
- Each agent has a `system_message` setting its role and behavior.
- Agents use tools (Python functions) to perform tasks like web searches, data analysis, or querying APIs.

Example:
```python
from autogen_agentchat.base import Handoff
lazy_agent = AssistantAgent(
    "lazy_assistant",
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    handoffs=[Handoff(target="user", message="Transfer to user.")]
)
```

## Tool Usage Best Practices

- Keep tools deterministic and side-effect-free if possible.
- Validate inputs and handle exceptions in tools.
- Return concise data for easy LLM integration.

## Conversation Flow Best Practices

- Use a `system_message` to constrain agent behavior.
- Introduce termination conditions to prevent infinite loops.
- Test scenarios with `run_stream` for real-time output.
- Start small with a single agent and scale up to multiple agents as needed.

## Testing and Debugging

Use Python standard testing frameworks (e.g. `pytest`) to test your tools and agent logic.
Store states and replay conversations to debug agent responses.
Adjust model parameters (temperature, max_tokens) to fine-tune behavior.

### Example Workflow

1. Define Agents (one that fetches stock data, one that analyzes it, one that writes a final report).
2. Combine them into a `Swarm` or `RoundRobinGroupChat`.
3. Set termination conditions to stop at the right time.
4. Run `team.run_stream(task="Analyze AAPL stock")` and observe messages.
5. If needed, introduce a `HandoffMessage` to ask the user for additional input.
6. Once done, store the team state for future resumes.

This should give your team a solid understanding of the Autogen framework’s rules, features, and best practices.

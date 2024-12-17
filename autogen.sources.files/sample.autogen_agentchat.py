"""
CheatSheet.py

This cheat sheet provides a comprehensive guide and best practices on how to use the Autogen framework for building agent-based applications with LLMs (Large Language Models).

Audience: Python developers and professionals looking to quickly understand and implement the Autogen framework in their applications.

Note: This script is intended to be executed in a Python environment where the Autogen framework and its dependencies are installed.
"""

import asyncio
from typing import Any

# Import necessary components from the Autogen framework
from autogen_core import CancellationToken, Image

# from autogen_core.models import OpenAIChatCompletionClient, FunctionCall
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_agentchat.agents import (
    AssistantAgent,
    UserProxyAgent,
    CodeExecutorAgent,
    SocietyOfMindAgent,
)
from autogen_agentchat.messages import TextMessage, HandoffMessage
from autogen_agentchat.teams import (
    RoundRobinGroupChat,
    SelectorGroupChat,
    Swarm,
    MagenticOneGroupChat,
)
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    HandoffTermination,
)
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken


# Section 1: Setting up the Model Client
# ---------------------------------------

# The model client is used by agents to generate responses.
# Here, we use OpenAI's GPT-4 model as an example.
# Replace 'your_openai_api_key' with your actual OpenAI API key.

model_client = OpenAIChatCompletionClient(
    model="gpt-4",
    api_key="your_openai_api_key",
)


# Section 2: Creating Assistant Agents
# -------------------------------------

# AssistantAgent is a general-purpose agent that can assist with tasks and use tools if provided.


async def create_assistant_agent():
    # Define tools that the agent can use.
    # Tools can be async or sync functions.

    async def get_current_time() -> str:
        from datetime import datetime

        return f"The current time is {datetime.now()}"

    # Create the assistant agent with the model client and tools.
    assistant_agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[get_current_time],  # List of tools the agent can use
    )

    # Example usage: the assistant agent responds to a message.
    response = await assistant_agent.on_messages(
        [TextMessage(content="What is the current time?", source="user")], CancellationToken()
    )
    print(response.chat_message.content)


# Section 3: Using Tools in Agents
# ---------------------------------

# Agents can use tools to perform actions or calculations.
# Define custom tools and add them to your agent.


async def create_agent_with_tools():
    # Define custom tools

    async def add_numbers(a: int, b: int) -> int:
        return a + b

    async def multiply_numbers(a: int, b: int) -> int:
        return a * b

    # Create the assistant agent with tools
    math_agent = AssistantAgent(
        name="math_agent",
        model_client=model_client,
        tools=[add_numbers, multiply_numbers],
        system_message="You are a math assistant that can add and multiply numbers.",
    )

    # The agent can now use these tools to answer questions
    response = await math_agent.on_messages(
        [TextMessage(content="What is 5 plus 7?", source="user")], CancellationToken()
    )
    print(response.chat_message.content)


# Section 4: Creating a User Proxy Agent
# ---------------------------------------

# UserProxyAgent represents a human user in conversations.
# It can be used to simulate user input or to involve a human in the loop.


async def user_proxy_interaction():
    # Create a UserProxyAgent
    user_agent = UserProxyAgent(name="user_proxy")

    # Simulate a conversation where the user is asked a question

    response = await user_agent.on_messages(
        [TextMessage(content="What is your favorite color?", source="assistant")], CancellationToken()
    )
    print(f"User's response: {response.chat_message.content}")


# Section 5: Creating a Code Executor Agent
# ------------------------------------------

# CodeExecutorAgent executes code snippets found in received messages.
# It is recommended to use a Docker container to execute code safely.


async def create_code_executor_agent():
    # Set up the code executor using Docker
    code_executor = DockerCommandLineCodeExecutor(work_dir="coding_workspace")
    await code_executor.start()  # Start the Docker container

    # Create the CodeExecutorAgent
    code_agent = CodeExecutorAgent(name="code_executor", code_executor=code_executor)

    # Agent will attempt to execute code from messages
    task = TextMessage(
        content="""
        Here is some code to print 'Hello World':
        ```python
        print('Hello World')
        ```
        """,
        source="user",
    )

    response = await code_agent.on_messages([task], CancellationToken())
    print(response.chat_message.content)

    await code_executor.stop()  # Stop the Docker container when done


# Section 6: Creating Teams of Agents
# ------------------------------------

# Agents can be organized into teams to collaborate on tasks.
# Teams use different strategies to manage agent interactions.


# Example: RoundRobinGroupChat
async def round_robin_team():
    # Create multiple assistant agents
    agent1 = AssistantAgent(name="Alice", model_client=model_client)
    agent2 = AssistantAgent(name="Bob", model_client=model_client)

    # Define a termination condition
    termination_condition = MaxMessageTermination(max_messages=6)

    # Create a RoundRobinGroupChat team
    team = RoundRobinGroupChat(participants=[agent1, agent2], termination_condition=termination_condition)

    # Run the team with an initial task
    await Console(
        team.run_stream(
            task="Tell a collaborative story, each person continues the story.", cancellation_token=CancellationToken()
        )
    )


# Section 7: Using SelectorGroupChat Team
# ----------------------------------------

# SelectorGroupChat uses a model to select the next speaker based on conversation context.


async def selector_group_chat():
    agent1 = AssistantAgent(name="Agent1", model_client=model_client, description="Expert in technology.")

    agent2 = AssistantAgent(name="Agent2", model_client=model_client, description="Expert in healthcare.")

    termination = TextMentionTermination(text="END_CONVERSATION")

    team = SelectorGroupChat(
        participants=[agent1, agent2], model_client=model_client, termination_condition=termination
    )

    # Start the conversation
    await Console(
        team.run_stream(
            task="Discuss the impact of AI in different industries.", cancellation_token=CancellationToken()
        )
    )


# Section 8: Using Swarm Team for Dynamic Handoffs
# -------------------------------------------------

# Swarm allows dynamic handoff between agents based on HandoffMessage.


async def swarm_example():
    agent1 = AssistantAgent(
        name="SupportBot",
        model_client=model_client,
        handoffs=["HumanAgent"],
        system_message="You are a support bot that helps customers and hands off to a human if necessary.",
    )

    agent2 = UserProxyAgent(name="HumanAgent")

    termination = HandoffTermination(target="HumanAgent") | MaxMessageTermination(10)

    team = Swarm(participants=[agent1, agent2], termination_condition=termination)

    # First run, until handoff occurs
    await Console(
        team.run_stream(task="I need help with my account, it's locked.", cancellation_token=CancellationToken())
    )

    # Resume conversation after handoff
    await Console(
        team.run_stream(
            task=HandoffMessage(
                content="Hello, I'm a human agent. How can I assist you further?",
                source="HumanAgent",
                target="SupportBot",
            ),
            cancellation_token=CancellationToken(),
        )
    )


# Section 9: Using MagenticOneGroupChat for Complex Orchestration
# ---------------------------------------------------------------

# MagenticOneGroupChat orchestrates complex interactions among agents.


async def magentic_one_group_chat():
    agent1 = AssistantAgent(
        name="Researcher", model_client=model_client, system_message="You specialize in data research."
    )

    agent2 = AssistantAgent(
        name="Analyst", model_client=model_client, system_message="You analyze data and derive insights."
    )

    termination = MaxMessageTermination(max_messages=10)

    team = MagenticOneGroupChat(
        participants=[agent1, agent2], model_client=model_client, termination_condition=termination
    )

    # Start the team conversation
    await Console(
        team.run_stream(
            task="Analyze the trends in renewable energy adoption over the past decade.",
            cancellation_token=CancellationToken(),
        )
    )


# Section 10: Best Practices and Tips
# ------------------------------------

# 1. Always set a termination condition to prevent infinite loops in conversations.
# 2. Use handoffs carefully to manage flow between agents and humans.
# 3. When using tools, ensure they are well-documented and handle exceptions.
# 4. For complex interactions, consider using the MagenticOneGroupChat for better orchestration.
# 5. Use the Console utility to render conversations in the terminal for debugging and presentation.

# Section 11: Running the Examples
# ---------------------------------

# Uncomment the asyncio.run() calls below to run the desired example.
# Note that you should run one example at a time.

# Example 1: Create and interact with an assistant agent
# asyncio.run(create_assistant_agent())

# Example 2: Create an agent with tools
# asyncio.run(create_agent_with_tools())

# Example 3: User proxy interaction
# asyncio.run(user_proxy_interaction())

# Example 4: Code executor agent
# asyncio.run(create_code_executor_agent())

# Example 5: Round robin team of agents
# asyncio.run(round_robin_team())

# Example 6: Selector group chat team
# asyncio.run(selector_group_chat())

# Example 7: Swarm team with dynamic handoff
# asyncio.run(swarm_example())

# Example 8: MagenticOneGroupChat for complex orchestration
# asyncio.run(magentic_one_group_chat())

"""
Note: Replace 'your_openai_api_key' with your actual OpenAI API key before running these examples.
Ensure that you have the necessary permissions and that your usage complies with the OpenAI policies.
"""


# Section 12: Conclusion
# -----------------------

# This cheat sheet provided an overview of the Autogen framework's capabilities.
# By leveraging agents, tools, and teams, you can build sophisticated conversational AI applications.
# Customize and extend the provided examples to suit your application's needs.

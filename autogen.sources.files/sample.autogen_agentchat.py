"""
CheatSheet.py

This cheat sheet provides a comprehensive guide and best practices on how to use the Autogen framework for building agent-based applications with LLMs (Large Language Models).

Audience: Python developers and professionals looking to quickly understand and implement the Autogen framework in their applications.

Note: This script is intended to be executed in a Python environment where the Autogen framework and its dependencies are installed.
"""

import asyncio
from typing import Any

from autogen_agentchat.base import Response

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


"""
Continuing CheatSheet.py

In this continuation, we delve into building complex agent-based systems using the Autogen framework.
We focus on how to design agents that use tools to query and process data (e.g., SQL databases) without
unnecessarily passing large amounts of data through the LLM, thereby saving tokens and stabilizing the solution.

Audience: Python developers and professionals looking to implement efficient and scalable agent systems using Autogen.
"""

# Section 13: Complex Agentic Systems with Tool Chaining
# -------------------------------------------------------

# In production environments, it's crucial to design agent systems that are both efficient and cost-effective.
# One common scenario involves querying a database and processing the results, but we want to avoid passing
# large datasets through the LLM unnecessarily.

# Best Practices:
# 1. Use tools to query data and process it outside the LLM when possible.
# 2. Pass only necessary information to the LLM for tasks like summarization or generating user-friendly output.
# 3. Chain tools by having one tool's output be directly used by another tool.
# 4. Avoid sending large data (e.g., entire DataFrames) through the LLM.

# Example: Agent System to Query SQL Data and Process It Without Passing Data Through LLM

import asyncio
import pandas as pd
import sqlite3
from typing import Any, Dict, List

from autogen_core import CancellationToken
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console

# Setup: Create a sample SQLite database for the example


def create_sample_db(db_name: str):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sales(
            id INTEGER PRIMARY KEY,
            product TEXT,
            quantity INTEGER,
            price REAL
        )
    """
    )

    # Insert sample data
    cursor.executemany(
        """
        INSERT INTO sales(product, quantity, price) VALUES (?, ?, ?)
    """,
        [
            ("Widget A", 100, 2.50),
            ("Widget B", 150, 3.75),
            ("Widget C", 200, 1.25),
        ],
    )

    conn.commit()
    conn.close()


# Create the sample database
create_sample_db("sales.db")

# Section 13.1: Defining Tools for Data Query and Processing
# -----------------------------------------------------------


# Tool 1: Query the SQL database
def query_sales_data(product_name: str = None) -> Dict[str, Any]:
    """
    Query sales data from the database.
    If product_name is provided, filter by product.
    Returns data as a list of dictionaries.
    """
    conn = sqlite3.connect("sales.db")
    cursor = conn.cursor()

    if product_name:
        cursor.execute("SELECT * FROM sales WHERE product = ?", (product_name,))
    else:
        cursor.execute("SELECT * FROM sales")

    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    # Convert rows to list of dicts
    data = [dict(zip(columns, row)) for row in rows]

    conn.close()
    return data  # Return data directly without passing through LLM


# Tool 2: Process the data (e.g., calculate total sales)
def calculate_total_sales(data: List[Dict[str, Any]]) -> float:
    """
    Calculate the total sales value from the data.
    Data is a list of dictionaries with keys: 'quantity' and 'price'.
    """
    total_sales = sum(item["quantity"] * item["price"] for item in data)
    return total_sales


# Tool 3: Format the result for output
def format_sales_report(total_sales: float) -> str:
    """
    Format the total sales into a user-friendly string.
    """
    return f"The total sales amount is ${total_sales:.2f}"


# Section 13.2: Creating the Assistant Agent with Chained Tools
# --------------------------------------------------------------

# We will create an assistant agent that uses the tools defined above.
# The agent should perform the following steps:
# 1. Query data from the database.
# 2. Process the data to calculate total sales.
# 3. Format the result for the user.


async def run_agent_with_chained_tools():
    # Define the tools using FunctionTool
    query_tool = FunctionTool(
        query_sales_data,
        name="query_sales_data",
        description="Queries sales data from the database. Optionally filters by product name.",
    )

    process_tool = FunctionTool(
        calculate_total_sales, name="calculate_total_sales", description="Calculates the total sales from the data."
    )

    format_tool = FunctionTool(
        format_sales_report,
        name="format_sales_report",
        description="Formats the total sales amount into a user-friendly string.",
    )

    # Create the assistant agent
    agent = AssistantAgent(
        name="SalesAssistant",
        model_client=model_client,
        tools=[query_tool, process_tool, format_tool],
        system_message="""
        You are a sales assistant who can query sales data, calculate total sales, 
        and format the results for the user. Use the available tools to accomplish
        these tasks without passing large amounts of data through yourself.
        """,
        # Set reflect_on_tool_use to False to prevent LLM from seeing tool outputs
        reflect_on_tool_use=False,
        # Customize tool call summary to avoid passing data through LLM
        tool_call_summary_format="{result}",
    )

    # User message requesting total sales
    user_message = TextMessage(content="What is the total sales amount?", source="user")

    response = await agent.on_messages([user_message], CancellationToken())

    # Output the final response
    print(f"Assistant: {response.chat_message.content}")


# Run the example
# asyncio.run(run_agent_with_chained_tools())

# Explanation:
# - The assistant agent uses tools to perform data querying and processing.
# - By setting reflect_on_tool_use to False, the agent returns the tool outputs directly
#   without sending them back through the LLM.
# - The tool_call_summary_format is set to "{result}" to include only the result in the assistant's response.
# - The large data (the queried data) is used directly by the tools without being passed through the LLM.

# Section 13.3: Advanced Tool Chaining with Agent Functions
# -----------------------------------------------------------

# For more complex scenarios, you might need to chain tools and manage the flow of data explicitly.
# This can be achieved by designing the tools to accept outputs from other tools.

# As an example, we can modify the tools to accept input data as parameters.


# Tool 2 (modified): Process data accepts data as input
def calculate_total_sales_v2(data: Any) -> float:
    """
    Calculates total sales from input data.
    Expects data to be the output from query_sales_data.
    """
    # Data integrity check
    if not isinstance(data, list):
        raise ValueError("Data is not in the expected format.")
    return calculate_total_sales(data)


# Tool 3 (modified): Format result accepts total_sales as input
def format_sales_report_v2(total_sales: Any) -> str:
    """
    Formats the total sales amount.
    Expects total_sales to be the output from calculate_total_sales.
    """
    # Data integrity check
    if not isinstance(total_sales, (int, float)):
        raise ValueError("total_sales is not a number.")
    return format_sales_report(total_sales)


# We can now design the agent to use these tools in sequence.


async def run_advanced_agent_with_chained_tools():
    # Define the tools using FunctionTool, wrapping them to accept dynamic data
    query_tool = FunctionTool(
        query_sales_data,
        name="query_sales_data",
        description="Queries sales data from the database. Optionally filters by product name.",
    )

    process_tool = FunctionTool(
        calculate_total_sales_v2,
        name="calculate_total_sales",
        description="Calculates the total sales from the data returned by query_sales_data.",
    )

    format_tool = FunctionTool(
        format_sales_report_v2, name="format_sales_report", description="Formats the total sales amount."
    )

    # We create a custom agent to manage the tool chaining
    class AdvancedAssistantAgent(AssistantAgent):
        async def on_messages_stream(self, messages, cancellation_token):
            # Override to handle tool chaining
            # Step 1: Query data
            data = query_sales_data()
            # Step 2: Process data
            total_sales = calculate_total_sales_v2(data)
            # Step 3: Format result
            final_output = format_sales_report_v2(total_sales)
            # Create a response message
            response = TextMessage(content=final_output, source=self.name)
            yield Response(chat_message=response)

    agent = AdvancedAssistantAgent(
        name="AdvancedSalesAssistant",
        model_client=model_client,
        tools=[],  # Tools are used internally, not registered with the agent
        system_message="You are an advanced sales assistant.",
    )

    # User message
    user_message = TextMessage(content="Please provide the total sales amount.", source="user")

    response = await agent.on_messages([user_message], CancellationToken())
    print(f"Assistant: {response.chat_message.content}")


# Run the example
# asyncio.run(run_advanced_agent_with_chained_tools())

# In this example, the agent overrides the on_messages_stream method to manage the tool calling sequence internally.
# This approach provides fine-grained control over the tool chaining and ensures that data is not passed
# through the LLM at any point.

# Section 13.4: Passing Parameters Between Tools Without LLM Involvement
# ----------------------------------------------------------------------

# To further optimize, you can design tools to accept and return data in a way that allows direct chaining.

# Example: Tool Chaining with Direct Data Passing

# Tool definitions remain as before.


# Define a manager function that orchestrates the tools
def handle_total_sales_query(product_name: str = None) -> str:
    """
    Orchestrates the process of querying sales data, calculating total sales,
    and formatting the result.
    """
    data = query_sales_data(product_name)
    total_sales = calculate_total_sales(data)
    result = format_sales_report(total_sales)
    return result


# Register the manager function as a tool
manager_tool = FunctionTool(
    handle_total_sales_query,
    name="handle_total_sales_query",
    description="Handles the process of providing the total sales amount.",
)


async def run_agent_with_manager_tool():
    agent = AssistantAgent(
        name="SalesAssistantWithManager",
        model_client=model_client,
        tools=[manager_tool],
        system_message="""
        You can use the 'handle_total_sales_query' tool to get the total sales amount.
        """,
    )

    user_message = TextMessage(content="What are the total sales for all products?", source="user")

    response = await agent.on_messages([user_message], CancellationToken())
    print(f"Assistant: {response.chat_message.content}")


# Run the example
# asyncio.run(run_agent_with_manager_tool())

# By creating a manager tool that encapsulates the entire process, we minimize the interaction with the LLM
# and prevent large data transfers.

# Section 13.5: Best Practices for Token Efficiency and Stability
# ---------------------------------------------------------------

# - **Minimize LLM Involvement**: Use tools to handle data-intensive operations without involving the LLM.
# - **Use Tool Chaining**: Design tools to accept outputs from other tools, enabling direct data passing.

# - **Limit Reflective Processing**: Set `reflect_on_tool_use` to `False` in `AssistantAgent` to prevent the agent from sending tool outputs back to the LLM for further processing.
# - **Customize Tool Summaries**: Use `tool_call_summary_format` to control what is included in the assistant's response. Avoid including large data outputs.
# - **Use Deterministic Tools**: For critical processing, rely on deterministic tools rather than the LLM to reduce variability.
# - **Validate Data**: Implement data integrity checks in your tools to ensure correct data is being processed.
# - **Efficient Prompts**: Keep system prompts concise and focused to reduce token usage.

# Section 14: In-depth Low-level Details and Logic
# -------------------------------------------------

# The Autogen framework allows for fine-grained control over agent behaviors and interactions.
# Understanding how agents, tools, and messages work together is crucial.

# **Agent Message Flow**:
# - Agents receive messages and produce responses.
# - Messages can be of various types (e.g., `TextMessage`, `HandoffMessage`).

# **Tool Calls**:
# - When an agent needs to perform an action, it can make a tool call.
# - The agent's `on_messages` method handles incoming messages and may trigger tool calls.

# **Reflecting on Tool Use**:
# - By default, agents may send tool outputs back to the LLM to generate a response.
# - This can be controlled using the `reflect_on_tool_use` parameter.

# **Custom Agents**:
# - You can create custom agents by subclassing `AssistantAgent` or `BaseChatAgent`.
# - Overriding methods like `on_messages_stream` allows for custom behavior.

# **Data Passing Between Tools**:
# - Tools can accept and return data structures (e.g., lists, dicts).
# - By designing tools that accept outputs from other tools, you can chain operations without involving the LLM.

# **Example of Custom Agent Handling Data Internally**:


class CustomDataAgent(AssistantAgent):
    async def on_messages(self, messages, cancellation_token):
        # Custom logic to handle messages and tool interactions
        user_query = messages[0].content
        # Determine action based on the user's message
        if "total sales" in user_query.lower():
            data = query_sales_data()
            total_sales = calculate_total_sales(data)
            result = format_sales_report(total_sales)
            response_message = TextMessage(content=result, source=self.name)
            return Response(chat_message=response_message)
        else:
            # Fallback to default behavior
            return await super().on_messages(messages, cancellation_token)


# **Tips**:
# - When designing agents, always consider the flow of data and how to minimize unnecessary LLM interactions.
# - Use agent states (`save_state`, `load_state`) to manage long-term conversations or sessions.
# - Implement error handling within tools to ensure robustness.

# Section 15: Additional Examples and Use Cases
# ----------------------------------------------

# Example: Agent that Analyzes Data and Generates a Report Without LLM


def analyze_sales_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes sales data to compute various metrics.
    Returns a dictionary with analysis results.
    """
    total_sales = calculate_total_sales(data)
    total_items_sold = sum(item["quantity"] for item in data)
    average_price = total_sales / total_items_sold if total_items_sold != 0 else 0
    return {"total_sales": total_sales, "total_items_sold": total_items_sold, "average_price": average_price}


def format_analysis_report(analysis_results: Dict[str, Any]) -> str:
    """
    Formats the analysis results into a readable report.
    """
    return (
        f"Sales Analysis Report:\n"
        f"- Total Sales: ${analysis_results['total_sales']:.2f}\n"
        f"- Total Items Sold: {analysis_results['total_items_sold']}\n"
        f"- Average Price per Item: ${analysis_results['average_price']:.2f}"
    )


def generate_sales_report(product_name: str = None) -> str:
    """
    Orchestrates the data retrieval, analysis, and report generation.
    """
    data = query_sales_data(product_name)
    analysis_results = analyze_sales_data(data)
    report = format_analysis_report(analysis_results)
    return report


# Register the report generation as a tool
report_tool = FunctionTool(
    generate_sales_report, name="generate_sales_report", description="Generates a detailed sales analysis report."
)


async def run_agent_with_report_tool():
    agent = AssistantAgent(
        name="SalesReportAgent",
        model_client=model_client,
        tools=[report_tool],
        system_message="""
        You can use the 'generate_sales_report' tool to create a sales analysis report.
        """,
    )

    user_message = TextMessage(content="Can you provide a sales analysis report?", source="user")

    response = await agent.on_messages([user_message], CancellationToken())
    print(f"Assistant: {response.chat_message.content}")


# Run the example
# asyncio.run(run_agent_with_report_tool())

# Section 16: Conclusion
# -----------------------

# By carefully designing your agents and tools, you can create efficient, cost-effective, and stable agentic systems.
# The key is to manage the flow of data such that large datasets are handled by tools directly, minimizing the LLM's involvement to only where it's truly beneficial.

# Remember to test your agents thoroughly and monitor token usage to ensure that your solution scales well in production environments.

# This extended cheat sheet adds detailed examples and best practices for creating complex agent systems using the Autogen framework. It addresses how to efficiently manage tool interactions, reduce unnecessary LLM usage, and chain tools for data processing without passing large amounts of data through the language model. By implementing these strategies, developers can build robust and scalable applications that are both efficient and cost-effective.

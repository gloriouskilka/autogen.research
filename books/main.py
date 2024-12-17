# About the framework used:
# AutoGen 0.4 is an open-source framework for building AI agent systems, enabling the creation of event-driven, distributed, scalable, and resilient agentic applications.
# GITHUB
#
# Key Components:
#
# Agents: Core entities that perform tasks, process messages, and interact with other agents. Agents can be equipped with tools and are responsible for handling specific functions within the system.
#
# Agent Runtime: The execution environment managing agent lifecycles and facilitating communication between agents. It ensures agents operate seamlessly, supporting both local and distributed setups.
#
# Messages: The primary means of communication between agents, enabling asynchronous interactions and coordination.
#
# Tools: Functions or utilities that agents can utilize to perform specific actions, such as executing code or retrieving data. Tools enhance agent capabilities by providing additional functionalities.
#
# Model Clients: Interfaces that allow agents to interact with Large Language Models (LLMs) for tasks like generating text or processing natural language inputs. These clients enable agents to leverage LLMs effectively.
#
# LLM Integration:
#
# Agents utilize LLMs through model clients, which process inputs such as function names, parameter names, and tool descriptions. This integration enables agents to generate appropriate responses, make decisions, and perform tasks based on natural language inputs.
#
# Combining Agents:
#
# Chats: Agents can be organized into teams that communicate via asynchronous messaging, supporting both event-driven and request/response interaction patterns. This setup allows agents to collaborate on tasks, share information, and coordinate actions.
#
# Message Passing: Agents interact by sending and receiving messages, enabling complex workflows and decision-making processes. The framework supports both direct messaging (similar to RPC) and broadcasting to topics (pub-sub), facilitating flexible communication strategies.
#
# Example Workflow:
#
# Agent Registration: Agents are registered with the runtime, specifying their types and associated factory functions for instantiation.
#
# Message Handling: Agents define handlers for specific message types, enabling them to process incoming messages and perform corresponding actions.
#
# Tool Usage: Agents can be equipped with tools, allowing them to execute functions or access external services as needed.
#
# LLM Interaction: Through model clients, agents can process natural language inputs, generate responses, and make informed decisions based on LLM outputs.
#
# Communication: Agents communicate by sending messages to each other, coordinating tasks, sharing information, and collaborating to achieve system objectives.
#
# This architecture allows developers to build sophisticated AI systems where agents work together, leveraging LLMs and tools to perform complex tasks efficiently.


import asyncio
import inspect
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

import loguru
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from loguru import logger

from pydantic_settings import BaseSettings, SettingsConfigDict

import sqlalchemy as sa
from setuptools.command.saveopts import saveopts
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Date, Text
from sqlalchemy.sql import text

from autogen_core.tools import FunctionTool

from util.util import model_client, settings, configure_tracing, OpenAIChatCompletionClientWrapper

# ---------------------------
# Database Setup
# ---------------------------

DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Create session factory
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Import the text function
from sqlalchemy.sql import text


# Initialize the database
async def init_db():
    # Read SQL scripts from files
    with open("books\\01_create.sql", "r") as f:
        create_table_sql = f.read()
    with open("books\\02_insert.sql", "r") as f:
        insert_data_sql = f.read()

    # Use the async engine to execute SQL statements
    async with engine.begin() as conn:
        # Split the create_table_sql into individual statements and execute them
        for stmt in create_table_sql.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                await conn.execute(text(stmt))
        # Split the insert_data_sql into individual statements and execute them
        for stmt in insert_data_sql.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                await conn.execute(text(stmt))

    # Optionally, fetch and display data to verify
    async with AsyncSessionLocal() as session:
        logger.debug("BOOKS Table:")
        result = await session.execute(text("SELECT * FROM BOOKS"))
        books_rows = result.fetchall()
        for row in books_rows:
            logger.debug(row)
        logger.debug("\nINVENTORY Table:")
        result = await session.execute(text("SELECT * FROM INVENTORY"))
        inventory_rows = result.fetchall()
        for row in inventory_rows:
            logger.debug(row)


# ---------------------------
# Tools
# ---------------------------


# Tool to execute SQL queries loaded from files
async def execute_sql_query(query_name: str) -> List[Dict[str, Any]]:
    """
    Execute an SQL query loaded from a file and return the results.
    """
    # Map of available query files
    query_files = {
        "excess_inventory": "03_query_01_excess.sql",
        "obsolete_inventory": "03_query_02_obsolete.sql",
        "top_selling_books": "03_query_03_top_selling.sql",
        "least_selling_books": "03_query_04_least_selling.sql",
        "inventory_turnover": "03_query_05_turnover.sql",
        "stock_aging": "03_query_06_stock_aging.sql",
        "forecast_demand": "03_query_07_forecast_demand.sql",
        "supplier_performance": "03_query_08_supplier_performance.sql",
        "seasonal_trends": "03_query_09_seasonal_trends.sql",
        "profit_margins": "03_query_10_profit_margins.sql",
        "warehouse_optimization": "03_query_11_warehouse_space_optimize.sql",
        "return_rates": "03_query_12_return_reasons.sql",
    }

    if query_name not in query_files:
        raise ValueError(f"Query '{query_name}' not found. Available queries: {list(query_files.keys())}")

    query_file = query_files[query_name]

    # Read the SQL query from the file
    with open("books\\" + query_file, "r") as f:
        query_sql = f.read()

    # Execute the SQL query
    async with AsyncSessionLocal() as session:
        result = await session.execute(text(query_sql))
        rows = result.fetchall()

        # Get column names from the result cursor
        column_names = result.keys()

        # Convert rows to list of dictionaries
        data = [dict(zip(column_names, row)) for row in rows]

    return data


# ---------------------------
# Agents
# ---------------------------

# Analysis Agent
analysis_agent = AssistantAgent(
    name="AnalysisAgent",
    model_client=model_client,
    tools=[execute_sql_query],
    system_message="""
You are the AnalysisAgent. Your role is to analyze the bookstore's database and provide recommendations to improve excess inventory, obsolescence, and other metrics.

- Use the `execute_sql_query` tool to run SQL queries loaded from files.
- The available queries are:

    - 'excess_inventory': Identify excess inventory.
    - 'obsolete_inventory': Identify obsolete inventory.
    - 'top_selling_books': Identify top-selling books.
    - 'least_selling_books': Identify least-selling books.
    - 'inventory_turnover': Calculate inventory turnover ratios.
    - 'stock_aging': Analyze stock aging.
    - 'forecast_demand': Forecast demand.
    - 'supplier_performance': Evaluate supplier performance.
    - 'seasonal_trends': Identify seasonal sales trends.
    - 'profit_margins': Monitor profit margins.
    - 'warehouse_optimization': Optimize warehouse space.
    - 'return_rates': Track return rates and reasons.

- After executing a query, analyze the results and provide actionable recommendations to the user.
- Do not include the SQL code or raw data in your responses. Focus on the analysis and recommendations.
- When you have completed your analysis, respond with 'TERMINATE' to end the conversation.
""",
)

# ---------------------------
# Termination Condition
# ---------------------------

termination_condition = TextMentionTermination("TERMINATE")

# ---------------------------
# Execution
# ---------------------------

#
# # Define a custom function to handle verification exceptions
# async def handle_verification(verification, expected_function_calls):
#     if isinstance(verification, OpenAIChatCompletionClientWrapper.FunctionCallVerification):
#         # Handle function call verification
#         actual_function_calls = []
#         for function_call_record in verification.function_calls:
#             function_name = function_call_record.function_name
#             arguments = function_call_record.arguments
#             logger.debug(f"Function called: {function_name} with arguments: {arguments}")
#             actual_function_calls.append({"name": function_name, "arguments": arguments})
#
#         # Compare actual function calls with expected function calls
#         if actual_function_calls == expected_function_calls:
#             logger.debug("Function calls match the expected function calls.")
#         else:
#             logger.debug("Function calls do not match the expected function calls.")
#             logger.debug("Expected:")
#             logger.debug(json.dumps(expected_function_calls, indent=2))
#             logger.debug("Actual:")
#             logger.debug(json.dumps(actual_function_calls, indent=2))
#     elif isinstance(verification, OpenAIChatCompletionClientWrapper.TextResultVerification):
#         # Handle text result verification
#         content = verification.content
#         logger.debug(f"Text content: {content}")
#         # Implement your business logic based on the text content
#     else:
#         # Handle unexpected verification types
#         raise Exception("Unknown verification type.")

#
# class TestGeneratorAgent(AssistantAgent):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.system_message = """
# You are a TestGeneratorAgent. Your role is to analyze another agent's system description and its tools, and generate testing tasks for it.
#
# Given an agent's description and tools, you should:
#
# - Read the agent's system message to understand its role and behavior.
# - Examine the tools the agent uses, including their names and parameters.
# - Generate a list of tasks that will effectively test the agent's functionality by invoking its tools with various parameters.
# - For each task, specify the expected function calls, including function name and arguments.
#
# Your output should be a Python list of dictionaries, where each dictionary contains:
#
# - 'task': A description of the task to be performed.
# - 'expected_function_calls': A list of dictionaries, each with 'name' and 'arguments' keys, representing the function calls that the agent is expected to make when performing the task.
#
# Output the list of tasks in valid Python code.
# """
#
#     async def generate_tasks(self, agent_to_test: BaseChatAgent) -> List[Dict[str, Any]]:
#         # Analyze the agent's description and tools to generate tasks
#         tasks = []
#
#         # Extract the tools and their parameter names
#         tools = agent_to_test.tools
#         # We'll assume that all tools require 'product_name' as a parameter based on your context
#         # But we'll actually extract the parameter names from the tool signatures for generality
#
#         # Generate a set of realistic product names for testing
#         product_names = ["Black Hat", "Yellow Hat", "Red Hat"]
#
#         for product_name in product_names:
#             # Describe the task
#             task_description = f"Retrieve sales data and customer feedback for {product_name}."
#             expected_function_calls = []
#             for tool in tools:
#                 function_name = tool.name
#                 # Extract parameter names from the tool's signature
#                 signature = inspect.signature(tool.func)
#                 parameters = signature.parameters
#
#                 arguments = {}
#                 for param in parameters.values():
#                     if param.name == "product_name":
#                         arguments["product_name"] = product_name
#                     else:
#                         arguments[param.name] = f"<{param.name}>"
#
#                 expected_function_calls.append({"name": function_name, "arguments": arguments})
#             tasks.append(
#                 {
#                     "task": task_description,
#                     "expected_function_calls": expected_function_calls,
#                 }
#             )
#
#         return tasks


async def main():
    # Initialize logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level="DEBUG",
    )

    # Initialize the database
    await init_db()

    # Set up the agent runtime
    runtime = SingleThreadedAgentRuntime()
    analysis_agent._runtime = runtime

    # Start the agent interaction
    logger.debug("\nWelcome to the Bookstore Inventory Analysis Tool!")
    logger.debug("You can ask the agent to analyze various aspects of your inventory.")
    logger.debug("Type 'TERMINATE' to end the conversation.\n")

    # while True:
    # task = input("Please enter your request: ")
    # if task.strip().upper() == "TERMINATE":
    #     logger.debug("Conversation terminated.")
    #     break

    task = "excess_inventory"

    # Run the agent and stream the output
    # await Console(analysis_agent.run_stream(task, termination_condition=termination_condition))
    result = await Console(analysis_agent.run_stream(task=task))
    i = 100


if __name__ == "__main__":
    asyncio.run(main())

# model_client.set_throw_on_create(True)
#
# saved_state = await team.save_state()
#
# # Run the tasks sequentially
# for task_entry in tasks:
#     task_text = task_entry["task"]
#     expected_function_calls = task_entry["expected_function_calls"]
#
#     logger.debug(f"--- Starting task: {task_text} ---\n")
#
#     try:
#         # Run the team and capture the output
#         await Console(team.run_stream(task=task_text))
#
#         # After running, check the verification results
#         for verification in model_client.create_results:
#             await handle_verification(verification)
#
#         # Clear the create_results after handling
#         model_client.create_results.clear()
#
#     except OpenAIChatCompletionClientWrapper.Verification as verification:
#         await handle_verification(verification, expected_function_calls)
#
#     logger.debug(f"\n--- Completed task: {task_text} ---\n")
#     # await team.reset()  # Reset the team state before each task
#     await team.load_state(saved_state)

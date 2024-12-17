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
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
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

from util import model_client, settings, configure_tracing, OpenAIChatCompletionClientWrapper

# ---------------------------
# Database Setup
# ---------------------------

DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# SQL string to create the tables
create_table_sql = """
-- Create the 'sales' table
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    color TEXT,
    size TEXT,
    quantity_sold INTEGER,
    date DATE
);

-- Create the 'customer_feedback' table
CREATE TABLE customer_feedback (
    id INTEGER PRIMARY KEY,
    product_name TEXT,
    feedback TEXT,
    date DATE
);
"""

# SQL string to insert data
insert_data_sql = """
-- Insert data into 'sales' table
INSERT INTO sales (id, product_name, color, size, quantity_sold, date) VALUES
    (1, 'Yellow Hat', 'Yellow', 'Large', 120, '2023-06-01'),
    (2, 'Yellow Hat', 'Yellow', 'Large', 80, '2023-07-01'),
    (3, 'Yellow Hat', 'Yellow', 'Large', 40, '2023-08-01'),
    (4, 'Yellow Hat', 'Yellow', 'Large', 20, '2023-09-01'),
    (5, 'Black Hat', 'Black', 'Medium', 50, '2023-06-01'),
    (6, 'Black Hat', 'Black', 'Medium', 60, '2023-07-01'),
    (7, 'Black Hat', 'Black', 'Medium', 70, '2023-08-01'),
    (8, 'Black Hat', 'Black', 'Medium', 80, '2023-09-01');

-- Insert data into 'customer_feedback' table
INSERT INTO customer_feedback (id, product_name, feedback, date) VALUES
    (1, 'Yellow Hat', 'The hat fades after washing.', '2023-07-15'),
    (2, 'Yellow Hat', 'Size runs too big.', '2023-08-10'),
    (3, 'Yellow Hat', 'Color is not as vibrant as pictured.', '2023-08-20'),
    (4, 'Yellow Hat', 'Uncomfortable to wear for long periods.', '2023-09-05'),
    (5, 'Black Hat', 'Great quality, very comfortable.', '2023-07-12'),
    (6, 'Black Hat', 'Stylish and fits well.', '2023-08-15'),
    (7, 'Black Hat', 'Perfect for everyday use.', '2023-09-08');
"""

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Create session factory
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Import the text function
from sqlalchemy.sql import text


# Initialize the database
async def init_db():
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
        print("Sales Table:")
        result = await session.execute(text("SELECT * FROM sales"))
        sales_rows = result.fetchall()
        for row in sales_rows:
            print(row)
        print("\nCustomer Feedback Table:")
        result = await session.execute(text("SELECT * FROM customer_feedback"))
        feedback_rows = result.fetchall()
        for row in feedback_rows:
            print(row)


# ---------------------------
# Tools
# ---------------------------


# Tool to get sales data
async def get_sales_data(product_name: str) -> Dict[str, Any]:
    """
    Retrieve sales data for a given product.
    """
    async with AsyncSessionLocal() as session:
        query = """
        SELECT date, quantity_sold
        FROM sales
        WHERE product_name = :product_name
        ORDER BY date
        """
        result = await session.execute(text(query), {"product_name": product_name})
        data = result.fetchall()
        return {
            "sales_data": [{"date": row.date.strftime("%Y-%m-%d"), "quantity_sold": row.quantity_sold} for row in data]
        }


# Tool to get customer feedback
async def get_customer_feedback(product_name: str) -> Dict[str, Any]:
    """
    Retrieve customer feedback for a given product.
    """
    async with AsyncSessionLocal() as session:
        query = """
        SELECT feedback, date
        FROM customer_feedback
        WHERE product_name = :product_name
        ORDER BY date
        """
        result = await session.execute(text(query), {"product_name": product_name})
        data = result.fetchall()
        return {
            "customer_feedback": [{"date": row.date.strftime("%Y-%m-%d"), "feedback": row.feedback} for row in data]
        }


# Tool to analyze data
async def analyze_data(product_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze sales and customer feedback data for a given product to provide insights.
    """
    analysis_results = {}

    # Analyze sales trends
    if "sales_data" in data:
        sales = data["sales_data"]
        quantities = [entry["quantity_sold"] for entry in sales]
        if len(quantities) >= 2:
            if quantities[-1] < quantities[0]:
                analysis_results["sales_trend"] = f"Sales of {product_name} have significantly decreased over time."
            elif quantities[-1] > quantities[0]:
                analysis_results["sales_trend"] = f"Sales of {product_name} have increased over time."
            else:
                analysis_results["sales_trend"] = f"Sales of {product_name} are stable."
        else:
            analysis_results["sales_trend"] = "Not enough data to determine sales trends."
    else:
        analysis_results["sales_trend"] = "Sales data not provided."

    # Analyze customer feedback
    if "customer_feedback" in data:
        feedback_list = data["customer_feedback"]
        negative_feedback = []
        for feedback in feedback_list:
            if any(
                keyword in feedback["feedback"].lower()
                for keyword in [
                    "fade",
                    "size",
                    "color",
                    "uncomfortable",
                    "not as pictured",
                    "runs too big",
                    "not",
                    "too",
                    "faded",
                    "issues",
                    "problem",
                    "dislike",
                    "unhappy",
                    "poor",
                    "bad",
                ]
            ):
                negative_feedback.append(feedback)
        if negative_feedback:
            analysis_results["customer_feedback_analysis"] = (
                f"Customer feedback for {product_name} indicates issues with product quality: {len(negative_feedback)} instances."
            )
        else:
            analysis_results["customer_feedback_analysis"] = (
                f"Customer feedback for {product_name} is generally positive."
            )
    else:
        analysis_results["customer_feedback_analysis"] = "Customer feedback data not provided."

    return analysis_results


# ---------------------------
# Agents
# ---------------------------

# Planner Agent
planner_agent = AssistantAgent(
    name="PlannerAgent",
    model_client=model_client,
    handoffs=["SQLQueryAgent", "DataAnalysisAgent", "ExplanationAgent"],
    system_message="""
You are the PlannerAgent. Your role is to coordinate the investigation to understand issues with any product.

- **Step 1**: Ask the SQLQueryAgent to retrieve sales data and customer feedback for the specified product.
- **Step 2**: When you receive data from SQLQueryAgent, include this data in the `HandoffMessage` content when sending it to DataAnalysisAgent. The content should be a JSON object containing `product_name` and `data`.
- **Step 3**: Request the DataAnalysisAgent to analyze the data.
- **Step 4**: Send the analysis results to ExplanationAgent for generating a comprehensive explanation.

Always include necessary data in your handoffs.
""",
)

# SQL Query Agent
sql_query_agent = AssistantAgent(
    name="SQLQueryAgent",
    model_client=model_client,
    tools=[get_sales_data, get_customer_feedback],
    handoffs=["PlannerAgent"],
    system_message="""
You are the SQLQueryAgent.

- Use the `get_sales_data` tool to retrieve sales data for the specified product.
- Use the `get_customer_feedback` tool to retrieve customer feedback for the specified product.
- Retrieve sales trends and customer feedback for the specified product.
- After execution, include the retrieved data in the HandoffMessage content and return it to PlannerAgent.
""",
)

# Data Analysis Agent
data_analysis_agent = AssistantAgent(
    name="DataAnalysisAgent",
    model_client=model_client,
    tools=[analyze_data],
    handoffs=["PlannerAgent"],
    system_message="""
You are the DataAnalysisAgent.

- Receive data from PlannerAgent via `HandoffMessage` content.
- The content will be a JSON object containing `product_name` and `data`.
- Before calling `analyze_data`, check if both `product_name` and `data` are present.
- If any are missing, inform `PlannerAgent` about the missing information.
- Use the `analyze_data` tool by providing `product_name` and `data` as arguments.
- Return the analysis results to PlannerAgent in a HandoffMessage.
""",
)

# Explanation Agent
explanation_agent = AssistantAgent(
    name="ExplanationAgent",
    model_client=model_client,
    handoffs=[],
    system_message="""
You are the ExplanationAgent.

- Generate a clear and comprehensive explanation based on the analysis results.
- Provide actionable insights and recommendations to improve the product's performance.
- Once done, reply directly to the user and mention 'TERMINATE' to end the conversation.
""",
)

# ---------------------------
# Termination Condition
# ---------------------------

termination_condition = TextMentionTermination("TERMINATE")

# ---------------------------
# Execution
# ---------------------------


# Define a custom function to handle verification exceptions
async def handle_verification(verification, expected_function_calls):
    if isinstance(verification, OpenAIChatCompletionClientWrapper.FunctionCallVerification):
        # Handle function call verification
        actual_function_calls = []
        for function_call_record in verification.function_calls:
            function_name = function_call_record.function_name
            arguments = function_call_record.arguments
            print(f"Function called: {function_name} with arguments: {arguments}")
            actual_function_calls.append({"name": function_name, "arguments": arguments})

        # Compare actual function calls with expected function calls
        if actual_function_calls == expected_function_calls:
            print("Function calls match the expected function calls.")
        else:
            print("Function calls do not match the expected function calls.")
            print("Expected:")
            print(json.dumps(expected_function_calls, indent=2))
            print("Actual:")
            print(json.dumps(actual_function_calls, indent=2))
    elif isinstance(verification, OpenAIChatCompletionClientWrapper.TextResultVerification):
        # Handle text result verification
        content = verification.content
        print(f"Text content: {content}")
        # Implement your business logic based on the text content
    else:
        # Handle unexpected verification types
        raise Exception("Unknown verification type.")


async def main():
    langfuse = Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )
    logger.info(f"Langfuse host: {langfuse.base_url}")
    logger.info(f"Langfuse project_id: {langfuse.project_id}")

    tracer_provider = configure_tracing(langfuse_client=langfuse)
    runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)

    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level="DEBUG",
    )

    # Initialize the database with sample data
    await init_db()

    # ---------------------------
    # Team Setup (Swarm)
    # ---------------------------

    # team = Swarm(
    #     participants=[
    #         planner_agent,
    #         sql_query_agent,
    #         data_analysis_agent,
    #         explanation_agent,
    #     ],
    #     termination_condition=termination_condition,
    # )

    team = sql_query_agent

    team._runtime = runtime

    tasks = [
        {
            "task": "Investigate why Black Hat is becoming more popular in our shop.",
            "expected_function_calls": [
                {"name": "get_sales_data", "arguments": {"product_name": "Black Hat"}},
                {"name": "get_customer_feedback", "arguments": {"product_name": "Black Hat"}},
            ],
        },
        {
            "task": "Investigate why Yellow Hat is not popular in our shop.",
            "expected_function_calls": [
                {"name": "get_sales_data", "arguments": {"product_name": "Yellow Hat"}},
                {"name": "get_customer_feedback", "arguments": {"product_name": "Yellow Hat"}},
            ],
        },
    ]

    model_client.set_throw_on_create(True)

    saved_state = await team.save_state()

    # Run the tasks sequentially
    for task_entry in tasks:
        task_text = task_entry["task"]
        expected_function_calls = task_entry["expected_function_calls"]

        logger.debug(f"--- Starting task: {task_text} ---\n")

        try:
            # Run the team and capture the output
            await Console(team.run_stream(task=task_text))

            # After running, check the verification results
            for verification in model_client.create_results:
                await handle_verification(verification)

            # Clear the create_results after handling
            model_client.create_results.clear()

        except OpenAIChatCompletionClientWrapper.Verification as verification:
            await handle_verification(verification, expected_function_calls)

        logger.debug(f"\n--- Completed task: {task_text} ---\n")
        # await team.reset()  # Reset the team state before each task
        await team.load_state(saved_state)


if __name__ == "__main__":
    asyncio.run(main())

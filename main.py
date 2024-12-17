import asyncio
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
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from autogen_core.tools import FunctionTool

from util import model_client, settings, configure_tracing, OpenAIChatCompletionClientWrapper

# ---------------------------
# Database Setup
# ---------------------------

DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# erDiagram
#     sales {
#         Integer id PK
#         String product_name
#         String color
#         String size
#         Integer quantity_sold
#         Date date
#     }
#     customer_feedback {
#         Integer id PK
#         String product_name
#         String feedback
#         Date date
#     }
#     sales ||--o{ customer_feedback : "receives"

metadata = sa.MetaData()

# Define tables
sales_table = sa.Table(
    "sales",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("product_name", sa.String),
    sa.Column("color", sa.String),
    sa.Column("size", sa.String),
    sa.Column("quantity_sold", sa.Integer),
    sa.Column("date", sa.Date),
)

customer_feedback_table = sa.Table(
    "customer_feedback",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("product_name", sa.String),
    sa.Column("feedback", sa.String),
    sa.Column("date", sa.Date),
)

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Create session factory
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# Initialize database and populate with sample data
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    async with AsyncSessionLocal() as session:
        # Sample sales data for Yellow Hats
        yellow_hat_sales = [
            {
                "product_name": "Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity_sold": 120,
                "date": datetime.strptime("2023-06-01", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity_sold": 80,
                "date": datetime.strptime("2023-07-01", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity_sold": 40,
                "date": datetime.strptime("2023-08-01", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity_sold": 20,
                "date": datetime.strptime("2023-09-01", "%Y-%m-%d").date(),
            },
        ]

        # Sample sales data for Black Hats
        black_hat_sales = [
            {
                "product_name": "Black Hat",
                "color": "Black",
                "size": "Medium",
                "quantity_sold": 50,
                "date": datetime.strptime("2023-06-01", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Black Hat",
                "color": "Black",
                "size": "Medium",
                "quantity_sold": 60,
                "date": datetime.strptime("2023-07-01", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Black Hat",
                "color": "Black",
                "size": "Medium",
                "quantity_sold": 70,
                "date": datetime.strptime("2023-08-01", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Black Hat",
                "color": "Black",
                "size": "Medium",
                "quantity_sold": 80,
                "date": datetime.strptime("2023-09-01", "%Y-%m-%d").date(),
            },
        ]

        # Sample customer feedback data for Yellow Hats
        yellow_hat_feedback = [
            {
                "product_name": "Yellow Hat",
                "feedback": "The hat fades after washing.",
                "date": datetime.strptime("2023-07-15", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Yellow Hat",
                "feedback": "Size runs too big.",
                "date": datetime.strptime("2023-08-10", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Yellow Hat",
                "feedback": "Color is not as vibrant as pictured.",
                "date": datetime.strptime("2023-08-20", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Yellow Hat",
                "feedback": "Uncomfortable to wear for long periods.",
                "date": datetime.strptime("2023-09-05", "%Y-%m-%d").date(),
            },
        ]

        # Sample customer feedback data for Black Hats
        black_hat_feedback = [
            {
                "product_name": "Black Hat",
                "feedback": "Great quality, very comfortable.",
                "date": datetime.strptime("2023-07-12", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Black Hat",
                "feedback": "Stylish and fits well.",
                "date": datetime.strptime("2023-08-15", "%Y-%m-%d").date(),
            },
            {
                "product_name": "Black Hat",
                "feedback": "Perfect for everyday use.",
                "date": datetime.strptime("2023-09-08", "%Y-%m-%d").date(),
            },
        ]

        # Insert data
        await session.execute(sales_table.insert(), yellow_hat_sales + black_hat_sales)
        await session.execute(customer_feedback_table.insert(), yellow_hat_feedback + black_hat_feedback)
        await session.commit()


# Input -> mermaid diagram of DB tables


# ---------------------------
# Tools
# ---------------------------


# Tool to get sales data
async def get_sales_data(product_name: str) -> Dict[str, Any]:
    """
    Retrieve sales data for a given product.
    """
    async with AsyncSessionLocal() as session:
        stmt = (
            select(
                sales_table.c.date,
                sales_table.c.quantity_sold,
            )
            .where(sales_table.c.product_name == product_name)
            .order_by(sales_table.c.date)
        )
        result = await session.execute(stmt)
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
        stmt = (
            select(
                customer_feedback_table.c.feedback,
                customer_feedback_table.c.date,
            )
            .where(customer_feedback_table.c.product_name == product_name)
            .order_by(customer_feedback_table.c.date)
        )
        result = await session.execute(stmt)
        data = result.fetchall()
        return {
            "customer_feedback": [{"date": row.date.strftime("%Y-%m-%d"), "feedback": row.feedback} for row in data]
        }


# Tool to analyze data
async def analyze_data(product_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # Analyze sales and customer feedback data for a given product to provide insights.
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
async def handle_verification(verification):
    if isinstance(verification, OpenAIChatCompletionClientWrapper.FunctionCallVerification):
        # Handle function call verification
        for function_call_record in verification.function_calls:
            function_name = function_call_record.function_name
            arguments = function_call_record.arguments
            print(f"Function called: {function_name} with arguments: {arguments}")
            # Implement your business logic based on the function calls
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

    # Define the tasks
    # ---------------------------
    # Team Setup (Swarm)
    # ---------------------------

    team = Swarm(
        participants=[
            planner_agent,
            sql_query_agent,
            data_analysis_agent,
            explanation_agent,
        ],
        termination_condition=termination_condition,
    )

    team._runtime = runtime

    tasks = [
        "Investigate why Black Hat is becoming more popular in our shop.",
        "Investigate why Yellow Hat is not popular in our shop.",
    ]

    model_client.set_throw_on_create(True)

    # Run the tasks sequentially
    for task in tasks:
        logger.debug(f"--- Starting task: {task} ---\n")

        try:
            # Run the team and capture the output
            await Console(team.run_stream(task=task))

            # After running, check the verification results
            for verification in model_client.create_results:
                await handle_verification(verification)

            # Clear the create_results after handling
            model_client.create_results.clear()

        except OpenAIChatCompletionClientWrapper.Verification as verification:
            # If throw_on_create is True and an exception is raised
            await handle_verification(verification)

        logger.debug(f"\n--- Completed task: {task} ---\n")
        await team.reset()  # Reset the team state before each task


if __name__ == "__main__":
    asyncio.run(main())

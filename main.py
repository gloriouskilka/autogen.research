import asyncio
from datetime import datetime
from typing import Dict, Any

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pydantic_settings import BaseSettings, SettingsConfigDict

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func

from autogen_core.tools import FunctionTool

# ---------------------------
# Settings and Configuration
# ---------------------------


class Settings(BaseSettings):
    openai_api_key: str = None
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# ---------------------------
# Database Setup
# ---------------------------

DATABASE_URL = "sqlite+aiosqlite:///:memory:"

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
        # Sample sales data
        sample_sales = [
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
        # Sample customer feedback data
        sample_feedback = [
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

        # Insert data
        await session.execute(sales_table.insert(), sample_sales)
        await session.execute(customer_feedback_table.insert(), sample_feedback)
        await session.commit()


# ---------------------------
# Tool Definitions
# ---------------------------


# Tool to execute SQL queries
async def execute_sql_query(query_name: str) -> Dict[str, Any]:
    async with AsyncSessionLocal() as session:
        if query_name == "get_sales_data":
            stmt = (
                select(
                    sales_table.c.date,
                    sales_table.c.quantity_sold,
                )
                .where(sales_table.c.product_name == "Yellow Hat")
                .order_by(sales_table.c.date)
            )
            result = await session.execute(stmt)
            data = result.fetchall()
            return {
                "sales_data": [
                    {"date": row.date.strftime("%Y-%m-%d"), "quantity_sold": row.quantity_sold} for row in data
                ]
            }
        elif query_name == "get_customer_feedback":
            stmt = (
                select(
                    customer_feedback_table.c.feedback,
                    customer_feedback_table.c.date,
                )
                .where(customer_feedback_table.c.product_name == "Yellow Hat")
                .order_by(customer_feedback_table.c.date)
            )
            result = await session.execute(stmt)
            data = result.fetchall()
            return {
                "customer_feedback": [{"date": row.date.strftime("%Y-%m-%d"), "feedback": row.feedback} for row in data]
            }
        else:
            return {"error": f"Unknown query name: {query_name}"}


# Tool to analyze data
async def analyze_data(data: Dict[str, Any]) -> Dict[str, Any]:
    analysis_results = {}

    # Analyze sales trends
    if "sales_data" in data:
        sales = data["sales_data"]
        quantities = [entry["quantity_sold"] for entry in sales]
        if len(quantities) >= 2:
            if quantities[-1] < quantities[0]:
                analysis_results["sales_trend"] = "Sales of Yellow Hats have significantly decreased over time."
            else:
                analysis_results["sales_trend"] = "Sales are stable or increasing."
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
                for keyword in ["fade", "size", "color", "uncomfortable", "not as pictured"]
            ):
                negative_feedback.append(feedback)
        if negative_feedback:
            analysis_results["customer_feedback_analysis"] = (
                f"Negative feedback indicates issues with product quality: {len(negative_feedback)} instances."
            )
        else:
            analysis_results["customer_feedback_analysis"] = "Customer feedback is generally positive."
    else:
        analysis_results["customer_feedback_analysis"] = "Customer feedback data not provided."

    return analysis_results


# Wrap tools with FunctionTool
execute_sql_query_tool = FunctionTool(
    execute_sql_query,
    description="Execute predefined SQL queries to retrieve sales or customer feedback data.",
)
analyze_data_tool = FunctionTool(
    analyze_data,
    description="Analyze sales and customer feedback data to provide insights.",
)

# ---------------------------
# Agents
# ---------------------------

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=settings.openai_api_key,
)

# Planner Agent
planner_agent = AssistantAgent(
    name="PlannerAgent",
    model_client=model_client,
    handoffs=["SQLQueryAgent", "DataAnalysisAgent", "ExplanationAgent"],
    system_message="""
You are the PlannerAgent. Your role is to coordinate the investigation to understand why Yellow Hats are not popular.

- **Step 1**: Ask the SQLQueryAgent to retrieve sales data and customer feedback for Yellow Hats.
- **Step 2**: When you receive data from SQLQueryAgent, include this data in the HandoffMessage content to DataAnalysisAgent.
- **Step 3**: Request the DataAnalysisAgent to analyze the data.
- **Step 4**: Send the analysis results to ExplanationAgent for generating a comprehensive explanation.

Always include necessary data in your handoffs.
""",
)

# SQL Query Agent
sql_query_agent = AssistantAgent(
    name="SQLQueryAgent",
    model_client=model_client,
    tools=[execute_sql_query_tool],
    handoffs=["PlannerAgent"],
    system_message="""
You are the SQLQueryAgent.

- Use the `execute_sql_query` tool to run predefined queries: "get_sales_data" or "get_customer_feedback".
- Retrieve sales trends and customer feedback for Yellow Hats.
- After execution, include the retrieved data in the HandoffMessage content and return it to PlannerAgent.
""",
)

# Data Analysis Agent
data_analysis_agent = AssistantAgent(
    name="DataAnalysisAgent",
    model_client=model_client,
    tools=[analyze_data_tool],
    handoffs=["PlannerAgent"],
    system_message="""
You are the DataAnalysisAgent.

- Receive data from PlannerAgent via HandoffMessage content.
- Use the `analyze_data` tool to analyze the sales and customer feedback data provided.
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
- Provide actionable insights and recommendations to improve the popularity of Yellow Hats.
- Once done, reply directly to the user and mention 'TERMINATE' to end the conversation.
""",
)

# ---------------------------
# Termination Condition
# ---------------------------

termination_condition = TextMentionTermination("TERMINATE")

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

# ---------------------------
# Execution
# ---------------------------

# Define the task
task = "Investigate why Yellow Hats are not popular in our shop."


# Run the team
async def main():
    # Initialize the database with sample data
    await init_db()

    # Run the task
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())

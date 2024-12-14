import asyncio

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pydantic_settings import BaseSettings, SettingsConfigDict

import asyncio
from typing import Any, Dict, List

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func

# Import Autogen components
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool, Tool


class Settings(BaseSettings):
    openai_api_key: str = ""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# ---------------------------
# Database Setup
# ---------------------------

DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Define metadata object
metadata = sa.MetaData()

# Define tables
sales_table = sa.Table(
    "sales",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("product_name", sa.String),
    sa.Column("color", sa.String),
    sa.Column("size", sa.String),
    sa.Column("quantity", sa.Integer),
    sa.Column("date", sa.Date),
)

user_feedback_table = sa.Table(
    "user_feedback",
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
                "product_name": "Large Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity": 100,
                "date": "2023-07-01",
            },
            {
                "product_name": "Large Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity": 80,
                "date": "2023-08-01",
            },
            {
                "product_name": "Large Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity": 50,
                "date": "2023-09-01",
            },
            {
                "product_name": "Large Yellow Hat",
                "color": "Yellow",
                "size": "Large",
                "quantity": 20,
                "date": "2023-10-01",
            },
        ]
        # Sample user feedback data
        sample_feedback = [
            {
                "product_name": "Large Yellow Hat",
                "feedback": "Great quality!",
                "date": "2023-07-05",
            },
            {
                "product_name": "Large Yellow Hat",
                "feedback": "Too big for me.",
                "date": "2023-08-15",
            },
            {
                "product_name": "Large Yellow Hat",
                "feedback": "Color faded after washing.",
                "date": "2023-09-10",
            },
            {
                "product_name": "Large Yellow Hat",
                "feedback": "Not as described.",
                "date": "2023-09-20",
            },
        ]

        # Insert data
        await session.execute(sales_table.insert(), sample_sales)
        await session.execute(user_feedback_table.insert(), sample_feedback)
        await session.commit()


# Run database initialization
asyncio.run(init_db())

# ---------------------------
# Tools
# ---------------------------


# Tool to execute SQL queries
async def execute_sql_query(query_name: str) -> Dict[str, Any]:
    async with AsyncSessionLocal() as session:
        if query_name == "sales_trends":
            # Example: Get monthly sales quantities
            stmt = (
                select(
                    sales_table.c.date,
                    func.sum(sales_table.c.quantity).label("total_quantity"),
                )
                .group_by(sales_table.c.date)
                .order_by(sales_table.c.date)
            )
            result = await session.execute(stmt)
            data = result.fetchall()
            return {
                "sales_trends": [
                    {"date": str(row.date), "total_quantity": row.total_quantity}
                    for row in data
                ]
            }
        elif query_name == "user_feedback":
            # Example: Get recent user feedback
            stmt = (
                select(user_feedback_table.c.feedback, user_feedback_table.c.date)
                .order_by(user_feedback_table.c.date.desc())
                .limit(5)
            )
            result = await session.execute(stmt)
            data = result.fetchall()
            return {
                "user_feedback": [
                    {"date": str(row.date), "feedback": row.feedback} for row in data
                ]
            }
        else:
            return {"error": f"Unknown query name: {query_name}"}


# Tool to analyze data
async def analyze_data(data: Dict[str, Any]) -> Dict[str, Any]:
    # Perform analysis on the data
    analysis_results = {}
    if "sales_trends" in data:
        sales_data = data["sales_trends"]
        quantities = [entry["total_quantity"] for entry in sales_data]
        if len(quantities) >= 2 and quantities[-1] < quantities[-2]:
            analysis_results["sales_trend"] = (
                "Decrease in sales observed in the most recent month."
            )
        else:
            analysis_results["sales_trend"] = "Sales are stable or increasing."
    if "user_feedback" in data:
        feedback_data = data["user_feedback"]
        negative_feedback = [
            fb
            for fb in feedback_data
            if "not" in fb["feedback"].lower()
            or "too" in fb["feedback"].lower()
            or "faded" in fb["feedback"].lower()
        ]
        if negative_feedback:
            analysis_results["user_feedback_analysis"] = (
                f"Negative feedback detected: {len(negative_feedback)} instances."
            )
        else:
            analysis_results["user_feedback_analysis"] = (
                "User feedback is generally positive."
            )
    return analysis_results


# Wrap tools with FunctionTool
execute_sql_query_tool = FunctionTool(
    execute_sql_query, description="Execute predefined SQL queries to retrieve data."
)
analyze_data_tool = FunctionTool(
    analyze_data, description="Analyze data and provide insights."
)

# ---------------------------
# Agents
# ---------------------------

# Planner Agent
planner_agent = AssistantAgent(
    name="PlannerAgent",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    handoffs=["SQLQueryAgent", "DataAnalysisAgent", "ExplanationAgent"],
    system_message="""You are the PlannerAgent. Your role is to coordinate the process of analyzing why users have stopped buying large yellow hats.

- **Step 1**: Ask the SQLQueryAgent to retrieve relevant sales and feedback data.
- **Step 2**: Pass the retrieved data to the DataAnalysisAgent for analysis.
- **Step 3**: Request the ExplanationAgent to generate a clear explanation based on the analysis.
- **Terminate**: Once the explanation is ready, ensure it is delivered to the user.

Always handoff to the appropriate agent after each step by sending a HandoffMessage.
""",
)

# SQL Query Agent
sql_query_agent = AssistantAgent(
    name="SQLQueryAgent",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    tools=[execute_sql_query_tool],
    handoffs=["PlannerAgent"],
    system_message="""You are the SQLQueryAgent.

- Use the `execute_sql_query` tool to run predefined queries.
- Available queries: "sales_trends", "user_feedback".
- Return the data to PlannerAgent in a HandoffMessage after execution.

If you encounter any issues, inform the PlannerAgent.
""",
)

# Data Analysis Agent
data_analysis_agent = AssistantAgent(
    name="DataAnalysisAgent",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    tools=[analyze_data_tool],
    handoffs=["PlannerAgent"],
    system_message="""You are the DataAnalysisAgent.

- Use the `analyze_data` tool to analyze the data provided.
- Return the analysis results to PlannerAgent in a HandoffMessage.

If data is missing or invalid, inform the PlannerAgent.
""",
)

# Explanation Agent
explanation_agent = AssistantAgent(
    name="ExplanationAgent",
    model_client=OpenAIChatCompletionClient(model="gpt-4"),
    handoffs=[],
    system_message="""You are the ExplanationAgent.

- Generate a clear and comprehensive explanation based on the analysis results.
- Provide actionable insights and recommendations.
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
task = "Why have users stopped buying large yellow hats?"


# Run the team
async def main():
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())

Certainly! I'll continue the implementation by filling in the missing parts and providing a complete example of a complex agentic application as per your requirements. This implementation will:

- **Query the database** via deterministic tools that load SQL queries from disk.
- **Analyze dataframes** using pandas and prepare small overview dictionaries.
- **Use an LLM decider** in the middle to decide the next pipeline based on the small overview dict.
- **Run a deterministic final pipeline** on the entire dataframe from previous steps.
- **Generate LLM recommendations** based on the overviews prepared by deterministic steps, resulting in natural language TODO actions for the user.

---

### **Updated Project Structure**

```
project_root/
├── main.py
├── config.py
├── agents/
│   ├── __init__.py
│   ├── coordinator_agent.py
│   ├── middle_decider_agent.py
│   ├── analysis_agent.py
│   ├── pipeline_a_agent.py
│   ├── pipeline_b_agent.py
│   ├── final_pipeline_agent.py
│   └── common.py
├── tools/
│   ├── __init__.py
│   ├── query_tools.py
│   └── function_tools.py
├── models/
│   ├── __init__.py
│   └── openai_client.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── db_utils.py
├── queries/
│   ├── __init__.py
│   ├── 01_create.sql
│   ├── 02_insert.sql
│   ├── 03_query_01_excess.sql
│   ├── ... (other SQL query files)
│   └── 03_query_12_return_reasons.sql
└── requirements.txt
```

---

### **Implementation Details**

#### **1. main.py**

Sets up the runtime, registers agents and tools, and starts the application.

```python
import asyncio
from config import OPENAI_API_KEY, MODEL_NAME, DATABASE_PATH
from autogen_core import SingleThreadedAgentRuntime
from models.openai_client import OpenAIChatCompletionClient
from agents.coordinator_agent import CoordinatorAgent
from agents.middle_decider_agent import MiddleDeciderAgent
from agents.analysis_agent import AnalysisAgent
from agents.pipeline_a_agent import PipelineAAgent
from agents.pipeline_b_agent import PipelineBAgent
from agents.final_pipeline_agent import FinalPipelineAgent
from autogen_core.tool_agent import ToolAgent
from tools.function_tools import (
    pipeline_a_tool,
    pipeline_b_tool,
    final_pipeline_tool,
    query_tool,
)
from agents.common import UserInput


async def main():
    runtime = SingleThreadedAgentRuntime()
    model_client = OpenAIChatCompletionClient(model_name=MODEL_NAME, api_key=OPENAI_API_KEY)

    # Register agents
    await CoordinatorAgent.register(
        runtime=runtime, type="coordinator_agent_type", factory=lambda: CoordinatorAgent(model_client)
    )
    await MiddleDeciderAgent.register(
        runtime=runtime,
        type="middle_decider_agent_type",
        factory=lambda: MiddleDeciderAgent(model_client),
    )
    await AnalysisAgent.register(
        runtime=runtime, type="analysis_agent_type", factory=lambda: AnalysisAgent(model_client)
    )
    await PipelineAAgent.register(
        runtime=runtime, type="pipeline_a_agent_type", factory=PipelineAAgent
    )
    await PipelineBAgent.register(
        runtime=runtime, type="pipeline_b_agent_type", factory=PipelineBAgent
    )
    await FinalPipelineAgent.register(
        runtime=runtime, type="final_pipeline_agent_type", factory=FinalPipelineAgent
    )

    # Register ToolAgent with query tool and pipeline tools
    pipeline_tools = [pipeline_a_tool, pipeline_b_tool, final_pipeline_tool, query_tool]
    tool_agent = ToolAgent(description="Pipeline and Query Tool Agent", tools=pipeline_tools)
    await tool_agent.register(runtime=runtime, type="tool_agent_type", factory=lambda: tool_agent)

    runtime.start()

    # Simulate user input and initiate processing
    coordinator_agent_id = await runtime.get("coordinator_agent_type", key="coordinator_agent")
    user_input_text = input("Enter your request: ")
    user_input = UserInput(text=user_input_text)

    final_result = await runtime.send_message(
        message=user_input,
        recipient=coordinator_agent_id,
    )

    print("Final Analysis Report:")
    print(final_result.result)

    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

---

#### **2. config.py**

Contains configuration settings.

```python
OPENAI_API_KEY = "your-openai-api-key"
MODEL_NAME = "gpt-4-0613"
DATABASE_PATH = "path/to/your/database.db"
```

---

#### **3. agents/common.py**

Defines shared data classes.

```python
from dataclasses import dataclass
from typing import Dict


@dataclass
class UserInput:
    text: str


@dataclass
class FinalResult:
    result: str
```

---

#### **4. tools/query_tools.py**

Defines the query tool that loads SQL queries from disk and executes them against the database.

```python
from autogen_core.tools import FunctionTool
from utils.db_utils import execute_sql_query
from autogen_core import CancellationToken
from typing import Any


async def query_database(query_name: str, cancellation_token: CancellationToken = None) -> Any:
    sql_query = load_sql_query(query_name)
    results = execute_sql_query(sql_query)
    return results


def load_sql_query(query_name: str) -> str:
    query_file_path = f"./queries/{query_name}.sql"
    with open(query_file_path, "r", encoding="utf-8") as file:
        sql_query = file.read()
    return sql_query


query_tool = FunctionTool(
    func=query_database, description="Query the database using a specified SQL file name."
)
```

---

#### **5. utils/db_utils.py**

Contains utility functions for database interaction.

```python
import sqlite3
import pandas as pd
from config import DATABASE_PATH


def execute_sql_query(sql_query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df
```

---

#### **6. utils/data_utils.py**

Handles data processing and analysis using pandas.

```python
import pandas as pd
from typing import Tuple, Dict, Any


def process_data_pipeline_a(data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict]:
    # Deterministic processing logic for Pipeline A
    overview = {
        "columns": data.columns.tolist(),
        "num_rows": len(data),
        "summary_statistics": data.describe().to_dict(),
    }
    return data, overview


def process_data_pipeline_b(data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict]:
    # Deterministic processing logic for Pipeline B
    data["new_column"] = data.select_dtypes(include="number").mean(axis=1)
    overview = {
        "columns": data.columns.tolist(),
        "num_rows": len(data),
        "added_new_column": True,
    }
    return data, overview


def analyze_full_data(dataframe: pd.DataFrame) -> Dict:
    # Deterministic final analysis logic
    correlation = dataframe.corr().to_dict()
    overview_info = {
        "correlation_matrix": correlation,
        "num_rows": len(dataframe),
    }
    return overview_info
```

---

#### **7. tools/function_tools.py**

Defines the pipeline functions as tools.

```python
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from typing import Dict, Any
from utils.db_utils import execute_sql_query
from utils.data_utils import (
    process_data_pipeline_a,
    process_data_pipeline_b,
    analyze_full_data,
)
import pandas as pd


async def pipeline_a(data: pd.DataFrame, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    dataframe, description_dict = process_data_pipeline_a(data)
    return {"dataframe": dataframe.to_dict(), "description_dict": description_dict}


async def pipeline_b(data: pd.DataFrame, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    dataframe, description_dict = process_data_pipeline_b(data)
    return {"dataframe": dataframe.to_dict(), "description_dict": description_dict}


async def final_pipeline(dataframe: pd.DataFrame, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    overview_info = analyze_full_data(dataframe)
    return {"overview_info": overview_info}


# Wrap functions as tools
pipeline_a_tool = FunctionTool(
    func=pipeline_a, description="Process data using Pipeline A."
)
pipeline_b_tool = FunctionTool(
    func=pipeline_b, description="Process data using Pipeline B."
)
final_pipeline_tool = FunctionTool(
    func=final_pipeline, description="Execute the final analysis pipeline."
)
```

---

#### **8. agents/coordinator_agent.py**

Modified to use the query tool and to initiate the first pipeline based on the LLM's decision.

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from agents.common import UserInput, FinalResult
from typing import List
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
                content="""You are an assistant that decides which SQL query to execute based on user input.
                Available queries are query_01_excess, query_02_obsolete, query_03_top_selling, etc."""
            ),
            UserMessage(content=user_text, source="user"),
        ]

        tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")

        # Use the caller loop to decide which query to run
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[query_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant",
        )

        # Extract the query result
        last_message = generated_messages[-1]
        if isinstance(last_message, AssistantMessage) and isinstance(last_message.content, str):
            try:
                result_data = json.loads(last_message.content)
                dataframe_dict = result_data
                # Proceed to the LLM-decided pipeline
                middle_decider_agent_id = await self.runtime.get(
                    "middle_decider_agent_type", key="middle_decider_agent"
                )
                decision_info = await self.send_message(
                    message={"dataframe": dataframe_dict},
                    recipient=middle_decider_agent_id,
                    cancellation_token=ctx.cancellation_token,
                )

                return decision_info  # FinalResult will be returned from the final pipeline
            except json.JSONDecodeError:
                return FinalResult(result="Error: Failed to parse the query result.")
        else:
            return FinalResult(result="Error: Unable to process input.")
```

---

#### **9. agents/middle_decider_agent.py**

Uses the small overview dict to decide which pipeline to run next.

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from typing import List
from agents.common import FinalResult
import json
import pandas as pd


class MiddleDeciderAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Middle Decider Agent")
        self.model_client = model_client

    @rpc
    async def decide_next_step(self, message: dict, ctx: MessageContext) -> FinalResult:
        dataframe_dict = message["dataframe"]
        # Prepare a small description dict (e.g., column names and row count)
        dataframe = pd.DataFrame.from_dict(dataframe_dict)
        description_dict = {
            "columns": dataframe.columns.tolist(),
            "num_rows": len(dataframe),
        }

        input_messages: List[LLMMessage] = [
            SystemMessage(
                content="""Based on the provided data description, decide which processing function to call next.
                Available functions are pipeline_a and pipeline_b."""
            ),
            UserMessage(content=str(description_dict), source="user"),
        ]

        tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")

        # Use the caller loop to decide the next pipeline
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[pipeline_a_tool.schema, pipeline_b_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant",
        )

        # Extract the pipeline processing result
        last_message = generated_messages[-1]
        if isinstance(last_message, AssistantMessage) and isinstance(last_message.content, str):
            try:
                result_data = json.loads(last_message.content)
                dataframe = pd.DataFrame.from_dict(result_data["dataframe"])
                description_dict = result_data["description_dict"]
                # Proceed to the final pipeline
                final_pipeline_agent_id = await self.runtime.get(
                    "final_pipeline_agent_type", key="final_pipeline_agent"
                )
                final_result = await self.send_message(
                    message={"dataframe": dataframe.to_dict(), "description": description_dict},
                    recipient=final_pipeline_agent_id,
                    cancellation_token=ctx.cancellation_token,
                )
                return final_result
            except json.JSONDecodeError:
                return FinalResult(result="Error: Failed to parse the pipeline result.")
        else:
            return FinalResult(result="Error: Unable to process data.")
```

---

#### **10. agents/final_pipeline_agent.py**

Runs the deterministic final analysis.

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from agents.common import FinalResult
import pandas as pd
from utils.data_utils import analyze_full_data


class FinalPipelineAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Final Pipeline Agent")

    @rpc
    async def run_final_pipeline(self, message: dict, ctx: MessageContext) -> FinalResult:
        dataframe_dict = message["dataframe"]
        dataframe = pd.DataFrame.from_dict(dataframe_dict)
        description = message.get("description", {})

        # Run the final deterministic analysis
        overview_info = analyze_full_data(dataframe)

        # Proceed to the analysis agent
        analysis_agent_id = await self.runtime.get("analysis_agent_type", key="analysis_agent")
        analysis_result = await self.send_message(
            message=overview_info,
            recipient=analysis_agent_id,
            cancellation_token=ctx.cancellation_token,
        )

        return FinalResult(result=analysis_result)
```

---

#### **11. agents/analysis_agent.py**

Generates LLM recommendations based on the overview.

```python
from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage
from agents.common import FinalResult
from typing import List


class AnalysisAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Analysis Agent")
        self.model_client = model_client

    @rpc
    async def generate_report(self, message: dict, ctx: MessageContext) -> str:
        overview_info = message  # The small overview dict

        input_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are a data analyst. Based on the following data analysis overview, provide actionable recommendations and expected results in natural language."""
            ),
            UserMessage(content=str(overview_info), source="user"),
        ]

        response = await self.model_client.create(
            messages=input_messages,
            cancellation_token=ctx.cancellation_token,
        )

        if isinstance(response.content, str):
            return response.content

        return "Error: Failed to generate report."
```

---

#### **12. models/openai_client.py**

Implementation of `ChatCompletionClient` using OpenAI's API.

```python
import openai
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, RequestUsage
from autogen_core import CancellationToken
from typing import List, Mapping, Any
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage, FunctionExecutionResultMessage, FunctionCall
from autogen_core.tools import ToolSchema, Tool
import asyncio


class OpenAIChatCompletionClient(ChatCompletionClient):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        openai.api_key = api_key

    async def create(
        self,
        messages: List[LLMMessage],
        tools: List[Tool | ToolSchema] = [],
        json_output: bool = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken = None,
    ) -> CreateResult:
        # Convert messages to OpenAI API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    api_messages.append({"role": "assistant", "content": msg.content})
                else:
                    # Handle FunctionCall instances
                    for func_call in msg.content:
                        api_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "function_call": {
                                    "name": func_call.name,
                                    "arguments": func_call.arguments,
                                },
                            }
                        )
            elif isinstance(msg, FunctionExecutionResultMessage):
                for result in msg.content:
                    api_messages.append(
                        {
                            "role": "function",
                            "name": result.call_id,
                            "content": result.content,
                        }
                    )

        # Prepare functions for the API
        api_functions = [tool.schema for tool in tools]

        # Call the OpenAI API
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=api_messages,
            functions=api_functions,
            function_call="auto",
            **extra_create_args,
        )

        # Extract the response
        choice = response.choices[0]
        message = choice.message
        content = message.get("content", None)

        if "function_call" in message:
            function_call = FunctionCall(
                id="unique_call_id",  # Generate or assign a unique ID
                name=message["function_call"]["name"],
                arguments=message["function_call"]["arguments"],
            )
            content = [function_call]

        return CreateResult(
            finish_reason=choice.finish_reason,
            content=content,
            usage=RequestUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            ),
            cached=False,
        )
```

---

### **Testing the Application**

To test the application, ensure you have:

- The required dependencies installed (as specified in `requirements.txt`).
- The database initialized with the SQL scripts provided in the `queries/` directory.
- The OpenAI API key configured in `config.py`.
- The database path correctly set in `config.py`.

**Example User Input:**

```
Enter your request: I need a report on excess inventory. Please analyze it and provide recommendations.
```

**Expected Flow:**

1. **Coordinator Agent:**

   - Uses the LLM to decide which SQL query to run based on user input.
   - The LLM should select `query_01_excess`.

2. **Query Tool:**

   - Loads `query_01_excess.sql` from disk.
   - Executes the query against the database.
   - Returns the result as a DataFrame.

3. **Middle Decider Agent:**

   - Prepares a small description dict from the DataFrame (e.g., columns and number of rows).
   - Uses the LLM to decide which pipeline to run next (e.g., `pipeline_a` or `pipeline_b`).

4. **Pipeline Agent:**

   - Executes the selected pipeline deterministically on the DataFrame.
   - Returns processed DataFrame and description_dict.

5. **Final Pipeline Agent:**

   - Performs final deterministic analysis on the full DataFrame (e.g., correlation matrix).
   - Prepares an overview_info dict.

6. **Analysis Agent:**

   - Uses the LLM to generate natural language recommendations based on the overview_info.
   - Returns the final analysis report.

**Final Output:**

- A natural language report with actionable recommendations for the user regarding the excess inventory.

---

### **Key Considerations**

- **Error Handling:** Ensure each agent handles exceptions and provides meaningful error messages to the user.
- **Efficiency:** Pass only small description dicts to the LLM to reduce token usage.
- **Data Serialization:** When passing DataFrames between agents, serialize them properly (e.g., using `to_dict()` and `from_dict()`).
- **Privacy and Compliance:** Be careful with the data sent to the LLM, especially if it contains sensitive information.
- **OpenAI Policies:** Ensure compliance with OpenAI's policies regarding data usage and safety.

---

### **Conclusion**

This implementation provides an example of a complex agentic application that:

- **Queries a database** using SQL files loaded from disk.
- **Processes data deterministically** using pandas DataFrames.
- **Uses LLMs for decision-making** at key points without exposing large data structures.
- **Generates natural language reports** with actionable recommendations for the user.

The modular architecture allows for easy extension, testing, and maintenance, aligning with best practices for building sophisticated agent-based systems.

If you have any questions or need further clarification on specific parts of the code, feel free to ask!
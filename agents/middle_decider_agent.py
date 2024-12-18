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

# middle_decider_agent.py

from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import SystemMessage, UserMessage


class MiddleDeciderAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Middle Decider Agent")
        self.model_client = model_client

    @rpc
    async def decide_next_step(self, message: dict, ctx: MessageContext) -> dict:
        overview_dict = message["overview"]
        input_messages = [
            SystemMessage(content="Based on the overview, decide the next action."),
            UserMessage(content=str(overview_dict)),
        ]
        # Interact with the LLM to decide the next pipeline
        # ...
        return {"decision": "Proceed with final analysis"}

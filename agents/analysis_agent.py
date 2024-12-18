from typing import Any, Mapping, List, Sequence, Dict
from autogen_core import CancellationToken
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, ChatMessage
from autogen_core.models import RequestUsage
from ..tools.sql_tools import execute_sql_query


class AnalysisAgent(BaseChatAgent):
    """
    Agent that analyzes database queries and provides recommendations.
    """

    def __init__(self, name: str):
        super().__init__(name=name, description="Analyzes database queries and provides insights.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # Extract the query name from the messages
        last_message = messages[-1] if messages else None
        if isinstance(last_message, TextMessage):
            query_name = last_message.content.strip()
        else:
            query_name = "excess_inventory"  # Default query

        # Execute the SQL query
        results = await execute_sql_query(query_name)

        # Analyze the results (implement your analysis logic)
        analysis = self.analyze_results(results)

        # Return the analysis as a TextMessage
        return Response(chat_message=TextMessage(content=analysis, source=self.name))

    def analyze_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Analyze the query results and generate recommendations.

        Args:
            results (List[Dict[str, Any]]): Query results.

        Returns:
            str: Analysis and recommendations.
        """
        # Placeholder for analysis logic.
        # Implement your domain-specific analysis here.
        if not results:
            return "No data found for the given query."
        else:
            # Example analysis (customize as needed)
            total_items = len(results)
            return f

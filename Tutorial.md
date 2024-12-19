# Tutorial.md

## Getting Started with New AutoGen Core API: A Step-by-Step Guide for Developers

### Introduction

Modern travelers expect quick and personalized assistance when planning their journeys. This involves booking flights, hotels, car rentals, and discovering activities at their destination. Traditionally, this requires multiple searches and interactions with different platforms, which can be cumbersome and time-consuming. To streamline this process, a multi-agent chatbot system is designed to handle a comprehensive travel planning request. The goal is to manage different components of a travel itinerary through specialized agents that work together seamlessly, ensuring that each part of the journey is well-coordinated and efficiently executed.

### Problem Statement

Imagine a user planning a complex trip—they need to book a flight, find a hotel, rent a car, and look for activities to enjoy during their stay. Handling all these tasks in a cohesive and intelligent manner requires a system that can:

- Understand user requests (e.g., greetings, specific travel details).
- Identify the correct components of the plan (e.g., hotel booking, flight search).
- Coordinate multiple tasks that need to happen in parallel or sequentially.
- Aggregate all the gathered information into a cohesive travel plan.

The challenge is to build a system that effectively routes each user query to the correct service, maintains session state, and provides a streamlined response to the user without requiring manual intervention. This is where the concept of multiple specialized agents comes in.

In this guide, we will build a travel chatbot that can help users with questions about destinations, travel planning, and booking cars, flights, and hotels.

### Topics Covered

1. Setting up the foundation (data types, user input handling)
2. Creating and integrating agents
3. Using structured outputs and LLMs for enhanced responses
4. Managing communication (publish vs. send)
5. Implementing routing and handling complex tasks
6. Expanding with specialized agents
7. Logging, error handling, and deployment
8. Integration to Teams

### Layout and Role of Agents

- **UserProxyAgent**: This agent serves as the main interface between the user and the rest of the system. It receives messages from the user and sends responses back. It ensures that the user's requests are correctly routed and that the responses from different agents are delivered back to the user via WebSocket connections.

- **Planner Agent (Router)**: This agent is the brain that interprets the user's initial message and decides on the next steps. It is responsible for creating a TravelPlan that outlines which subtasks need to be handled and whether those tasks can be assigned to individual agents or require coordination through the GroupChatManager. If the user's message is just a greeting, this agent responds immediately to acknowledge and engage the user.

- **GroupChatManager**: This agent plays a vital role when a travel plan contains multiple subtasks that need to be handled by different specialized agents. The GroupChatManager is responsible for coordinating these subtasks—sending specific requests to the appropriate agents and aggregating the responses into a cohesive final travel plan that can be sent back to the user.

- **Specialized Agents (Flight, Hotel, Car Rental, Activities)**: These agents are responsible for executing specific parts of the travel plan:
  - **FlightAgent** handles flight bookings based on the provided travel dates and destinations.
  - **HotelAgent** takes care of hotel reservations, providing the best options for accommodations.
  - **CarRentalAgent** helps the user rent a car if needed for their trip.
  - **ActivitiesAgent** suggests activities based on the user's destination and interests.

### TravelPlan and Routing Decisions

When the user provides a complex request, the Planner Agent creates a TravelPlan detailing all required subtasks (e.g., booking a hotel, reserving flights, etc.). If there is only one subtask, it is directly assigned to the corresponding agent. However, if multiple subtasks are present, they are routed to the GroupChatManager for effective coordination.

### High-Level Overview of the Flow

#### Create Initial Data Types for Interaction

Define data types for agent communication, such as `EndUserMessage` for taking input messages from the user and `AgentStructuredResponse` for structured responses. Defining data types early is crucial for scalability and maintainability, as it helps in standardizing communication between agents, making the system easier to extend and modify as requirements evolve. Defining data types helps in routing messages between different agents and building logic.

```python
from pydantic import BaseModel

class EndUserMessage(BaseModel):
    content: str
    source: str

class AgentStructuredResponse(BaseModel):
    agent_type: str
    data: dict
    message: str
```

#### Set Up a User Agent to Accept Input

Create a user proxy agent that interfaces with users and relays input to other agents. See the below code where we have `handle_user_message` that will be executed every time an `EndUserMessage` is available. `UserProxyAgent` is a subclass of `RoutedAgent` that facilitates communication between agents, enabling seamless routing of messages. This modular approach enhances the architecture by allowing agents to focus on their specific tasks while easily interacting with other agents.

```python
from autogen_core.components import RoutedAgent, message_handler

class UserProxyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("UserProxyAgent")

    @message_handler
    async def handle_user_message(self, message: EndUserMessage, ctx) -> None:
        await self.publish_message(message, ctx.topic_id)
```

Agents in AutoGen Core communicate exclusively through serializable messages, which can be defined using Pydantic's BaseModel or dataclasses. Message handlers process these messages, and the RoutedAgent class simplifies routing messages by type. Direct messaging allows request/response interactions, while broadcasting publishes messages to topics without expecting responses.

Below is an example of how we are leveraging `send_message` to call agents. We leverage this in `travel_group_chat.py` to consolidate messages from different agents into a single response.

```python
# You can send a message to a specific topic like below

self.send_message(
    TravelRequest(
        source="GroupChatManager",
        content=task.task_details,
        original_task=message.main_task,
    ),
    AgentId(type=task.assigned_agent, key=self._session_id),
)
```

#### Create Your First Agent

Let's create a destination agent that will send info about the destination as a response, and as an added bonus, we will have the output in a structured format so we can easily develop a frontend to display the info as a card or other format than just plain text.

Using structured outputs simplifies downstream processing, and in this case, we can use the data types to ensure we have a consistent output for downstream processing or to send to the frontend to display.

```python
@type_subscription(topic_type="destination_info")
class DestinationAgent(RoutedAgent):
    def __init__(self, model_client: AzureOpenAIChatCompletionClient) -> None:
        super().__init__("DestinationAgent")
        self._system_messages: List[LLMMessage] = [
            SystemMessage("You are a helpful AI assistant that helps with destination information.")
        ]
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: EndUserMessage, ctx: MessageContext) -> None:
        response_content = await self._model_client.create(
            [
                UserMessage(
                    content=f"Provide info for {message.content}",
                    source="DestinationAgent",
                )
            ],
            extra_create_args={"response_format": DestinationInfo},
        )
        destination_info_structured = DestinationInfo.model_validate(
            json.loads(response_content.content)
        )
        await self.publish_message(
            AgentStructuredResponse(
                agent_type=self.id.type,
                data=destination_info_structured,
                message=message.content,
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )
```

#### Integrate Existing Agents

Autogen is extendible and supports agents built using different frameworks. Let's look at how to bring in an existing LLama Index agent that uses Wikipedia as a tool.

```python
@type_subscription("default_agent")
class LlamaIndexAgent(RoutedAgent):
    def __init__(self, llama_index_agent: AgentRunner, memory: Optional[BaseMemory] = None) -> None:
        super().__init__("LlamaIndexAgent")
        self._llama_index_agent = llama_index_agent
        self._memory = memory

    @message_handler
    async def handle_user_message(self, message: EndUserMessage, ctx: MessageContext) -> None:
        self._session_id = ctx.topic_id.source

        # Retrieve historical messages if memory is available
        history_messages: List[ChatMessage] = []
        if self._memory:
            history_messages = self._memory.get(input=message.content)

        # Get response from LlamaIndex agent
        response = await self._llama_index_agent.achat(
            message=message.content,
            history_messages=history_messages
        )

        # Store messages in memory if available
        if self._memory:
            self._memory.put(ChatMessage(role=MessageRole.USER, content=message.content))
            self._memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response.response))

        # Compile resources from response
        resources = [
            Resource(content=source_node.get_text(), score=source_node.score, node_id=source_node.id_)
            for source_node in response.source_nodes
        ]

        # Publish response message
        await self.publish_message(
            AgentStructuredResponse(
                agent_type="default_agent",
                data=None,
                message=f"\n{response.response}\n",
            ),
            DefaultTopicId(type="user_proxy", source=self._session_id),
        )
```

#### Planning Agent

A planning agent or intent detection agent can determine user intent and respond accordingly. With structured output, you can create a list of subtasks and assign them to the appropriate agent.

Below is a sample data model you can use for planning:

```python
# Enum to Define Agent Types
class AgentEnum(str, Enum):
    FlightBooking = "flight_booking"
    HotelBooking = "hotel_booking"
    CarRental = "car_rental"
    ActivitiesBooking = "activities_booking"
    DestinationInfo = "destination_info"
    DefaultAgent = "default_agent"
    GroupChatManager = "group_chat_manager"

# Travel SubTask Model
class TravelSubTask(BaseModel):
    task_details: str
    assigned_agent: AgentEnum

    class Config:
        use_enum_values = True  # To serialize enums as their values

# Travel Plan Model
class TravelPlan(BaseModel):
    main_task: str
    subtasks: List[TravelSubTask]
    is_greeting: bool
```

#### Managing Simple vs. Complex Tasks

In real-world scenarios, optimizing user experience by responding quickly is crucial. For simple tasks that a single agent can handle effectively, ensure they are routed appropriately.

- **Simple Task**: A FlightAgent handles a specific flight query.
- **Complex Task**: A TravelRouterAgent delegates tasks like booking flights, hotels, and car rentals to specialized agents.

To manage this, we use a SemanticRouterAgent to route messages effectively.

```python
@type_subscription(topic_type="router")
class SemanticRouterAgent(RoutedAgent):
    def __init__(self, name: str, model_client: AzureOpenAIChatCompletionClient, agent_registry: AgentRegistry, session_manager: SessionStateManager) -> None:
        super().__init__("SemanticRouterAgent")
        self._name = name
        self._model_client = model_client
        self._registry = agent_registry
        self._session_manager = session_manager

    @message_handler
    async def route_message(self, message: EndUserMessage, ctx: MessageContext) -> None:
        session_id = ctx.topic_id.source

        # Add the current message to session history
        self._session_manager.add_to_history(session_id, message)

        # Analyze conversation history for better context
        history = self._session_manager.get_history(session_id)
        travel_plan: TravelPlan = await self._get_agents_to_route(message, history)

        if travel_plan.is_greeting:
            logger.info("User greeting detected")
            await self.publish_message(
                AgentStructuredResponse(
                    agent_type="default_agent",
                    data=Greeter(
                        greeting="Greetings, Adventurer! 🌍 Let's get started!"
                    ),
                    message=f"User greeting detected: {message.content}",
                ),
                DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
            )
            return
```

#### Expanding with Additional Agents

Add more agents dedicated to tasks like flight booking, car rental, and hotel booking. Specialized agents improve system performance by distributing workload and making individual components more efficient. Defining corresponding data types for each agent facilitates better communication, making the system easier to extend and maintain.

#### Logging and Using the Aspire Dashboard

Use logging to trace messages and troubleshoot issues during development. Autogen provides built-in support for exporting logs to the Aspire dashboard via OpenTelemetry.

#### Deploying with FastAPI and Azure Container Apps

One common way to interact with chatbots is via WebSockets, which help with real-time message sending and receiving. You can create a WebSocket manager to track connections and respond to users.

```python
class WebSocketConnectionManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    def add_connection(self, session_id: str, websocket: WebSocket) -> None:
        self.connections[session_id] = websocket

    async def handle_websocket(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.add_connection(session_id, websocket)
        try:
            while True:
                user_message_text = await websocket.receive_text()
                chat_id = str(uuid.uuid4())
                user_message = EndUserMessage(content=user_message_text, source="User")
                logger.info(f"Received message with chat_id: {chat_id}")
                # Publish the user's message to the agent
                await agent_runtime.publish_message(
                    user_message, DefaultTopicId(type="user_proxy", source=session_id)
                )
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Exception in WebSocket connection {session_id}: {str(e)}")

connection_manager = WebSocketConnectionManager()

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for handling user chat messages.

    Args:
        websocket (WebSocket): The WebSocket connection.
    """
    session_id = str(uuid.uuid4())
    await connection_manager.handle_websocket(websocket, session_id)
```

Azure Container Apps is a serverless platform designed to run containerized applications and microservices without the need to manage complex infrastructure.

#### Microsoft Teams Integration

In addition to the backend AutoGen implementation described above, this project includes a Microsoft Teams integration layer using the Bot Framework. This enables users to interact with the travel assistant directly within Teams, making it easily accessible in a familiar collaboration environment.

**Teams Features:**

- Real-time messaging through Bot Framework integration
- Interactive suggested actions for common queries
- Structured message formatting for travel information
- Persistent WebSocket connections for real-time responses
- Emoji support and rich text formatting

The Teams integration seamlessly connects to the AutoGen backend through WebSockets, maintaining the sophisticated agent system while providing a user-friendly interface in Teams. Users can access all travel planning features, destination information, and booking assistance without leaving their Teams workspace.

**Code Overview**

```python
# 1. Teams Bot - Handles user interactions
class MyBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        # Forward to AutoGen backend via WebSocket
        if self.ws_handler:
            await self.ws_handler.send_message(turn_context.activity.text)
        # Show interactive buttons
        await turn_context.send_activity(self.get_suggested_actions())

# 2. WebSocket Handler - Manages real-time communication
class WebSocketHandler:
    async def process_websocket_message(self, message: str):
        message_data = json.loads(message)
        formatted_text = self.message_formatter.format_message(message_data)
        await self.bot_handler.send_response(formatted_text)

# 3. Message Formatter - Structures responses for Teams display
class MessageFormatter:
    def format_destination_info(self, data: dict) -> str:
        info = data.get('data', {})
        return f"""🌏 {info.get('city')}, {info.get('country')}
                  📝 {info.get('description')}
                  ⏰ Best Time: {info.get('best_time_to_visit')}"""
```

This guide provides a comprehensive overview of setting up a multi-agent system using AutoGen Core API, integrating it with Microsoft Teams, and deploying it using FastAPI and Azure Container Apps. By following these steps, developers can create a robust and scalable travel planning assistant that meets modern travelers' needs.

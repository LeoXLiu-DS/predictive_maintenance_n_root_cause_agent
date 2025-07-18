from typing import Any, Generator, Optional, Sequence, Union
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.chat_models.databricks import ChatDatabricks
from langgraph.checkpoint.memory import InMemorySaver
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from databricks.vector_search.client import VectorSearchClient
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.langchain.chat_agent_langgraph import parse_message
from langgraph.graph.state import CompiledStateGraph

import mlflow

mlflow.langchain.autolog()

mlflow.set_registry_uri("databricks-uc")

# Configure
CATALOG = "workspace"
SCHEMA = "genai_demo"

# Anomaly Detection Model
MODEL_NAME = "isolation_forest_pm_model"
MODEL_NAME_FULL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME  }"
MODEL_VERSION = 1
MODEL_URI = f'models:/{MODEL_NAME_FULL}/{MODEL_VERSION}'

# Vector Index
INDEX_NAME = "maintenance_docs_index"
INDEX_NAME_FULL = f"{CATALOG}.{SCHEMA}.{INDEX_NAME}"

# LLM
LLM_MODEL = "gpt-41"
TEMPERATURE = 0.1
# LLM_MODEL = "databricks-llama-4-maverick"


# Load resources: model, retriever, LLM
ad_model = mlflow.sklearn.load_model(MODEL_URI)

vsc = VectorSearchClient()
index = vsc.get_index(index_name=INDEX_NAME_FULL)

# ws = WorkspaceClient()
# chat_client = ws.serving_endpoints.get_open_ai_client()
llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=LLM_MODEL,
    temperature=TEMPERATURE,
)


# Define tools
@tool
def anomaly_detector(vibration: float, pressure: float, temperature: float) -> str:
    """
    Detects anomalies in equipment behavior using vibration, pressure, and temperature.
    """
    try:
        prediction = ad_model.predict([[vibration, pressure, temperature]])
        result = "Anomalous" if prediction[0] == -1 else "Normal"
        return f"Anomaly Detection Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def vector_search(query: str) -> str:
    """
    Searches the vector index for machine manual documents."""
    try:
        # Search the index with the query string
        res = index.similarity_search(
            query_text=query,
            columns=["chunk_text"],
            num_results=1,
            query_type="hybrid"
            )
        context = "\n\n".join([r[0] for r in res["result"]["data_array"]])
        return context
    except Exception as e:
        return f"Vector search error: {str(e)}"


tools = [anomaly_detector, vector_search]

# Define Nodes
system_prompt = SystemMessage(
    content=(
        "You are a predictive maintenance engineer. Answer machine maintenance queries using the search index. "
        "If sensor data is provided, use the anomaly detection tool. "
        "If the machine is anomalous, ask user whether RCA and resolution is required if user does not suggest anything otherwise continue the task."
    )
)


# Add memory
checkpointer = InMemorySaver()

def create_agent(llm, tools, system_prompt):

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    llm_with_tools = llm.bind_tools(tools)

    # assistant node
    def assistant_node(state: AgentState) -> AgentState:
        msgs = state["messages"]
        # Prepend system prompt if first turn
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [system_prompt] + msgs

        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}
    
    # Tools node
    tools_node = ToolNode(tools)

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )
    builder.add_edge("tools", "assistant")

    # agent = builder.compile(checkpointer=checkpointer)
    agent = builder.compile()

    return agent


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self, 
        messages: list[ChatAgentMessage], 
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}
        res = self.agent.invoke(request)
        response = [ChatAgentMessage(**parse_message(r)) for r in res["messages"]]
        return ChatAgentResponse(messages=response)
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                for m in node_data.get("messages", []):
                    msg = parse_message(m)
                    yield ChatAgentChunk(delta=ChatAgentMessage(**msg))


pm_agent = create_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(pm_agent)
mlflow.models.set_model(AGENT)

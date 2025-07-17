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
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

import mlflow

mlflow.langchain.autolog()

# Configure
catalog = "workspace"
schema = "genai_demo"
model_name = "isolation_forest_pm_model"
model_version = 1
AD_MODEL = f'models:/{model_name}/{model_version}'
VECTOR_INDEX = "workspace.genai_demo.maintenance_docs_index"
# EMBEDDING_MODEL = "databricks-gte-large-en"
# LLM_MODEL = "databricks-llama-4-maverick"
LLM_MODEL = "gpt-41"
# LLM_MODEL = "databricks-gemma-3-12b"
# LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"


# Load resources: model, retriever, LLM
ad_model = mlflow.sklearn.load_model(AD_MODEL)

# vsc = VectorSearchClient()
# index = vsc.get_index(index_name=VECTOR_INDEX)  # adjust catalog/schema

# ws = WorkspaceClient()
# chat_client = ws.serving_endpoints.get_open_ai_client()
llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=LLM_MODEL,
    temperature=0.1,
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
        # results = index.similarity_search(query)
        # return "\n".join([str(res) for res in results])
        return "The machine's bearings are wore down and need to be replaced."
    except Exception as e:
        return f"Vector search error: {str(e)}"


tools = [anomaly_detector, vector_search]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define Nodes
system_prompt = SystemMessage(
    content=(
        "You are a predictive maintenance engineer. Answer machine maintenance queries using the search index. "
        "If sensor data is provided, use the anomaly detection tool. "
        "If the machine is anomalous, ask user whether RCA and resolution is required if user does not suggest anything otherwise continue the task."
    )
)

def assistant_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    # Prepend system prompt if first turn
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [system_prompt] + msgs

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


# Tools node for execution
tools_node = ToolNode(tools)

# Add memory
checkpointer = InMemorySaver()

def create_agent():
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


# class LangGraphChatAgent(ChatAgent):
#     def __init__(self, agent: CompiledStateGraph):
#         self.agent = agent

#     def predict(
#         self,
#         messages: list[ChatAgentMessage],
#         context: Optional[ChatContext] = None,
#         custom_inputs: Optional[dict[str, Any]] = None,
#     ) -> ChatAgentResponse:
#         request = {"messages": self._convert_messages_to_dict(messages)}

#         messages = []
#         for event in self.agent.stream(request, stream_mode="updates"):
#             for node_data in event.values():
#                 messages.extend(
#                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
#                 )
#         return ChatAgentResponse(messages=messages)

#     def predict_stream(
#         self,
#         messages: list[ChatAgentMessage],
#         context: Optional[ChatContext] = None,
#         custom_inputs: Optional[dict[str, Any]] = None,
#     ) -> Generator[ChatAgentChunk, None, None]:
#         request = {"messages": self._convert_messages_to_dict(messages)}
#         for event in self.agent.stream(request, stream_mode="updates"):
#             for node_data in event.values():
#                 yield from (
#                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
#                 )

pm_agent = create_agent()
mlflow.models.set_model(pm_agent)
# AGENT = LangGraphChatAgent(pm_agent)
# mlflow.models.set_model(AGENT)

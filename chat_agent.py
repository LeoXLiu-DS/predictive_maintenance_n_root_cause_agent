import os
from typing import Any, Generator, Optional, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from databricks_langchain import ChatDatabricks
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import parse_message
from databricks.vector_search.client import VectorSearchClient
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import mlflow

os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"

mlflow.langchain.autolog()

mlflow.set_registry_uri("databricks-uc")

# Configure
CATALOG = "workspace"
SCHEMA = "genai_demo"
MODEL_NAME = "isolation_forest_pm_model"
MODEL_NAME_FULL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME  }"
MODEL_VERSION = 1
MODEL_URI = f'models:/{MODEL_NAME_FULL}/{MODEL_VERSION}'
INDEX_NAME = "maintenance_docs_index"
INDEX_NAME_FULL = f"{CATALOG}.{SCHEMA}.{INDEX_NAME}"
LLM_MODEL = "gpt-41"
TEMPERATURE = 0.1
# LLM_MODEL = "databricks-llama-4-maverick"


# Load resources: model, retriever, LLM
ad_model = mlflow.sklearn.load_model(MODEL_URI)
vsc = VectorSearchClient()
index = vsc.get_index(index_name=INDEX_NAME_FULL)
llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=LLM_MODEL,
    temperature=TEMPERATURE,
)


# Define tools
@tool
def anomaly_detector(temperature: float, vibration: float, pressure: float) -> str:
    """
    Detects anomalies in equipment behavior using vibration, pressure, and temperature.
    """
    try:
        prediction = ad_model.predict([[temperature, vibration, pressure]])
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
        "You are a maintenance engineer tasked with addressing machine maintenance queries, conducting root cause analysis (RCA), and recommending appropriate resolutions." "Use the search index to retrieve relevant information that supports accurate and context-aware responses."
        "If sensor data is provided, analyze it using the anomaly detection tool to assess whether the machine is operating normally or exhibiting anomalous behavior. "
        "If an anomaly is detected and the user has not provided specific instructions, prompt them to confirm whether they would like to proceed with RCA and resolution. If instructions are provided, continue with the assigned task accordingly."
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

    agent = builder.compile(checkpointer=checkpointer)
    # agent = builder.compile()

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
        # res = self.agent.invoke(request)
        if custom_inputs and 'configurable' in custom_inputs:
            config = {"configurable": custom_inputs["configurable"]}
        else: 
            config = {"configurable": {'thread_id':99}}
        res = self.agent.invoke(request, config)

        response = [ChatAgentMessage(**parse_message(r)) for r in res["messages"]]
        return ChatAgentResponse(messages=response)
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        if custom_inputs and 'configurable' in custom_inputs:
            config = {"configurable": custom_inputs["configurable"]}
        else: 
            config = {"configurable": {'thread_id':1}}
        for event in self.agent.stream(request, config, stream_mode="updates"):
            for node_data in event.values():
                for m in node_data.get("messages", []):
                    msg = parse_message(m)
                    yield ChatAgentChunk(delta=ChatAgentMessage(**msg))


chat_pm_agent = create_agent(llm, tools, system_prompt)
CHAT_AGENT = LangGraphChatAgent(chat_pm_agent)
mlflow.models.set_model(CHAT_AGENT)

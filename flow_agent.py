# Import dependencies
from typing import Any, Optional
import uuid
import json
from datetime import datetime

from pydantic import BaseModel
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import ChatDatabricks

mlflow.langchain.autolog()

# Configure
CATALOG = "workspace"
SCHEMA = "genai_demo"
MODEL_NAME = "isolation_forest_pm_model"
MODEL_VERSION = 1
MODEL_NAME_FULL = f"models:/{CATALOG}.{SCHEMA}.{MODEL_NAME}/{MODEL_VERSION}"
INDEX_NAME = "maintenance_docs_index"
INDEX_NAME_FULL = f"{CATALOG}.{SCHEMA}.{INDEX_NAME}"
# LLM_MODEL = "databricks-llama-4-maverick"
LLM_MODEL = "gpt-41"
TEMPERATURE = 0.1


# Define State Schema
class AgentState(BaseModel):
    timestamp: datetime
    machine_id: int
    temperature: float
    vibration: float
    pressure: float
    # Normal operating ranges
    normal_temp: tuple[float, float] = (20, 36)
    normal_vibration: tuple[float, float] = (1, 2.2)
    normal_pressure: tuple[float, float] = (2, 4.5)
    # RCA logs
    is_anomaly: bool = False
    raw_query: str = ""
    query: str = ""
    context: str = ""
    suggestion: str = ""


# Load trained model from MLflow
ad_model = mlflow.sklearn.load_model(MODEL_NAME_FULL)

# Initialize vector retriever
vsc = VectorSearchClient()
index = vsc.get_index(index_name=INDEX_NAME_FULL)

# Initialize LLM (Maverick)
llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=LLM_MODEL,
    temperature=TEMPERATURE,
)


# Define Nodes
def detect_anomaly(state: AgentState) -> dict:
    X = [[state.temperature, state.vibration, state.pressure]]
    state.is_anomaly = ad_model.predict(X)[0] == -1
    return {"is_anomaly": state.is_anomaly}


def query_otimization(state: AgentState):
    raw_query = (f"Machine {state.machine_id} anomaly: T={state.temperature} [normal {state.normal_temp[0] - state.normal_temp[1]}], "
                 f"V={state.vibration} [normal {state.normal_vibration[0]} - {state.normal_vibration[1]}], "
                 f"P={state.pressure} [normal {state.normal_pressure[0]} - {state.normal_pressure[1]}]")
    
    q_opt_msg = [
        {"role":"system","content":"Rewrite the following to a concise, technical search query focusing on deviation from normal operation."},
        {"role":"user","content":raw_query}
    ]

    q_opt = llm.invoke(q_opt_msg).content
    return {"raw_query": raw_query, "query": q_opt}

def vector_search(state: AgentState):
    hits = index.similarity_search(query_text=state.query, columns=["chunk_text"], num_results=2, query_type="hybrid")
    context = "\n\n".join(hit[0] for hit in hits["result"]["data_array"])
    return {"context": context}

def rca(state: AgentState):
    prompt = [
        {"role":"system","content":"You're an engineer analyzing machinery anomalies."},
        {"role":"user","content":
         f"Anomaly details:\n{state.raw_query}\n\nContext:\n{state.context}\n\nProvide root cause and maintenance actions."}
    ]
    response = llm.invoke(prompt)
    suggestion = response.content
    return {"suggestion": suggestion}



def normal(state: AgentState) -> dict:
    suggestion = "✅ Machine is operating properly."
    state.suggestion = suggestion
    return {"suggestion": "✅ Machine is operating properly."}

def create_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("detect_anomaly", detect_anomaly)
    workflow.add_node("query_optimization", query_otimization)
    workflow.add_node("vector_search", vector_search)
    workflow.add_node("rca", rca)

    workflow.add_edge(START, "detect_anomaly")
    workflow.add_conditional_edges("detect_anomaly",
        lambda s: "query_optimization" if s.is_anomaly else "normal",
        {"query_optimization":"query_optimization", "normal": "normal"}
    )
    workflow.add_node("normal", normal)
    workflow.add_edge("normal", END)
    workflow.add_edge("query_optimization", "vector_search")
    workflow.add_edge("vector_search", "rca")
    workflow.add_edge("rca", END)

    agent = workflow.compile()
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
        messages = self._convert_messages_to_dict(messages)
        request = json.loads(messages[-1]['content'])
        response = self.agent.invoke(request)
        return ChatAgentResponse(messages=[ChatAgentMessage(role="assistant", content=response["suggestion"], id=str((uuid.uuid4())))])
        # return [ChatAgentMessage(role="assistant", content=response["suggestion"], id=str((uuid.uuid4())))]

flow_agent = create_agent()
FLOW_AGENT = LangGraphChatAgent(flow_agent)
mlflow.models.set_model(FLOW_AGENT)

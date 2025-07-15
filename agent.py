from langgraph.graph import StateGraph, START, END
from typing import Literal, Any, Generator, Optional, Sequence, Union
from pydantic import BaseModel
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import pickle
from sklearn.ensemble import IsolationForest
import mlflow
from datetime import datetime

from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext
from langgraph.graph.state import CompiledStateGraph
import uuid
import json

mlflow.langchain.autolog()


# Configure
catalog = "workspace"
schema = "genai_demo"
model_name = "isolation_forest_pm_model"
model_version = 4
AD_MODEL = f"models:/{catalog}.{schema}.{model_name}/{model_version}"
VECTOR_INDEX = "workspace.genai_demo.maintenance_docs_index"
EMBEDDING_MODEL = "databricks-gte-large-en"
LLM_MODEL = "databricks-llama-4-maverick"


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
    query: str = ""
    context: str = ""
    suggestion: str = ""


# Load resources for model, retriever, LLM
# Load trained model from MLflow
ad_model = mlflow.sklearn.load_model(AD_MODEL)

# Initialize vector retriever
vsc = VectorSearchClient()
index = vsc.get_index(index_name=VECTOR_INDEX)  # adjust catalog/schema

# Initialize LLM (Maverick)
ws = WorkspaceClient()
chat_client = ws.serving_endpoints.get_open_ai_client()


# Define Nodes
def detect_anomaly(state: AgentState) -> dict:
    X = [[state.temperature, state.vibration, state.pressure]]
    state.is_anomaly = ad_model.predict(X)[0] == -1
    return {"is_anomaly": state.is_anomaly}

def rca_with_query_optimization(state: AgentState):
    # 1. Optimize query
    raw_query = (f"Machine {state.machine_id} anomaly: T={state.temperature} [normal {state.normal_temp[0] - state.normal_temp[1]}], "
                 f"V={state.vibration} [normal {state.normal_vibration[0]} - {state.normal_vibration[1]}], "
                 f"P={state.pressure} [normal {state.normal_pressure[0]} - {state.normal_pressure[1]}]")
    q_opt_msg = [
        {"role":"system","content":"Rewrite the following to a concise, technical search query focusing on deviation from normal operation."},
        {"role":"user","content":raw_query}
    ]
    q_opt = chat_client.chat.completions.create(model=LLM_MODEL, messages=q_opt_msg).choices[0].message.content
    state.query = q_opt

    # 2. Retrieve relevant documents
    hits = index.similarity_search(query_text=q_opt, columns=["chunk_text"], num_results=2, query_type="hybrid")
    context = "\n\n".join(hit[0] for hit in hits["result"]["data_array"])
    state.context = context
    # 3. Generate root cause & action
    prompt = [
        {"role":"system","content":"You're an engineer analyzing machinery anomalies."},
        {"role":"user","content":
         f"Anomaly details:\n{raw_query}\n\nContext:\n{context}\n\nProvide root cause and maintenance actions."}
    ]
    response = chat_client.chat.completions.create(model=LLM_MODEL, messages=prompt)
    suggestion = response.choices[0].message.content
    state.suggestion = suggestion
    return {"suggestion": suggestion}


def normal(state: AgentState) -> dict:
    suggestion = "✅ Machine is operating properly."
    state.suggestion = suggestion
    return {"suggestion": "✅ Machine is operating properly."}



# Build the LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("detect_anomaly", detect_anomaly)
workflow.add_node("rca", rca_with_query_optimization)
workflow.add_edge(START, "detect_anomaly")
workflow.add_conditional_edges("detect_anomaly",
    lambda s: "rca" if s.is_anomaly else "normal",
    {"rca":"rca", "normal": "normal"}
)
workflow.add_node("normal", normal)
workflow.add_edge("normal", END)
workflow.add_edge("rca", END)

agent = workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self, 
        messages: list[ChatAgentMessage], 
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None
    ) -> ChatAgentResponse:
        messages = self._convert_messages_to_dict(messages)
        try:
            input = json.loads(messages[0]["content"])
            result = self.agent.invoke(input)["suggestion"]
            outputs = [ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content=result)]
        except Exception as e:
            outputs = [ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content=str(e))]
        return ChatAgentResponse(messages=outputs)

AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)







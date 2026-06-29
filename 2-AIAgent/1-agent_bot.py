from typing import List,TypedDict
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv


load_dotenv()

class AgentState(TypedDict):
    message : List[HumanMessage]

llm = ChatGroq(model = "qwen/qwen3-32b")

def process(state:AgentState)->AgentState:

    response = llm.invoke(state["message"])
    print(f"\n AI : {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("enter:")
agent.invoke({"message":[HumanMessage(content= user_input)]})
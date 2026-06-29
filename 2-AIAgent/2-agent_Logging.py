from typing import List,TypedDict, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from urllib3 import response

load_dotenv()

class AgentState(TypedDict):
    message : List[Union[HumanMessage,AIMessage]]

llm = ChatGroq(model = "qwen/qwen3-32b")

def process(state:AgentState)->AgentState:
    """ this will solve user request """
    response = llm.invoke(state["message"])

    state["message"].append(AIMessage(content = response.content))
    print(f'\n {response.content}')
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []
user_input = input("Enter: ")

while user_input!= 'exit':
    conversation_history.append(HumanMessage(content = user_input))
    result = agent.invoke({"message" : conversation_history})

    print(result['message'])
    conversation_history = result["message"]

    user_input = input("enter : ")

with open("logging.txt", "w", encoding="utf-8") as file:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")

        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")

    file.write("\n--- End of Conversation ---\n")

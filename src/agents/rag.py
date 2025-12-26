from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
import random

llm = init_chat_model("openai:gpt-4o", temperature=0)

file_search_tool = {
    "type": "file_search",
    "vector_store_ids": ["vs_694ecf74ead881919d6ceea04b26a207"],
}

llm = llm.bind_tools([file_search_tool])


class State(MessagesState):
    customer_name: str
    my_age: int


def extractor(state: State):
    return {}


def conversation(state: State):
    new_state: State = {}

    if state.get("customer_name") is None:
        new_state["customer_name"] = "Jhon Doe"
    else:
        new_state["my_age"] = random.randint(20, 40)

    history = state["messages"]
    last_message = history[-1] if history else AIMessage(content="")
    ai_message = llm.invoke(last_message.text)
    new_state["messages"] = [ai_message]

    return new_state


builder = StateGraph(State)
builder.add_node("conversation", conversation)
builder.add_node("extractor", extractor)

builder.add_edge(START, "extractor")
builder.add_edge("extractor", "conversation")
builder.add_edge("conversation", END)

agent = builder.compile()

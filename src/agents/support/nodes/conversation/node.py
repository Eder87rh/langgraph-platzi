from agents.support.state import State
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
from agents.support.nodes.conversation.tools import tools
from agents.support.nodes.conversation.prompt import SYSTEM_PROMPT

llm = init_chat_model("openai:gpt-4o", temperature=0)


llm = llm.bind_tools(tools)


def conversation(state: State):
    new_state: State = {}

    history = state["messages"]
    last_message = history[-1] if history else AIMessage(content="")
    customer_name = state.get("customer_name", "Jhon Doe")
    ai_message = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("user", last_message.text)
    ])
    new_state["messages"] = [ai_message]

    return new_state

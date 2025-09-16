import json
import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

system_prompt = """You are an AI assistant for car repair manuals. Use the retriever tool to answer questions, cite sources from documents."""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_agent(llm, retriever):
    @tool
    def retriever_tool(query: str) -> str:
        """
        This tool searches and returns the information from the car repair manuals documents.
        """
        docs = retriever.invoke(query)

        if not docs:
            return "I found no relevant information in the car repair manuals."

        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            results.append(f"Document {i+1} from {os.path.basename(source)}:\n{doc.page_content}")

        return "\n\n".join(results)

    llm_with_tools = llm.bind_tools([retriever_tool])

    def call_llm(state):
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm_with_tools.invoke(messages)
        return {"messages": [message]}

    def should_continue(state):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return True
        return False

    def take_action(state):
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls
        tool_messages = []
        for tool_call in tool_calls:
            result = retriever_tool.invoke(tool_call["args"])
            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                content=str(result)
            )
            tool_messages.append(tool_message)
        return {"messages": tool_messages}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    return graph.compile()

def run_agent(user_input: str, session_id="default", retriever=None, llm=None):
    if not llm:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    if not retriever:
        raise ValueError("Retriever required")
    graph = create_agent(llm, retriever)
    history_file = f"chat_history_{session_id}.json"
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    history = history[-20:] if len(history) > 20 else history
    history_messages = []
    for msg_dict in history:
        msg_type = msg_dict.get('type')
        if msg_type == 'human':
            history_messages.append(HumanMessage(**msg_dict))
        elif msg_type == 'ai':
            history_messages.append(AIMessage(**msg_dict))
        elif msg_type == 'tool':
            history_messages.append(ToolMessage(**msg_dict))
        else:
            print(f"Skipping unknown message type: {msg_type}")
            continue
    initial_messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=user_input)]
    result = graph.invoke({"messages": initial_messages})
    new_human = HumanMessage(content=user_input).dict()
    new_ai = result['messages'][-1].dict()
    full_history = history + [new_human, new_ai]
    with open(history_file, 'w') as f:
        json.dump(full_history, f)
    return result['messages'][-1].content
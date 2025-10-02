import json
import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langsmith import traceable

try:
    import tiktoken
except ImportError:
    tiktoken = None

system_prompt = """You are an AI assistant for car repair manuals. Use the retriever tool to answer questions, cite sources from documents."""

def truncate_history_by_tokens(history, max_tokens=2000, model="gpt-5-nano"):
    """
    Keep only recent history that fits within token budget.
    
    Args:
        history: List of message dictionaries
        max_tokens: Maximum tokens to keep in history
        model: Model name for token encoding
    
    Returns:
        Truncated history list
    """
    if not tiktoken or max_tokens <= 0 or not history:
        return history
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    truncated = []
    
    # Process from most recent backwards
    for msg_dict in reversed(history):
        msg_content = msg_dict.get('content', '')
        msg_tokens = len(encoding.encode(str(msg_content)))
        
        if total_tokens + msg_tokens > max_tokens:
            break
        
        truncated.insert(0, msg_dict)
        total_tokens += msg_tokens
    
    return truncated

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_agent(llm, retriever):
    @traceable(name="retriever_tool_call")
    def _retrieve_with_trace(query: str):
        """Wrapped retrieval with tracing metadata."""
        docs = retriever.invoke(query)
        return docs
    
    @tool
    def retriever_tool(query: str) -> str:
        """
        This tool searches and returns the information from the car repair manuals documents.
        """
        docs = _retrieve_with_trace(query)

        if not docs:
            return "I found no relevant information in the car repair manuals."

        results = []
        for i, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            source = meta.get('source', 'Unknown')
            heading_path = meta.get('heading_path') or meta.get('heading')
            page_start = meta.get('page_start')
            page_end = meta.get('page_end')
 
            # Compact format: "Doc N [source, p.X]: content"
            fname = os.path.basename(source) if source else "Unknown"
            page_info = ""
            if page_start is not None and page_end is not None and page_start != page_end:
                page_info = f"pp.{page_start}-{page_end}"
            elif page_start is not None or page_end is not None:
                p = page_start if page_start is not None else page_end
                page_info = f"p.{p}"
            
            # Build compact header
            header_parts = [fname]
            if page_info:
                header_parts.append(page_info)
            if heading_path:
                header_parts.append(str(heading_path))
            
            header = ", ".join(header_parts)
            results.append(f"Doc {i+1} [{header}]:\n{doc.page_content}")
 
        return "\n\n".join(results)

    llm_with_tools = llm.bind_tools([retriever_tool])

    def call_llm(state):
        # System prompt is already in state["messages"] from run_agent() initialization
        # No need to add it again here - this was causing duplicate system prompts
        messages = list(state["messages"])
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
        llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    if not retriever:
        raise ValueError("Retriever required")
    graph = create_agent(llm, retriever)
    history_file = f"chat_history_{session_id}.json"
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    # Apply configurable message-based truncation
    max_messages = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
    history = history[-max_messages:] if len(history) > max_messages else history
    
    # Apply token-based truncation (enabled by default with 2500 token limit)
    max_tokens = int(os.getenv("MAX_HISTORY_TOKENS", "2500"))
    if max_tokens > 0:
        model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
        history = truncate_history_by_tokens(history, max_tokens, model_name)
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
    # Configure LangGraph recursion limit (default 25). Allow override via env.
    _rec_str = os.getenv("RECURSION_LIMIT") or os.getenv("LANGGRAPH_RECURSION_LIMIT") or "25"
    try:
        _rec_limit = max(1, int(_rec_str))
    except Exception:
        _rec_limit = 50
    
    try:
        result = graph.invoke({"messages": initial_messages}, config={"recursion_limit": _rec_limit})
    except GraphRecursionError as e:
        # Catch LangGraph recursion limit error and provide helpful context
        raise GraphRecursionError(
            f"The agent exceeded the maximum number of reasoning steps ({_rec_limit}) while processing your query. "
            f"This usually happens with complex or ambiguous questions. Please try:\n"
            f"  - Simplifying your question\n"
            f"  - Being more specific about what you're looking for\n"
            f"  - Breaking it into smaller questions\n"
            f"You can also increase the limit by setting RECURSION_LIMIT in your .env file."
            f"If all else fails, the answer to your question is not in the knowledge base."
        ) from e
    
    new_human = HumanMessage(content=user_input).dict()
    new_ai = result['messages'][-1].dict()
    full_history = history + [new_human, new_ai]
    with open(history_file, 'w') as f:
        json.dump(full_history, f)
    return result['messages'][-1].content
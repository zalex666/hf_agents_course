import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from tools import tools, vector_store

load_dotenv()

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
openai_api_key = os.getenv("OPENAI_API_KEY")


# Load system prompt
_PROMPT_FILE_PATH: Path = Path(__file__).with_name("prompt.txt")

try:
    system_prompt: str = _PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
except FileNotFoundError as exc:
    raise FileNotFoundError(
        "The system prompt file 'prompt.txt' was not found. Please ensure it exists in the same directory as 'agent.py'."
    ) from exc

# System message
sys_msg = SystemMessage(content=system_prompt)

# Build graph
def build_graph(provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
    """Build and return a LangGraph agent with the specified LLM provider and model.
    
    Args:
        provider: The LLM provider to use ('openai', 'google', etc.)
        model_name: The specific model name to use
        
    Returns:
        A compiled LangGraph agent
    """
    # Initialize the appropriate LLM based on provider
    if provider == "openai":
        llm = ChatOpenAI(
            model=model_name, 
            temperature=0,
            api_key=openai_api_key
        )
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Define the assistant node function
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Define the retriever node function
    def retriever(state: MessagesState):
        similar = vector_store.similarity_search(state["messages"][0].content)
        if similar:
            example_msg = HumanMessage(content=f"Here is a similar question:\n\n{similar[0].page_content}")
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        return {"messages": [sys_msg] + state["messages"]}

    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()
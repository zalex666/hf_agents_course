import warnings
from bs4 import GuessedAtParserWarning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

import os
from pathlib import Path
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
load_dotenv()
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# -----------------------------------------------------------------------------
# Vector store construction
# -----------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the metadata file used to build the vector store
with open("metadata.jsonl", "r", encoding="utf-8") as jsonl_file:
    json_list = list(jsonl_file)

json_QA = [json.loads(json_str) for json_str in json_list]

# Convert each QA pair to a Document that can be indexed
_documents: list[Document] = []
for sample in json_QA:
    content = f"Question : {sample['Question']}\n\nFinal answer : {sample['Final answer']}"
    metadata = {"source": sample["task_id"]}
    _documents.append(Document(page_content=content, metadata=metadata))

# Create or load the persisted Chroma collection
vector_store: Chroma = Chroma.from_documents(
    documents=_documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection",
)
print("Documents inserted into Chroma collection:", vector_store._collection.count())

# -----------------------------------------------------------------------------
# Tool definitions
# -----------------------------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:  # noqa: D401
    """Multiply two numbers.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The product of *a* and *b*.
    """
    return a * b


@tool
def add(a: int, b: int) -> int:  # noqa: D401
    """Add two numbers.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The sum of *a* and *b*.
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:  # noqa: D401
    """Subtract *b* from *a*.

    Args:
        a: Minuend.
        b: Subtrahend.

    Returns:
        The result of *a - b*.
    """
    return a - b


@tool
def divide(a: int, b: int) -> float:  # noqa: D401
    """Divide *a* by *b*.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The division result.

    Raises:
        ValueError: If *b* equals zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:  # noqa: D401
    """Compute *a* modulo *b*.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The remainder after dividing *a* by *b*.
    """
    return a % b


@tool
def wiki_search(query: str) -> dict[str, str]:
    """Search Wikipedia and return up to two documents.

    Args:
        query: The search query.

    Returns:
        A mapping with a single key ``"wiki_results"`` containing a string of
        concatenated document data separated by ``---`` blocks.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted = "\n\n---\n\n".join(
        [
            (
                f'<Document source="{doc.metadata["source"]}" '
                f'page="{doc.metadata.get("page", "")}"/>'
                f"\n{doc.page_content}\n</Document>"
            )
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted}


@tool
def web_search(query: str) -> dict[str, str]:
    """Search the web via Tavily and return up to three documents.

    Args:
        query: The search query.

    Returns:
        A mapping with a single key ``"web_results"`` containing a string of
        concatenated document data separated by ``---`` blocks.
    """
    search_docs = TavilySearch(max_results=3).invoke(query)
    formatted = "\n\n---\n\n".join(
        [
            (
                f'<Document source="{doc.metadata["source"]}" '
                f'page="{doc.metadata.get("page", "")}"/>'
                f"\n{doc.page_content}\n</Document>"
            )
            for doc in search_docs
        ]
    )
    return {"web_results": formatted}


@tool
def arvix_search(query: str) -> dict[str, str]:
    """Search arXiv and return up to three documents.

    Args:
        query: The search query.

    Returns:
        A mapping with a single key ``"arvix_results"`` containing a string of
        concatenated document data separated by ``---`` blocks.
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted = "\n\n---\n\n".join(
        [
            (
                f'<Document source="{doc.metadata["source"]}" '
                f'page="{doc.metadata.get("page", "")}"/>'
                f"\n{doc.page_content[:1000]}\n</Document>"
            )
            for doc in search_docs
        ]
    )
    return {"arvix_results": formatted}


@tool
def similar_question_search(question: str) -> dict[str, str]:
    """Retrieve semantically similar questions from the vector store.

    Args:
        question: The user-provided question.

    Returns:
        A mapping with a single key ``"similar_questions"`` containing a string
        of concatenated documents separated by ``---`` blocks.
    """
    matched_docs = vector_store.similarity_search(question, k=3)
    formatted = "\n\n---\n\n".join(
        [
            (
                f'<Document source="{doc.metadata["source"]}" '
                f'page="{doc.metadata.get("page", "")}"/>'
                f"\n{doc.page_content[:1000]}\n</Document>"
            )
            for doc in matched_docs
        ]
    )
    return {"similar_questions": formatted}

# -----------------------------------------------------------------------------
# Tool registry
# -----------------------------------------------------------------------------

# Optional: expose the retriever as a standard LangChain retriever tool
retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="Retrieve similar questions from the embedded QA database.",
)

# Exported list of all tools that the agent can access
TOOLS = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
    similar_question_search,
]

# Maintain backward compatibility with the original `tools` variable name
tools = TOOLS  # noqa: N816

__all__ = [
    "TOOLS",
    "tools",
    "vector_store",
    "retriever_tool",
    # Export individual tools for direct import if desired
    "multiply",
    "add",
    "subtract",
    "divide",
    "modulus",
    "wiki_search",
    "web_search",
    "arvix_search",
    "similar_question_search",
] 
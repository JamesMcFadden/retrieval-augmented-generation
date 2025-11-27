"""
This script provides utility functions for the web implementation of the RAG app
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

LLM_URL = ""
LLM_TEMP = 0.0
MODEL = os.environ.get("MODEL", "")  # Enter default model into environment variable
TOKEN = os.environ.get(
    "FMAPI_KEY"
)  # Generate key on OpenAI, store in .env file at top level


def initialize_llm(model: str = MODEL, base_url: str = LLM_URL, temp: float = LLM_TEMP):
    """Creates LLM.

    Args:
        model_name: The model name of the LLM.
        base_url: Address of API endpoint to make requests for model interactions.
        temp: The desired LLM temperature to set context for learning patterns and yield more
              deterministic outputs.

    Returns:
        llm: The LLM to use as the model for the agent.
    """

    api_key = ""
    model = "gpt-4.1"

    try:
        with open(
            "/Users/jamesmcfadden/Documents/retrieval-augmented-generation/src/openai_key.txt",
            "r",
        ) as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            "No OpenAI API key found. Set OPENAI_API_KEY or create openai_key.txt."
        )

    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        openai_api_key=api_key,
        streaming=False,
        temperature=temp,
    )

    return llm


def run_rag_pipeline(query: str) -> str:
    """Handles prompt messages and agent responses.

    Args:
        query: The user's request.

    Returns: The final response from the agent.
    """

    loader = DirectoryLoader(
        "/Users/jamesmcfadden/Documents/retrieval-augmented-generation/data",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    retriever = BM25Retriever.from_documents(docs)
    results = retriever.invoke(query)

    llm = initialize_llm()
    response = llm.invoke(
        f"{query}\nAnswer using the retrieved documents:\n{results}"
    ).content

    return response

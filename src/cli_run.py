"""
This script runs retrieval augmented generation from the command line
"""

import os

import cli_utils as LLM

MODEL = os.environ.get("MODEL", "gpt-oss-120b")


def llm_call():
    """Calls LLM.

    Args: None

    Returns:
        output: The model output of the user's query
    """

    llm = LLM.initialize_llm()

    output = LLM.message_handler(llm)

    return output


def main():
    """
    Executes CLI RAG app and saves results.
    """
    output, query = llm_call()

    with open("results.txt", "a", encoding="utf-8") as file:
        file.write(f"Model: {os.environ['MODEL']}\n")
        file.write(f"Query: {query}\n")
        file.write(f"Output: {output}\n")
        file.write("\n")


if __name__ == "__main__":
    main()

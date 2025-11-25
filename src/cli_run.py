import os
from pathlib import Path
from typing import Union

import cli_utils as LLM

MODEL = os.environ.get("MODEL", "gpt-oss-120b")


def LLM_call():

    llm = LLM.initialize_llm()

    output = LLM.message_handler(llm)

    return output


def main():

    output, query = LLM_call()

    with open("results.txt", "a", encoding="utf-8") as file:
        file.write(f"Model: {os.environ['MODEL']}\n")
        file.write(f"Query: {query}\n")
        file.write(f"Output: {output}\n")
        file.write("\n")


if __name__ == "__main__":
    main()

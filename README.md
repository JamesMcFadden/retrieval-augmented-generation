# retrieval-augmented-generation
This repository implements retrieval augmented generation leveraging OpenAI's GPT model as well as local data.

Code for creating executable:
  pyinstaller --clean --onefile \
  --name RAGflaskapp \
  --add-data "templates:templates" \
  --add-data "openai_key.txt:." \
  app.py

To update executable to new version:
git add .
git commit -m "Bump version to v1.2.0"
# create the tag locally
git tag -a v1.2.0 -m "Release v1.2.0"

# push just that tag to GitHub
git push origin v1.2.0



## Table of contents

- Overview
- Features
- Architecture
- Quickstart
- Configuration
- Usage
    - Indexing documents
    - Querying
- Local development & testing
- Contributing
- License

## Overview

This repository demonstrates a retrieval-augmented generation (RAG) pipeline that combines local document retrieval with an LLM (e.g., OpenAI GPT) to produce informed, context-aware responses. It supports embedding local corpora, indexing into a vector store, and using retrieval results as prompts for generation.

## Features

- Embed local documents and build a searchable vector index
- Retrieve relevant context at query time to augment LLM prompts
- Modular storage backends (FAISS, SQLite vector stores, or managed vector DBs)
- Simple CLI and example Python client for indexing and querying
- Config-driven: switch models, backends, and chunking behavior via environment/config

## Architecture

1. Document ingestion: read files, split into chunks, normalize text.
2. Embedding: compute vector representations with a chosen embeddings model.
3. Vector store: persist embeddings and metadata for nearest-neighbor search.
4. Retriever: given a user query, fetch top-k relevant chunks.
5. Generator: construct an augmented prompt and call the LLM to produce the final answer.

## Quickstart

Prerequisites:
- Python 3.9+
- An OpenAI API key (or other supported model credentials)
- pip

Install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare environment variables (example):
```
export OPENAI_API_KEY="sk-..."
export VECTOR_BACKEND="faiss"   # or pinecone, sqlite, etc.
```

Index documents (example):
```
python scripts/index.py --source ./data --backend faiss --chunk-size 800
```

Query the index:
```
python scripts/query.py --query "How do I set up the project?"
```

## Configuration

- OPENAI_API_KEY: API key for OpenAI requests.
- VECTOR_BACKEND: which vector store to use (faiss, pinecone, sqlite, etc.).
- BACKEND_CONFIG: optional JSON / env-specific settings for managed vector DBs.
- CHUNK_SIZE / OVERLAP: controls document chunking for embeddings.
- MODEL: generative model name (e.g., gpt-4o, gpt-4, gpt-3.5-turbo) and embeddings model.

See config.sample.toml for full options and defaults.

## Usage

Indexing documents (Python example):
```python
from rag.indexer import Indexer

indexer = Indexer(backend="faiss", embedding_model="text-embedding-3-large")
indexer.index_folder("data/")
```

Querying programmatically:
```python
from rag.client import RAGClient

client = RAGClient(model="gpt-4o", backend="faiss")
resp = client.ask("Summarize how to run the tests.")
print(resp.text)
```

CLI examples:
- python scripts/index.py --source ./docs
- python scripts/query.py --query "What is retrieval-augmented generation?"

## Local development & testing

- Run unit tests:
    ```
    pytest
    ```
- Linting & formatting:
    ```
    pre-commit run --all-files
    ```

If you want to develop without external API calls, set a mock embeddings provider in tests or use recorded fixtures.

## Contributing

- Fork the repository, create a feature branch, open a PR.
- Keep changes focused and document behavior changes.
- Add tests for new features and ensure CI passes.

## Troubleshooting

- If embeddings fail, confirm API key and network access.
- If retrieval results are irrelevant, try increasing chunk overlap or using a different embeddings model.
- For large corpora, consider a managed vector DB or disk-backed FAISS indices.

## License

This project is licensed under the MIT License. See LICENSE for details.
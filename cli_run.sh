#!/bin/bash

set -e

echo "Activating virtual environment..."
source RAG/bin/activate

export FLASK_APP=src/app.py
export FLASK_ENV=development

echo "Starting Flask server..."
flask run --host=0.0.0.0 --port=5001
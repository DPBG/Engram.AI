#!/bin/bash
# Initialize Ollama models for ActiveLearningAI

set -e

OLLAMA_HOST="${OLLAMA_URL:-http://localhost:11434}"

echo "Waiting for Ollama to be ready..."
until curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
    echo "Ollama not ready, waiting..."
    sleep 5
done

echo "Ollama is ready. Pulling models..."

# Pull code generation model
echo "Pulling deepseek-coder:6.7b..."
curl -X POST "$OLLAMA_HOST/api/pull" -d '{"name": "deepseek-coder:6.7b"}' || echo "Failed to pull deepseek-coder"

# Pull embedding model
echo "Pulling nomic-embed-text..."
curl -X POST "$OLLAMA_HOST/api/pull" -d '{"name": "nomic-embed-text"}' || echo "Failed to pull nomic-embed-text"

echo "Model initialization complete!"

# List available models
echo "Available models:"
curl -s "$OLLAMA_HOST/api/tags" | python3 -c "import sys, json; data = json.load(sys.stdin); print('\n'.join([m['name'] for m in data.get('models', [])]))"

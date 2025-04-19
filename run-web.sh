#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Export necessary environment variables
export PYTHONPATH=$(pwd)
export OPENCV_AVFOUNDATION_SKIP_AUTH=1

# Run the web application with uvicorn
echo "Starting ScaleTent Web Interface..."
uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload 
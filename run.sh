#!/bin/bash

# Simple script to run main.py
echo "🚀 Starting the pipeline..."
echo
python main.py

# Check if execution succeeded
if [ $? -eq 0 ]; then
    echo
    echo "🍻 Pipeline executed successfully!"
else
    echo
    echo "❌ Error: Pipeline execution failed."
    exit 1
fi

#!/bin/bash
# Run script for Hand Tracking Canvas using Python 3.8

# Path to Python 3.8 executable
PYTHON_PATH="/Library/Frameworks/Python.framework/Versions/3.8/bin/python3"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python 3.8 not found at $PYTHON_PATH"
    echo "Please check your Python installation."
    exit 1
fi

# Run the script
echo "Starting Hand Tracking Canvas..."
"$PYTHON_PATH" hand_tracking_canvas.py

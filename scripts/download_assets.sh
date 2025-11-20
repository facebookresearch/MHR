#!/bin/bash

# Check if assets directory already exists
if [ ! -d "assets" ]; then
    echo "Downloading MHR model assets..."
    curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
    unzip -o assets.zip
    rm assets.zip
    echo "Assets downloaded successfully!"
else
    echo "MHR assets already present."
fi

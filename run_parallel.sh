#!/bin/bash

# Loop through numeric parameters (5 to 9) and run the Python script in the background
for ((param=5; param<=9; param++)); do
    python3.10 main.py "$param" &
done

# Wait for all background processes to finish
wait
#!/bin/bash

mkdir -p samples/out

for input_path in samples/in/*; do
    if [ -f "$input_path" ]; then
        filename=$(basename "$input_path")
        output_path="samples/out/$filename"

        echo "Processing: $input_path -> $output_path"

        python3 src/main.py "$input_path" -o "$output_path"
    fi
done

echo "Done!"

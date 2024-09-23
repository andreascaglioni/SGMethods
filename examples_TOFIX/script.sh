#!/bin/bash

for file in test_*; do
    if [ -f "$file" ]; then
        mv "$file" "old_$file"
        echo "Renamed $file to old_$file"
    fi
done

#!/bin/bash

trap "exit" INT

for FILE in configs/*; do
    if [ -f "$FILE" ]; then
        if [[ "$FILE" == *.json ]]; then
            python train.py "$FILE"
            python evaluate.py "$FILE"
        fi
    fi
done
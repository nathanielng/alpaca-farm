#!/bin/bash

# Configuration with defaults
MAX_PARALLEL="${1:-10}"

# Read stdin into a temporary file
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

cat > "$TEMP_FILE"

# Count number of lines in the temp file
NUM_LINES=$(wc -l < "$TEMP_FILE")

# Check if there's any input
if [ "$NUM_LINES" -eq 0 ]; then
    echo "Error: No input provided!" >&2
    echo "Usage: cat questions.txt | $0 [max_parallel]" >&2
    exit 1
fi

# Determine number of parallel jobs (minimum of NUM_LINES and MAX_PARALLEL)
if [ "$NUM_LINES" -lt "$MAX_PARALLEL" ]; then
    PARALLEL_JOBS=$NUM_LINES
else
    PARALLEL_JOBS=$MAX_PARALLEL
fi

echo "Processing $NUM_LINES questions with $PARALLEL_JOBS parallel jobs..." >&2
echo "" >&2

# Run parallel processing
parallel --keep-order -j "$PARALLEL_JOBS" \
    'echo "**Question**: {}" && echo {} | python strands_websearch.py --engine exa && echo "=============================="' \
    :::: "$TEMP_FILE"

echo "" >&2
echo "Done!" >&2

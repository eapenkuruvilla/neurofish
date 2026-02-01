#!/bin/bash

SF_ENGINE="stockfish"
DEPTH=0

echo "Stockfish Position Evaluator (depth $DEPTH)"
echo "Enter FEN positions to evaluate, or 'quit'/'exit' to end."
echo ""

while true; do
    read -p "FEN> " FEN

    case "${FEN,,}" in
        quit|exit)
            echo "Goodbye!"
            exit 0
            ;;
        "")
            echo "Please enter a FEN or 'quit' to exit."
            continue
            ;;
    esac


    # Use a temp file to capture output
    TMPFILE=$(mktemp)

    # Run stockfish and capture output - use process substitution to avoid subshell issue
    {
        echo "uci"
        echo "isready"
        echo "position fen $FEN"
        echo "eval"
        # Give stockfish time to process
        sleep 2
        echo "quit"
    } | "$SF_ENGINE" 2>/dev/null > "$TMPFILE"

    SF_OUTPUT=$(cat "$TMPFILE")
    rm -f "$TMPFILE"

    NNUE_SCORE=$(echo "$SF_OUTPUT" | grep "^NNUE evaluation" | awk '{print $3}')

    if [ -n "$NNUE_SCORE" ]; then
        echo "NNUE Score: $NNUE_SCORE"
    else
        echo "Could not find a valid evaluation. Check your FEN."
        echo "Debug output:"
        echo "$SF_OUTPUT"
    fi
    echo ""
done

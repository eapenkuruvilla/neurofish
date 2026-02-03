#!/usr/bin/env bash

set -euo pipefail

# ----------------------------
# Argument validation.
# ----------------------------

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <stockfish-elo> <games> [--debug|-debug]"
    exit 1
fi

ELO="$1"
GAMES="$2"

# Validate numeric arguments
if ! [[ "$ELO" =~ ^[0-9]+$ ]]; then
    echo "Error: stockfish-elo must be a positive integer"
    exit 1
fi

if ! [[ "$GAMES" =~ ^[0-9]+$ ]]; then
    echo "Error: games must be a positive integer"
    exit 1
fi

# Optional debug flag
DEBUG_FLAG=""

for arg in "$@"; do
    case "$arg" in
        -debug|--debug)
            DEBUG_FLAG="-debug"
            ;;
        -*)
            # Catch unknown flags early
            if [[ "$arg" != "-debug" && "$arg" != "--debug" ]]; then
                echo "Error: unknown option '$arg'"
                exit 1
            fi
            ;;
    esac
done

# ----------------------------
# Run cutechess
# ----------------------------
CMD_DIR="$(dirname "$0")"
CUTECHESS_PATH="$CMD_DIR"/../../cutechess
OUTFILE_TEMPLATE="/tmp/fileXXXXXX.pgn"
OUTFILE=$(mktemp --dry-run "$OUTFILE_TEMPLATE")
THREADS=6

echo "PGN File: $OUTFILE"
python3 -c "import config"

echo ""$CUTECHESS_PATH"/build/cutechess-cli \
-engine cmd="$CMD_DIR"/../uci.sh name=neurofish ponder option.Threads=$THREADS "$DEBUG_FLAG" \
-engine cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo="$ELO" name=stockfish"$ELO" \
-each proto=uci tc=40/120+1 timemargin=9999 \
-draw movenumber=40 movecount=5 score=50 \
-resign movecount=3 score=500 twosided=false \
-maxmoves 100 -recover -games "$GAMES" \
-pgnout "$OUTFILE""

"$CUTECHESS_PATH"/build/cutechess-cli \
-engine cmd="$CMD_DIR"/../uci.sh name=neurofish ponder option.Threads="$THREADS" "$DEBUG_FLAG" \
-engine cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo="$ELO" name=stockfish"$ELO" \
-each proto=uci tc=40/120+1 timemargin=9999 \
-draw movenumber=40 movecount=5 score=50 \
-resign movecount=3 score=500 twosided=false \
-maxmoves 100 -recover -games "$GAMES" \
-pgnout "$OUTFILE"

echo "PGN File: $OUTFILE"
python3 -c "import config"


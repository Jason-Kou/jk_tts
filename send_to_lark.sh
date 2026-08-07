#!/bin/bash
# send_to_lark.sh - Send generated audio to a Lark group via the amz CLI
# Usage: ./send_to_lark.sh [--dry-run] [file...]
#   no file  -> newest wav in output/
#   LARK_CHAT_ID overrides the target group
#
# Example:
#   ./send_to_lark.sh                              # newest output
#   ./send_to_lark.sh output/zenty_control_en.wav
#   ./send_to_lark.sh --dry-run output/*.wav       # validate, send nothing

set -euo pipefail

cd "$(dirname "$0")"

CHAT_ID="${LARK_CHAT_ID:-oc_7b2f79b495a2071ce034380738d6b5aa}"   # Jason-Test Group

DRY=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY="--dry-run"
    shift
fi

if [ $# -eq 0 ]; then
    # _seg_ files are process.py's in-flight intermediates, never the deliverable
    newest=$(ls -t output/*.wav 2>/dev/null | grep -v '_seg_[0-9]*\.wav$' | head -1)
    if [ -z "$newest" ]; then
        echo "ERROR: no wav files in output/" >&2
        exit 1
    fi
    set -- "$newest"
fi

for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: not a file: $f" >&2
        exit 1
    fi
    amz lark im send --chat-id "$CHAT_ID" --file "$f" $DRY >/dev/null
    echo "sent: $f"
done

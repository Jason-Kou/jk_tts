#!/bin/bash
# tts_and_notify.sh - Run TTS, convert to OGG, output result path
# Usage: ./tts_and_notify.sh [mode] [voice]
#   mode: voice_design (default, female) | base | cosyvoice3 | moss_local
#   voice: for base/cosyvoice3/moss_* mode, e.g. "jason" or "official_female"
# 
# Example:
#   ./tts_and_notify.sh                    # female voice (default)
#   ./tts_and_notify.sh base jason         # Jason's voice
#   ./tts_and_notify.sh voice_design       # female voice (explicit)
#   ./tts_and_notify.sh cosyvoice3 jason   # Fun-CosyVoice3 local backend
#
# Output: prints OGG file path(s) to stdout when done
# Exit codes: 0=success, 1=no input files, 2=tts failed, 3=convert failed

set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-voice_design}"
VOICE="${2:-}"

# Check for input files
INPUT_COUNT=$(ls input/*.txt 2>/dev/null | wc -l | tr -d ' ')
if [ "$INPUT_COUNT" -eq 0 ]; then
    echo "ERROR: No .txt files in input/" >&2
    exit 1
fi

echo "Starting TTS: mode=$MODE voice=$VOICE files=$INPUT_COUNT" >&2

# Run TTS (stderr only for progress, suppress stdout noise)
if [ -n "$VOICE" ]; then
    uv run python process.py "$MODE" "$VOICE" >&2 2>&1
else
    uv run python process.py "$MODE" >&2 2>&1
fi

TTS_EXIT=$?
if [ $TTS_EXIT -ne 0 ]; then
    echo "ERROR: TTS failed with exit code $TTS_EXIT" >&2
    exit 2
fi

# Convert all new WAV files to OGG
OGG_FILES=()
for wav in output/*.wav; do
    [ -f "$wav" ] || continue
    basename=$(basename "$wav" .wav)
    ogg="/tmp/${basename}.ogg"
    if ffmpeg -i "$wav" -c:a libopus -b:a 64k "$ogg" -y >/dev/null 2>&1; then
        OGG_FILES+=("$ogg")
    else
        echo "WARNING: Failed to convert $wav" >&2
    fi
done

if [ ${#OGG_FILES[@]} -eq 0 ]; then
    echo "ERROR: No OGG files produced" >&2
    exit 3
fi

# Output OGG paths (stdout - this is what LLM reads)
for ogg in "${OGG_FILES[@]}"; do
    echo "$ogg"
done

echo "Done! ${#OGG_FILES[@]} file(s) ready" >&2

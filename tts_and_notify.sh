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

# The files process.py will pick up: everything it has not already prefixed
# with "finished_". Remember them now, so only this run's output gets converted.
PENDING=()
for txt in input/*.txt; do
    [ -f "$txt" ] || continue
    stem=$(basename "$txt" .txt)
    case "$stem" in finished_*) continue ;; esac
    PENDING+=("$stem")
done

if [ ${#PENDING[@]} -eq 0 ]; then
    echo "ERROR: No unprocessed .txt files in input/" >&2
    exit 1
fi

echo "Starting TTS: mode=$MODE voice=$VOICE files=${#PENDING[@]}" >&2

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

# Convert this run's WAV files to OGG
OGG_FILES=()
for stem in "${PENDING[@]}"; do
    wav="output/${stem}.wav"
    [ -f "$wav" ] || { echo "WARNING: $wav was not produced" >&2; continue; }
    ogg="/tmp/${stem}.ogg"
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

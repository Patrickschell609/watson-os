#!/bin/bash
#
# VERITAS - One-command deepfake detection
# Usage: ./run.sh <video_url_or_file> [options]
#

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

cd "$(dirname "$0")"

# Show help if no args
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           PROJECT VERITAS v2.0                            ║${NC}"
    echo -e "${CYAN}║           Forensic Deepfake Detection                     ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Usage:"
    echo "    ./run.sh <video_url_or_file> [options]"
    echo ""
    echo "  Options:"
    echo "    --no-video    Skip annotated video generation (faster)"
    echo "    --no-html     Skip HTML report generation"
    echo "    --json-only   Only output JSON report"
    echo ""
    echo "  Examples:"
    echo "    ./run.sh https://youtube.com/watch?v=VIDEO_ID"
    echo "    ./run.sh /path/to/video.mp4"
    echo "    ./run.sh video.mp4 --no-video"
    echo ""
    echo "  Output:"
    echo "    veritas_report.json  - Machine-readable analysis"
    echo "    veritas_report.html  - Shareable HTML report"
    echo "    veritas_output.mp4   - Annotated video with overlays"
    echo ""
    exit 0
fi

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}[ERROR]${NC} Virtual environment not found."
    echo "        Run ./install.sh first to set up VERITAS."
    exit 1
fi

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} ffmpeg not found."
    echo "        Install with: sudo apt install ffmpeg (Linux)"
    echo "                   or: brew install ffmpeg (Mac)"
    exit 1
fi

# Run analysis
echo ""
.venv/bin/python veritas.py "$@"
echo ""

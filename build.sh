#!/bin/bash
# Watson OS — Master Build Script
# Builds SHEILD from source (optional), packages everything, creates ISO
#
# Usage:
#   ./build.sh                Build ISO (uses pre-built binaries in repo)
#   ./build.sh --full         Compile SHEILD from source first, then build ISO
#   ./build.sh --sheild-dir   Specify SHEILD source location (default: ../SHEILD)
#   ./build.sh --help         Show this help
#
# Requirements:
#   - Debian Bookworm (or compatible)
#   - live-build package (apt install live-build)
#   - Root access (live-build needs chroot)
#   - ~4GB free disk space
#   - For --full: Rust toolchain, cargo-tauri, Node.js
set -e

WATSON_DIR="$(cd "$(dirname "$0")" && pwd)"
SHEILD_DIR=""
FULL_BUILD=false

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
AMBER='\033[0;33m'
GRAY='\033[0;90m'
NC='\033[0m'

usage() {
    echo "Watson OS Build Script"
    echo ""
    echo "  ./build.sh                 Build ISO from pre-built binaries"
    echo "  ./build.sh --full          Compile SHEILD + Ghost Code, then build ISO"
    echo "  ./build.sh --sheild-dir P  Path to SHEILD source (default: ../SHEILD)"
    echo "  ./build.sh --help          Show this message"
    echo ""
    echo "Requirements:"
    echo "  - Debian Bookworm or compatible (Ubuntu 22.04+, Kali, etc.)"
    echo "  - live-build package: sudo apt install live-build"
    echo "  - Root access (sudo)"
    echo "  - ~4GB free disk space"
    echo ""
    echo "The quick build (no --full) uses the pre-built SHEILD, Ghost Code,"
    echo "and Ghost Analyst binaries already in the repo. No Rust toolchain needed."
}

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --full)
            FULL_BUILD=true
            shift
            ;;
        --sheild-dir)
            SHEILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Default SHEILD location: sibling directory
if [ -z "$SHEILD_DIR" ]; then
    SHEILD_DIR="$(dirname "$WATSON_DIR")/SHEILD"
fi

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║         WATSON OS BUILDER           ║"
echo "  ║   Pure Debian · SHEILD · No Trace    ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ============================================
# Preflight checks
# ============================================
echo -e "${CYAN}[*] Preflight checks...${NC}"

if ! command -v lb &>/dev/null; then
    echo -e "${RED}[!] live-build not found. Install it:${NC}"
    echo "    sudo apt install live-build"
    exit 1
fi

if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    echo -e "${AMBER}[*] This script needs root for live-build. You may be prompted for sudo.${NC}"
fi

# Check binaries exist (for quick build)
SHEILD_BIN="$WATSON_DIR/config/includes.chroot_after_packages/opt/sheild/sheild"
GHOST_CODE_BIN="$WATSON_DIR/config/includes.chroot_after_packages/opt/ghost-code/ghost-code"
ANALYST_BIN="$WATSON_DIR/config/includes.chroot_after_packages/opt/analyst/ghost-analyst"

if [ "$FULL_BUILD" = false ]; then
    missing=false
    [ ! -f "$SHEILD_BIN" ] && echo -e "${RED}[!] Missing: SHEILD binary ($SHEILD_BIN)${NC}" && missing=true
    [ ! -f "$GHOST_CODE_BIN" ] && echo -e "${AMBER}[*] Missing: Ghost Code binary (optional, editor won't be included)${NC}"
    [ ! -f "$ANALYST_BIN" ] && echo -e "${AMBER}[*] Missing: Ghost Analyst binary (optional, analyst won't be included)${NC}"

    if [ "$missing" = true ]; then
        echo ""
        echo "  Quick build needs pre-built binaries in the repo."
        echo "  Either add them or run: ./build.sh --full"
        exit 1
    fi
fi

echo -e "${GREEN}[+] Preflight OK${NC}"
echo ""

# ============================================
# Step 1: Compile from source (--full only)
# ============================================
if [ "$FULL_BUILD" = true ]; then
    SHEILD_DEST="$WATSON_DIR/config/includes.chroot_after_packages/opt/sheild"

    # Build SHEILD
    echo -e "${CYAN}[1/4] Building SHEILD browser from source...${NC}"

    if [ ! -d "$SHEILD_DIR" ]; then
        echo -e "${RED}[!] SHEILD source not found at $SHEILD_DIR${NC}"
        echo "    Clone it next to watson-os or use --sheild-dir <path>"
        exit 1
    fi

    if ! command -v cargo &>/dev/null; then
        echo -e "${RED}[!] Rust toolchain not found. Install: https://rustup.rs${NC}"
        exit 1
    fi

    cd "$SHEILD_DIR"
    cargo tauri build --bundles none 2>&1 | tail -5

    SHEILD_COMPILED="$SHEILD_DIR/src-tauri/target/release/sheild"
    if [ ! -f "$SHEILD_COMPILED" ]; then
        echo -e "${RED}[!] SHEILD build failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}[+] SHEILD binary: $(du -h "$SHEILD_COMPILED" | cut -f1)${NC}"

    # Package SHEILD
    echo -e "${CYAN}[2/4] Packaging SHEILD for ISO...${NC}"

    mkdir -p "$SHEILD_DEST/frontend" "$SHEILD_DEST/ghost" "$SHEILD_DEST/scripts"

    cp "$SHEILD_COMPILED" "$SHEILD_DEST/sheild"
    chmod +x "$SHEILD_DEST/sheild"

    cp -r "$SHEILD_DIR/frontend/"* "$SHEILD_DEST/frontend/"
    cp "$SHEILD_DIR/ghost/ghost.js" "$SHEILD_DEST/ghost/"
    cp "$SHEILD_DIR/ghost/package.json" "$SHEILD_DEST/ghost/"
    cp "$SHEILD_DIR/ghost/package-lock.json" "$SHEILD_DEST/ghost/" 2>/dev/null || true
    cp "$SHEILD_DIR/scripts/rotate-ip.sh" "$SHEILD_DEST/scripts/" 2>/dev/null || true
    cp "$SHEILD_DIR/src-tauri/icons/icon.png" "$SHEILD_DEST/icon.png" 2>/dev/null || true

    echo -e "${GREEN}[+] SHEILD packaged: $(du -sh "$SHEILD_DEST" | cut -f1)${NC}"
else
    echo -e "${CYAN}[1/4] Using pre-built binaries (skip compile)${NC}"
    echo -e "${GRAY}       SHEILD: $(du -h "$SHEILD_BIN" 2>/dev/null | cut -f1)${NC}"
    [ -f "$GHOST_CODE_BIN" ] && echo -e "${GRAY}       Ghost Code: $(du -h "$GHOST_CODE_BIN" | cut -f1)${NC}"
    [ -f "$ANALYST_BIN" ] && echo -e "${GRAY}       Ghost Analyst: $(du -h "$ANALYST_BIN" | cut -f1)${NC}"
    echo ""
    echo -e "${CYAN}[2/4] Binaries already in place${NC}"
fi

# ============================================
# Step 3: Build the ISO
# ============================================
echo -e "${CYAN}[3/4] Building Watson OS ISO...${NC}"
echo -e "${GRAY}       This takes 5-20 minutes depending on cache state${NC}"
echo ""

cd "$WATSON_DIR"

# Clean stale binary output but keep cached chroot (saves 10+ min)
sudo lb clean 2>/dev/null || true

# Build
sudo lb build

# ============================================
# Step 4: Report
# ============================================
ISO_FILE=$(ls "$WATSON_DIR/"*.iso 2>/dev/null | head -1)

if [ -n "$ISO_FILE" ]; then
    ISO_SIZE=$(du -h "$ISO_FILE" | cut -f1)
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  WATSON OS ISO READY${NC}"
    echo -e "${GREEN}  File: $(basename "$ISO_FILE")${NC}"
    echo -e "${GREEN}  Size: $ISO_SIZE${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Flash to USB:"
    echo "    sudo dd if=$ISO_FILE of=/dev/sdX bs=4M status=progress"
    echo ""
    echo "  Test in VM:"
    echo "    qemu-system-x86_64 -m 2G -cdrom $ISO_FILE"
    echo ""
    echo -e "${AMBER}  Replace /dev/sdX with your USB device. Double check.${NC}"
else
    echo -e "${RED}[!] ISO build may have failed. Check output above.${NC}"
    exit 1
fi

#!/bin/bash
# Watson OS — Master Build Script
# Builds SHEILD from source, packages into live-build, creates ISO
set -e

WATSON_DIR="$(cd "$(dirname "$0")" && pwd)"
SHEILD_DIR="/home/oday/Desktop/SHEILD"
SHEILD_DEST="$WATSON_DIR/config/includes.chroot_after_packages/opt/sheild"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║         WATSON OS BUILDER           ║"
echo "  ║   Pure Debian · SHEILD · No Trace    ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ============================================
# Step 1: Build SHEILD release binary
# ============================================
echo -e "${CYAN}[1/4] Building SHEILD browser...${NC}"

if [ ! -d "$SHEILD_DIR" ]; then
    echo -e "${RED}[!] SHEILD not found at $SHEILD_DIR${NC}"
    exit 1
fi

cd "$SHEILD_DIR"

# Build Tauri release
cargo tauri build --bundles none 2>&1 | tail -5

SHEILD_BIN="$SHEILD_DIR/src-tauri/target/release/sheild"
if [ ! -f "$SHEILD_BIN" ]; then
    echo -e "${RED}[!] SHEILD build failed — no binary at $SHEILD_BIN${NC}"
    exit 1
fi

echo -e "${GREEN}[+] SHEILD binary: $(du -h "$SHEILD_BIN" | cut -f1)${NC}"

# ============================================
# Step 2: Package SHEILD into ISO includes
# ============================================
echo -e "${CYAN}[2/4] Packaging SHEILD for ISO...${NC}"

mkdir -p "$SHEILD_DEST/frontend"
mkdir -p "$SHEILD_DEST/ghost"
mkdir -p "$SHEILD_DEST/scripts"

# Binary
cp "$SHEILD_BIN" "$SHEILD_DEST/sheild"
chmod +x "$SHEILD_DEST/sheild"

# Frontend (poison.js is the key file)
cp -r "$SHEILD_DIR/frontend/"* "$SHEILD_DEST/frontend/"

# Ghost Mode sources (node_modules installed at first boot)
cp "$SHEILD_DIR/ghost/ghost.js" "$SHEILD_DEST/ghost/"
cp "$SHEILD_DIR/ghost/package.json" "$SHEILD_DEST/ghost/"
cp "$SHEILD_DIR/ghost/package-lock.json" "$SHEILD_DEST/ghost/" 2>/dev/null || true

# IP rotation script
cp "$SHEILD_DIR/scripts/rotate-ip.sh" "$SHEILD_DEST/scripts/" 2>/dev/null || true

# Icon
cp "$SHEILD_DIR/src-tauri/icons/icon.png" "$SHEILD_DEST/icon.png" 2>/dev/null || true

echo -e "${GREEN}[+] SHEILD packaged: $(du -sh "$SHEILD_DEST" | cut -f1)${NC}"

# ============================================
# Step 3: Build the ISO
# ============================================
echo -e "${CYAN}[3/4] Building Watson OS ISO...${NC}"

cd "$WATSON_DIR"

# Clean stale binary output but keep cached chroot (saves 20+ min)
# Use --purge only for full rebuild: sudo lb clean --purge
sudo lb clean 2>/dev/null || true

# Build with full output so errors are visible
sudo lb build

# ============================================
# Step 4: Report
# ============================================
ISO_FILE=$(ls "$WATSON_DIR/"*.iso 2>/dev/null | head -1)

if [ -n "$ISO_FILE" ]; then
    ISO_SIZE=$(du -h "$ISO_FILE" | cut -f1)
    echo ""
    echo -e "${GREEN}══════════════════════════════════════${NC}"
    echo -e "${GREEN}  WATSON OS ISO READY${NC}"
    echo -e "${GREEN}  File: $ISO_FILE${NC}"
    echo -e "${GREEN}  Size: $ISO_SIZE${NC}"
    echo -e "${GREEN}══════════════════════════════════════${NC}"
    echo ""
    echo "  Flash to USB:  dd if=$ISO_FILE of=/dev/sdX bs=4M status=progress"
    echo "  Test in VM:    qemu-system-x86_64 -m 2G -cdrom $ISO_FILE"
else
    echo -e "${RED}[!] ISO build may have failed. Check output above.${NC}"
fi

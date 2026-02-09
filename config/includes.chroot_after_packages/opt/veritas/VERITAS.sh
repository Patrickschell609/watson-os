#!/bin/bash
#
# VERITAS GUI Launcher
# Double-click this to start the graphical interface
#

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "First run detected - running installer..."
    echo ""

    # Try to run installer
    if [ -f "./install.sh" ]; then
        chmod +x install.sh
        ./install.sh
    else
        echo "ERROR: install.sh not found"
        echo "Please run install.sh manually first"
        exit 1
    fi
fi

# Launch GUI
echo "Starting VERITAS..."
.venv/bin/python veritas_gui.py

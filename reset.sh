#!/bin/bash

# =====================================================================================================================================
# 🧹 RESET PROJECT SCRIPT (with colored output)
# This script will delete all contents in the "models" and "output"
# folders. Use with caution: this cannot be undone!
# =====================================================================================================================================

# Colors & styles
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${YELLOW}${BOLD}⚠️  WARNING:${RESET} This will delete all files in 'models/' and 'output/'."
read -p "Are you sure you want to continue? (y/n): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo -e "❌ Operation cancelled by user.${RESET}"
    exit 0
fi

FOLDERS=("models" "output")

for FOLDER in "${FOLDERS[@]}"; do
    if [ -d "$FOLDER" ]; then
        rm -rf "$FOLDER"/*
        echo -e "└── Cleared $FOLDER/${RESET}"
    else
        echo -e "└── ⚠️  Folder $FOLDER does not exist, skipping.${RESET}"
    fi
done

echo
echo -e "🎉 Project reset completed!${RESET}"

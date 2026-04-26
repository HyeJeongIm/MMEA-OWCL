#!/bin/bash

# Base directory containing sweep YAML files
BASE_DIR="scripts-train/TBN/All"
PROJECT="mmea-owcl/Experimental Results on the MMEA-OWCL (CL Training & Evaluation)"

# Define sweep YAML files and corresponding dataset names
declare -A SWEEP_FILES=(
    # ["MAND"]="${BASE_DIR}/1.uestc-mmea-mand-herding.yaml"
    ["MAND_wo_morst"]="${BASE_DIR}/2.uestc-mmea-mand-herding_wo_morst.yaml"
    # ["iCaRL"]="${BASE_DIR}/x.uestc-mmea-icarl.yaml"
    # ["Replay"]="${BASE_DIR}/x.uestc-mmea-replay.yaml"
)

RUN_STD_FILE="${BASE_DIR}/run_sweep.sh"

# Backup the existing run_sweep.sh before modifying it
cp "$RUN_STD_FILE" "${RUN_STD_FILE}.bak"

echo "Starting sweeps and writing to $RUN_STD_FILE..."

# Clear existing content and write the header
cat <<EOT > "$RUN_STD_FILE"
# TBN (All) - MAND
EOT

# Loop through each method in order and append the Sweep ID
for METHOD in "MAND_wo_morst"; do
    SWEEP_FILE="${SWEEP_FILES[$METHOD]}"

    echo "🔹 Running sweep for $METHOD with config: $SWEEP_FILE"

    # Run the wandb sweep command and extract the sweep ID
    OUTPUT=$(wandb sweep "$SWEEP_FILE" 2>&1)
    echo "$OUTPUT"

    if [[ $OUTPUT =~ sweep\ with\ ID:\ ([a-zA-Z0-9]+) ]]; then
        SWEEP_ID="${BASH_REMATCH[1]}"
        FULL_KEY="${PROJECT}/${SWEEP_ID}"
        echo "✅ Extracted Sweep ID: $SWEEP_ID for $METHOD"

        # Append the formatted sweep command to run_sweep.sh
        echo -e "# $METHOD\nwandb agent '${FULL_KEY}'" >> "$RUN_STD_FILE"

    else
        echo "❌ Failed to extract Sweep ID for $METHOD"
        echo "   Make sure you are logged in: wandb login <your_api_key>"
    fi
done

echo ""
echo "✅ Done! Run: bash ${BASE_DIR}/auto_agent.sh"

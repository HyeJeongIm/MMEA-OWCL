#!/bin/bash

# Base directory containing sweep YAML files
BASE_DIR="scripts-eval/TBN/All/All_v1"


# Define sweep YAML files and corresponding dataset names
declare -A SWEEP_FILES=(
    ["iCaRL"]="${BASE_DIR}/1.dataego-icarl.yaml"
    ["Replay"]="${BASE_DIR}/2.dataego-replay.yaml"
    ["EWC"]="${BASE_DIR}/3.dataego-ewc.yaml"
    ["LwF"]="${BASE_DIR}/4.dataego-lwf.yaml"
    ["Upperbound"]="${BASE_DIR}/5.dataego-upperbound.yaml"
)

RUN_STD_FILE="${BASE_DIR}/run_sweep.sh"

# Backup the existing run_std.sh before modifying it
cp "$RUN_STD_FILE" "${RUN_STD_FILE}.bak"

echo "Starting sweeps and appending to $RUN_STD_FILE..."

# Clear existing content and write the header
cat <<EOT > "$RUN_STD_FILE"
# TBN (All) - 5 Models
EOT

# Loop through each method in order and append the Sweep ID
for METHOD in "iCaRL" "Replay" "EWC" "LwF" "Upperbound" ; do
    SWEEP_FILE="${SWEEP_FILES[$METHOD]}"

    echo "🔹 Running sweep for $METHOD with config: $SWEEP_FILE"

    # Run the wandb sweep command and extract the sweep ID
    OUTPUT=$(wandb sweep "$SWEEP_FILE" 2>&1)

    if [[ $OUTPUT =~ sweep\ with\ ID:\ ([a-zA-Z0-9]+) ]]; then
        SWEEP_ID="${BASH_REMATCH[1]}"
        echo "✅ Extracted Sweep ID: $SWEEP_ID for $METHOD"

        # Append the formatted sweep command to the script
        echo -e "# $METHOD\nwandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/$SWEEP_ID'" >> "$RUN_STD_FILE"
    
        # # Execute the wandb agent immediately
        # echo "🚀 Running wandb agent for $METHOD..."
        # CUDA_VISIBLE_DEVICES=1 wandb agent "mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/$SWEEP_ID" --count 1
    else
        echo "❌ Failed to extract Sweep ID for $METHOD"
    fi
done

echo "✅ All sweeps have been executed sequentially and appended to $RUN_STD_FILE!"
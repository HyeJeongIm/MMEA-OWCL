#!/bin/bash

# Base directory containing sweep YAML files
BASE_DIR="scripts-eval/TBN/All"

# Define sweep YAML files and corresponding method names
declare -A SWEEP_FILES=(
    ["MAND"]="${BASE_DIR}/1.uestc-mmea-mand-herding.yaml"
    # ["MAND_wo_morst"]="${BASE_DIR}/2.uestc-mmea-mand-wo-morst-herding.yaml"
    # ["MAND_MoASCL"]="${BASE_DIR}/3.uestc-mmea-mand-moas-cl-herding.yaml"
)

RUN_STD_FILE="${BASE_DIR}/run_sweep.sh"

# Backup the existing run_sweep.sh before modifying it
if [ -f "$RUN_STD_FILE" ]; then
    cp "$RUN_STD_FILE" "${RUN_STD_FILE}.bak"
fi

echo "Starting sweeps and writing to $RUN_STD_FILE..."

# Write the header
cat <<EOT > "$RUN_STD_FILE"
# TBN (All) - MAND (PrototypeAdaptive OOD)
EOT

# Loop through each method and create sweep
for METHOD in "MAND"; do
    SWEEP_FILE="${SWEEP_FILES[$METHOD]}"

    echo "🔹 Running sweep for $METHOD with config: $SWEEP_FILE"

    OUTPUT=$(wandb sweep "$SWEEP_FILE" 2>&1)

    if [[ $OUTPUT =~ sweep\ with\ ID:\ ([a-zA-Z0-9]+) ]]; then
        SWEEP_ID="${BASH_REMATCH[1]}"
        echo "✅ Extracted Sweep ID: $SWEEP_ID for $METHOD"

        # Append the formatted sweep command to run_sweep.sh
        echo -e "# $METHOD\nwandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/$SWEEP_ID'" >> "$RUN_STD_FILE"

        # Update auto_agent.sh with the new sweep ID
        if [ "$METHOD" == "MAND" ]; then
            sed -i "s|\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/[a-zA-Z0-9]*\" # MAND-Herding (PrototypeAdaptive OOD.*|\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL \& OOD)/$SWEEP_ID\" # MAND-Herding (PrototypeAdaptive OOD) — 5seeds×3incs=15runs|g" "${BASE_DIR}/auto_agent.sh"
            sed -i "s|\[\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/[a-zA-Z0-9]*\"\]=5 # MAND-Herding (PrototypeAdaptive OOD.*|\[\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL \& OOD)/$SWEEP_ID\"\]=5 # MAND-Herding (PrototypeAdaptive OOD) — 5seeds×3incs=15runs|g" "${BASE_DIR}/auto_agent.sh"
        elif [ "$METHOD" == "MAND_MoASCL" ]; then
            # [a-zA-Z0-9_]* — underscore 포함 (PLACEHOLDER_MOAS_CL 매칭 위해)
            sed -i "s|\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/[a-zA-Z0-9_]*\" # MAND-MoASCL-Herding.*|\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL \& OOD)/$SWEEP_ID\" # MAND-MoASCL-Herding — 5seeds×3incs=15runs|g" "${BASE_DIR}/auto_agent.sh"
            sed -i "s|\[\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/[a-zA-Z0-9_]*\"\]=5 # MAND-MoASCL-Herding.*|\[\"mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL \& OOD)/$SWEEP_ID\"\]=5 # MAND-MoASCL-Herding — 5seeds×3incs=15runs|g" "${BASE_DIR}/auto_agent.sh"
        fi

        echo "✅ Updated auto_agent.sh with Sweep ID: $SWEEP_ID"
    else
        echo "❌ Failed to extract Sweep ID for $METHOD"
        echo "Output was: $OUTPUT"
    fi
done

echo "✅ Done! Run sweep with:"
echo "   bash ${RUN_STD_FILE}        # single agent"
echo "   bash ${BASE_DIR}/auto_agent.sh  # multi-agent with Slack"

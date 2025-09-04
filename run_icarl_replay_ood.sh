#!/bin/bash

# iCaRL & Replay OOD Evaluation Script
# Automatically evaluates all seeds and calculates averages for iCaRL and Replay models

echo "🏆 iCaRL & Replay OOD Evaluation (Increment 8)"
echo "=============================================="

# Configuration
RESULTS_DIR="/workspace/MMEA-OWCL/ood_results_icarl_replay_$(date +%Y%m%d_%H%M%S)"

# Ensure main results directory exists
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "📁 Creating main results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
else
    echo "📁 Results directory already exists: $RESULTS_DIR"
fi

# Model configurations with modality detection
declare -A MODELS
declare -A MODEL_MODALITIES
declare -A MODEL_OOD_METHODS

# iCaRL model
MODELS["icarl"]="/workspace/MMEA-OWCL/weights/mmea_tbn_tbn_icarl_rgbgyroacce_ep50_bs8_pb1_fr0_inc8_mem320_train"
# Detect modality from path: rgbgyroacce means RGB+Gyro+Acce
if [[ "${MODELS["icarl"]}" == *"rgbgyroacce"* ]]; then
    MODEL_MODALITIES["icarl"]='["RGB", "Gyro", "Acce"]'
    MODEL_OOD_METHODS["icarl"]='["MSP", "ODIN", "Energy", "LTS_Individual", "LTS_Fusion", "LTS_RGB_Only", "LTS_Gyro_Only", "LTS_Acce_Only", "LTS_Late_Fusion"]'
elif [[ "${MODELS["icarl"]}" == *"rgb"* ]]; then
    MODEL_MODALITIES["icarl"]='["RGB"]'
    MODEL_OOD_METHODS["icarl"]='["MSP", "ODIN", "Energy", "LTS_RGB_Only"]'
fi

# Replay model  
MODELS["replay"]="/workspace/MMEA-OWCL/weights/mmea_tbn_tbn_replay_rgbgyroacce_ep50_bs8_pb1_fr0_inc8_mem320_train"
# Detect modality from path: rgbgyroacce means RGB+Gyro+Acce
if [[ "${MODELS["replay"]}" == *"rgbgyroacce"* ]]; then
    MODEL_MODALITIES["replay"]='["RGB", "Gyro", "Acce"]'
    MODEL_OOD_METHODS["replay"]='["MSP", "ODIN", "Energy", "LTS_Individual", "LTS_Fusion", "LTS_RGB_Only", "LTS_Gyro_Only", "LTS_Acce_Only", "LTS_Late_Fusion"]'
elif [[ "${MODELS["replay"]}" == *"rgb"* ]]; then
    MODEL_MODALITIES["replay"]='["RGB"]'
    MODEL_OOD_METHODS["replay"]='["MSP", "ODIN", "Energy", "LTS_RGB_Only"]'
fi

# Function to run OOD evaluation for a single seed
run_single_evaluation() {
    local model_type="$1"
    local weights_path="$2"
    local seed_name="$3"
    local log_file="$4"
    
    # Create temporary config file
    TEMP_CONFIG=$(mktemp --suffix=.json)
    
    # Ensure temp config was created successfully
    if [[ ! -f "$TEMP_CONFIG" ]]; then
        echo "   ❌ Failed to create temporary config file"
        return 1
    fi
    
    cat > "$TEMP_CONFIG" << EOF
{
  "dataset": "mmea-tbn",
  "modality": ["RGB", "Gyro", "Acce"],
  "model_name": "tbn_${model_type}",
  
  "backbone": "tbn",
  "arch": "BNInception",
  "fusion_type": "concat",
  "consensus_type": "avg",
  "before_softmax": true,
  
  "train_list": "mydataset_total_train.txt",
  "test_list": "mydataset_test.txt",
  "mpu_path": "./datasets/UESTC-MMEA-CL/UESTC-MMEA-CL/mpu/",
  
  "num_segments": 8,
  "batch_size": 8,
  "workers": 4,
  "dropout": 0.5,
  
  "lr": 0.001,
  "lr_steps": [10, 20],
  "momentum": 0.9,
  "weight_decay": 0.0005,
  "epochs": 1,
  "clip_gradient": 20,
  
  "memory_size": 320,
  "increment": 8,
  "shuffle": false,
  
  "partialbn": true,
  "freeze": false,
  
  "mode": "eval",
  "enable_ood": true,
  "ood_methods": ["MSP", "ODIN", "Energy", "LTS_Individual", "LTS_Fusion", "LTS_RGB_Only", "LTS_Gyro_Only", "LTS_Acce_Only", "LTS_Late_Fusion"],
  "weights_path": "$weights_path",
  
  "seed": 1993,
  "device": [0],
  "prefix": "mmea-owcl",
  "log_test_acc": false,
  "use_wandb": false,
  "debug_mode": false
}
EOF
    
    echo "   📝 Config created: $TEMP_CONFIG"
    echo "   🚀 Running evaluation... (this may take 5-10 minutes)"
    
    # Ensure log directory exists
    local log_dir=$(dirname "$log_file")
    if [[ ! -d "$log_dir" ]]; then
        echo "   📁 Creating log directory: $log_dir"
        mkdir -p "$log_dir"
    fi
    
    # Show a simple spinner while evaluation is running
    (
        while kill -0 $$ 2>/dev/null; do
            for spinner in '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏'; do
                printf "\r   $spinner Evaluating $seed_name..."
                sleep 0.2
            done
        done
    ) &
    SPINNER_PID=$!
    
    # Run evaluation
    if WANDB_MODE=disabled python main.py -d mmea-tbn -m "tbn_${model_type}" --config "$TEMP_CONFIG" > "$log_file" 2>&1; then
        kill $SPINNER_PID 2>/dev/null
        printf "\r   ✅ Evaluation completed successfully for $seed_name\n"
        rm -f "$TEMP_CONFIG"
        
        # Check if log file was actually created
        if [[ ! -f "$log_file" ]]; then
            echo "   ⚠️  Warning: Log file not created, but evaluation reported success"
            return 1
        fi
        return 0
    else
        kill $SPINNER_PID 2>/dev/null
        printf "\r   ❌ Evaluation failed for $seed_name\n"
        if [[ -f "$log_file" ]]; then
            echo "   📋 Check log: $log_file"
        else
            echo "   📋 No log file generated"
        fi
        rm -f "$TEMP_CONFIG"
        return 1
    fi
}

# Function to extract OOD results from log
extract_ood_results() {
    local log_file="$1"
    local seed_name="$2"
    local results_file="$3"
    
    echo "   📊 Extracting results from: $log_file"
    
    # Check if log file exists
    if [[ ! -f "$log_file" ]]; then
        echo "   ❌ Log file not found: $log_file"
        return 1
    fi
    
    # Check if results file directory exists
    local results_dir=$(dirname "$results_file")
    if [[ ! -d "$results_dir" ]]; then
        echo "   📁 Creating results directory: $results_dir"
        mkdir -p "$results_dir"
    fi
    
    # Ensure results file exists with header
    if [[ ! -f "$results_file" ]]; then
        echo "   📄 Creating results file: $results_file"
        echo "Seed,Method,AUROC" > "$results_file"
    fi
    
    # Extract AUROC results (look for lines with percentages)
    local found_results=0
    local methods_count=0
    
    echo "   🔍 Searching for OOD results in log file..."
    
    # Check for skipped methods
    if grep -q "returned None - skipping this method" "$log_file" 2>/dev/null; then
        echo "   ⚠️  Some methods were skipped due to modality incompatibility:"
        grep "returned None - skipping this method" "$log_file" | sed 's/^.*⚠️  /      ⚠️  /' | head -5
    fi
    
    if grep -E "=>" "$log_file" | grep -E "[0-9]+\.[0-9]+%" >/dev/null 2>&1; then
        echo "   📋 Found OOD results, extracting..."
        
        grep -E "=>" "$log_file" | grep -E "[0-9]+\.[0-9]+%" | while read line; do
            if [[ $line =~ ([A-Z_]+[A-Za-z_]*):\ +([0-9]+\.[0-9]+)% ]]; then
                method="${BASH_REMATCH[1]}"
                auroc="${BASH_REMATCH[2]}"
                echo "$seed_name,$method,$auroc" >> "$results_file"
                echo "      ✅ $method: $auroc% (added to results)"
                found_results=1
                ((methods_count++))
            fi
        done
        
        echo "   📊 Extracted $methods_count OOD method results for seed $seed_name"
    fi
    
    if [[ $found_results -eq 0 ]]; then
        echo "   ⚠️  No OOD results found in log file"
        echo "   📋 Log file contents (last 10 lines):"
        tail -n 10 "$log_file" | sed 's/^/      /'
        return 1
    fi
    
    return 0
}

# Function to evaluate a model (all seeds)
evaluate_model() {
    local model_type="$1"
    local model_dir="$2"
    
    echo ""
    echo "🎯 Evaluating $model_type Model"
    echo "================================"
    echo "📁 Model directory: $model_dir"
    
    # Check if model directory exists
    if [[ ! -d "$model_dir" ]]; then
        echo "❌ Model directory not found: $model_dir"
        return 1
    fi
    
    # Get all seed directories
    SEED_DIRS=($(find "$model_dir" -maxdepth 1 -type d -name "*-*-*" | sort))
    
    if [[ ${#SEED_DIRS[@]} -eq 0 ]]; then
        echo "❌ No seed directories found in: $model_dir"
        return 1
    fi
    
    echo "🌱 Found ${#SEED_DIRS[@]} seeds:"
    for seed in "${SEED_DIRS[@]}"; do
        echo "   - $(basename $seed)"
    done
    
    # Results file for this model
    MODEL_RESULTS="$RESULTS_DIR/${model_type}_results.csv"
    
    # Ensure results directory exists
    if [[ ! -d "$RESULTS_DIR" ]]; then
        echo "📁 Creating results directory: $RESULTS_DIR"
        mkdir -p "$RESULTS_DIR"
    fi
    
    # Create results file with header
    echo "Seed,Method,AUROC" > "$MODEL_RESULTS"
    
    # Process each seed with progress bar
    success_count=0
    echo ""
    echo "🔄 Processing ${#SEED_DIRS[@]} seeds..."
    
    for i in "${!SEED_DIRS[@]}"; do
        SEED_DIR="${SEED_DIRS[$i]}"
        SEED_NAME=$(basename "$SEED_DIR")
        
        # Progress bar using tqdm-style output
        progress=$((($i + 1) * 100 / ${#SEED_DIRS[@]}))
        filled=$((progress / 5))
        empty=$((20 - filled))
        
        printf "\r🔄 [%s%s] %d%% (%d/%d) Processing: %s" \
            "$(printf '%*s' $filled | tr ' ' '█')" \
            "$(printf '%*s' $empty | tr ' ' '░')" \
            $progress $((i+1)) ${#SEED_DIRS[@]} "$SEED_NAME"
        
        echo ""  # New line for cleaner output
        
        # Check if weights exist (look for .pkl files for continual learning models)
        if [[ -z "$(find "$SEED_DIR" -name "*.pkl" -o -name "*.pth*" -o -name "*.tar" | head -1)" ]]; then
            echo "   ⏳ No trained weights found - skipping"
            continue
        fi
        
        echo "   ✅ Found trained weights in $SEED_NAME"
        
        # Run evaluation
        LOG_FILE="$RESULTS_DIR/${model_type}_${SEED_NAME}.log"
        
        if run_single_evaluation "$model_type" "$SEED_DIR" "$SEED_NAME" "$LOG_FILE"; then
            # Extract results
            if extract_ood_results "$LOG_FILE" "$SEED_NAME" "$MODEL_RESULTS"; then
                ((success_count++))
            else
                echo "   ⚠️  Failed to extract results for $SEED_NAME"
            fi
        fi
    done
    
    echo ""
    echo "📊 $model_type Model Summary: $success_count/${#SEED_DIRS[@]} seeds completed successfully"
    
    # Calculate averages for this model
    if [[ $success_count -gt 0 ]]; then
        echo ""
        echo "📈 Calculating averages for $model_type across $success_count seeds..."
        echo "📊 Processing results from: $MODEL_RESULTS"
        
        python3 << EOF
import pandas as pd
import numpy as np

try:
    import os
    
    # Check if results file exists
    if not os.path.exists('$MODEL_RESULTS'):
        print(f"❌ No results file found for $model_type: $MODEL_RESULTS")
        print("   This usually means no seeds completed successfully.")
        exit()
    
    # Read results
    df = pd.read_csv('$MODEL_RESULTS')
    
    if len(df) == 0:
        print(f"❌ Results file is empty for $model_type")
        exit()
    
    print(f"📋 Raw results loaded: {len(df)} entries from {df['Seed'].nunique()} unique seeds")
    print(f"🔍 Available methods: {', '.join(df['Method'].unique())}")
    print(f"🌱 Seeds included: {', '.join(df['Seed'].unique())}")
    
    # Group by method and calculate statistics
    summary = df.groupby('Method')['AUROC'].agg(['mean', 'std', 'count']).round(1)
    summary.columns = ['Mean', 'Std', 'Count']
    summary = summary.sort_values('Mean', ascending=False)
    
    # Ensure summary directory exists
    summary_dir = '$RESULTS_DIR'
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        print(f"📁 Created summary directory: {summary_dir}")
    
    # Save summary
    summary_file = f'{summary_dir}/${model_type}_summary.csv'
    summary.to_csv(summary_file)
    print(f"💾 Summary saved: {summary_file}")
    
    print(f"\n🏆 $model_type FINAL AVERAGED RESULTS:")
    print("=" * 60)
    print(f"{'Method':<18} {'Mean AUROC':<12} {'±Std Dev':<10} {'Seeds Used':<10}")
    print("-" * 60)
    
    for method, row in summary.iterrows():
        mean = row['Mean']
        std = row['Std']
        count = int(row['Count'])
        print(f"{method:<18} {mean:>8.1f}%     ±{std:>6.1f}%    {count:>8}")
    
    # Best method
    best_method = summary.index[0]
    best_score = summary.loc[best_method, 'Mean']
    
    print("-" * 60)
    print(f"🥇 BEST METHOD: {best_method} (Average: {best_score:.1f}%)")
    print(f"📊 Results are averaged across {summary.loc[best_method, 'Count']} seeds")
    
except Exception as e:
    print(f"❌ Error calculating averages for $model_type: {e}")
    print("   This usually indicates no successful evaluations were completed.")
    # Don't print full traceback for cleaner output
EOF
    else
        echo "❌ No successful evaluations for $model_type"
    fi
}

# Main execution
echo "🚀 Starting iCaRL & Replay OOD Evaluation..."
echo "Models: iCaRL, Replay"
echo "Modality: RGB+Gyro+Acce"
echo "Increment: 8"
echo "Expected seeds: 5 per model"
echo ""

# Time tracking
START_TIME=$(date +%s)
echo "⏰ Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "⏱️  Estimated total time: 60-90 minutes (6-9 min per seed)"
echo ""

# Total progress tracking
TOTAL_MODELS=(${!MODELS[@]})
TOTAL_MODEL_COUNT=${#TOTAL_MODELS[@]}

echo "📊 Overall Progress: Evaluating $TOTAL_MODEL_COUNT models"
echo "════════════════════════════════════════════════════════"

# Evaluate each model with overall progress
model_counter=0
for model_type in "${!MODELS[@]}"; do
    ((model_counter++))
    model_dir="${MODELS[$model_type]}"
    
    # Overall progress bar
    overall_progress=$(($model_counter * 100 / $TOTAL_MODEL_COUNT))
    overall_filled=$(($overall_progress / 10))
    overall_empty=$((10 - $overall_filled))
    
    printf "🌟 Overall: [%s%s] %d%% - Starting %s\n" \
        "$(printf '%*s' $overall_filled | tr ' ' '█')" \
        "$(printf '%*s' $overall_empty | tr ' ' '░')" \
        $overall_progress "$model_type"
    
    evaluate_model "$model_type" "$model_dir"
    
    printf "✅ Completed %s (%d/%d models)\n\n" "$model_type" $model_counter $TOTAL_MODEL_COUNT
done

# Generate final comparison report
echo ""
echo "📋 Generating final comparison report..."

FINAL_REPORT="$RESULTS_DIR/comparison_report.txt"

# Ensure results directory exists
if [[ ! -d "$RESULTS_DIR" ]]; then
    mkdir -p "$RESULTS_DIR"
fi

cat > "$FINAL_REPORT" << 'EOF'
================================================================
🏆 iCaRL vs Replay OOD Performance Comparison
================================================================

This report compares OOD detection performance between iCaRL and 
Replay models on RGB+Gyro+Acce modality with increment 8.

Each result is the AVERAGE across multiple seeds (typically 5 seeds).
Results Format: Method -> AUROC_Mean ± AUROC_Std (Seeds_Count)

IMPORTANT: All percentages shown are averages calculated from individual
seed evaluations. Standard deviation shows variability across seeds.

EOF

# Add results for each model
for model_type in icarl replay; do
    summary_file="$RESULTS_DIR/${model_type}_summary.csv"
    
    if [[ -f "$summary_file" ]]; then
        echo "" >> "$FINAL_REPORT"
        echo "## $(echo $model_type | tr '[:lower:]' '[:upper:]') MODEL" >> "$FINAL_REPORT"
        echo "-----------------------------------------------------------" >> "$FINAL_REPORT"
        
        python3 << EOF >> "$FINAL_REPORT"
import pandas as pd
import os
try:
    if os.path.exists('$summary_file'):
        df = pd.read_csv('$summary_file', index_col=0)
        if len(df) > 0:
            for method, row in df.iterrows():
                mean = row['Mean']
                std = row['Std']
                count = int(row['Count'])
                print(f"{method:<18} -> {mean:5.1f}% ± {std:4.1f}% ({count} seeds)")
            
            best_method = df.index[0]
            best_score = df.loc[best_method, 'Mean']
            print(f"\n🥇 Best: {best_method} ({best_score:.1f}%)")
        else:
            print("No results available (empty summary file)")
    else:
        print("No results available (summary file not found)")
    
except Exception as e:
    print(f"Error processing results: {e}")
EOF
        echo "" >> "$FINAL_REPORT"
    else
        echo "" >> "$FINAL_REPORT"
        echo "## $(echo $model_type | tr '[:lower:]' '[:upper:]') MODEL" >> "$FINAL_REPORT"
        echo "-----------------------------------------------------------" >> "$FINAL_REPORT"
        echo "❌ No results available" >> "$FINAL_REPORT"
        echo "" >> "$FINAL_REPORT"
    fi
done

# Overall comparison
echo "" >> "$FINAL_REPORT"
echo "================================================================" >> "$FINAL_REPORT"
echo "🎯 OVERALL COMPARISON" >> "$FINAL_REPORT"
echo "================================================================" >> "$FINAL_REPORT"

python3 << EOF >> "$FINAL_REPORT"
import pandas as pd
import os

try:
    results = []
    
    for model in ['icarl', 'replay']:
        summary_file = f'$RESULTS_DIR/{model}_summary.csv'
        if os.path.exists(summary_file):
            try:
                df = pd.read_csv(summary_file, index_col=0)
                if len(df) > 0:
                    best_row = df.iloc[0]  # First row (highest mean)
                    results.append({
                        'Model': model.upper(),
                        'Best_Method': best_row.name,
                        'AUROC': best_row['Mean'],
                        'Std': best_row['Std'],
                        'Seeds': int(best_row['Count'])
                    })
            except Exception as e:
                print(f"Error reading summary for {model}: {e}")
    
    if results:
        print(f"{'Model':<8} {'Best Method':<18} {'AUROC':<8} {'±Std':<6} {'Seeds'}")
        print("-" * 50)
        
        for result in results:
            print(f"{result['Model']:<8} {result['Best_Method']:<18} {result['AUROC']:>5.1f}% ±{result['Std']:>4.1f} {result['Seeds']}")
        
        # Find overall winner
        if len(results) > 1:
            best_result = max(results, key=lambda x: x['AUROC'])
            print("-" * 50)
            print(f"🏆 WINNER: {best_result['Model']} with {best_result['Best_Method']} ({best_result['AUROC']:.1f}%)")
        elif len(results) == 1:
            result = results[0]
            print("-" * 50)
            print(f"🏆 ONLY RESULT: {result['Model']} with {result['Best_Method']} ({result['AUROC']:.1f}%)")
    else:
        print("❌ No results available for comparison")
        print("   This usually means all evaluations failed.")
        
except Exception as e:
    print(f"❌ Error in overall comparison: {e}")
EOF

# Calculate total execution time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "🎉 Evaluation completed!"
echo "⏰ Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
printf "⏱️  Total execution time: "
if [ $HOURS -gt 0 ]; then
    printf "%dh " $HOURS
fi
if [ $MINUTES -gt 0 ] || [ $HOURS -gt 0 ]; then
    printf "%dm " $MINUTES
fi
printf "%ds\n" $SECONDS

echo ""
echo "📊 Results Summary:"
echo "   📂 All results: $RESULTS_DIR"
echo "   📋 Comparison report: $FINAL_REPORT"
echo ""
echo "📈 Quick view of comparison report:"
echo "=================================="
if [[ -f "$FINAL_REPORT" ]]; then
    tail -n 15 "$FINAL_REPORT"
else
    echo "❌ Comparison report not found: $FINAL_REPORT"
fi

echo ""
echo "✅ iCaRL & Replay OOD evaluation pipeline completed successfully!"

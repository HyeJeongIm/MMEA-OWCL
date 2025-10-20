#!/bin/bash

# DataEgo TBN RGB_Gyro 5개 모델 병렬 실행 스크립트
echo "========== DataEgo TBN RGB_Gyro 5 Models Parallel Execution =========="

# iCaRL
echo "🚀 Starting iCaRL..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/4lex3y4p' --count 1 &

# Replay
echo "🚀 Starting Replay..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/aisdwni4' --count 1 &

# EWC
echo "🚀 Starting EWC..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/3ugrouln' --count 1 &

# LwF
echo "🚀 Starting LwF..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/r0jwnk87' --count 1 &

# Upperbound
echo "🚀 Starting Upperbound..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/lub0gph7' --count 1 &

echo "✅ All 5 RGB_Gyro models started in parallel!"
echo "📊 Monitor progress at: https://wandb.ai/mmea-owcl/Experimental%20Results%20on%20the%20MMEA-OWCL%20%28CL%20Training%20%26%20Evaluation%29"

wait
echo "🎉 All RGB_Gyro models completed!"
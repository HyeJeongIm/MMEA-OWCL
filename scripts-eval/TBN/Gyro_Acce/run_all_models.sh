#!/bin/bash

# DataEgo TBN Gyro_Acce 5개 모델 병렬 실행 스크립트
echo "========== DataEgo TBN Gyro_Acce 5 Models Parallel Execution =========="

# iCaRL
echo "🚀 Starting iCaRL..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/eewgrol4' --count 1 &

# Replay
echo "🚀 Starting Replay..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/f8mm2kt0' --count 1 &

# EWC
echo "🚀 Starting EWC..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/r387nssg' --count 1 &

# LwF
echo "🚀 Starting LwF..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/ungeghaf' --count 1 &

# Upperbound
echo "🚀 Starting Upperbound..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)/3v5m4ulb' --count 1 &

echo "✅ All 5 Gyro_Acce models started in parallel!"
echo "📊 Monitor progress at: https://wandb.ai/mmea-owcl/Experimental%20Results%20on%20the%20MMEA-OWCL%20%28CL%20Training%20%26%20Evaluation%29"

wait
echo "🎉 All Gyro_Acce models completed!"
#!/bin/bash

# DataEgo TBN RGB 5개 모델 병렬 실행 스크립트
echo "========== DataEgo TBN RGB 5 Models Parallel Execution =========="

# iCaRL
echo "🚀 Starting iCaRL..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (CL Training & Evaluation)/jctt211b' --count 1 &

# Replay
echo "🚀 Starting Replay..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (CL Training & Evaluation)/sk3z6ege' --count 1 &

# EWC
echo "🚀 Starting EWC..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (CL Training & Evaluation)/505y2o9x' --count 1 &

# LwF
echo "🚀 Starting LwF..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (CL Training & Evaluation)/hvvfotc2' --count 1 &

# Upperbound
echo "🚀 Starting Upperbound..."
wandb agent 'mmea-owcl/Experimental Results on the MMEA-OWCL (CL Training & Evaluation)/nquew071' --count 1 &

echo "✅ All 5 RGB models started in parallel!"
echo "📊 Monitor progress at: https://wandb.ai/mmea-owcl/Experimental%20Results%20on%20the%20MMEA-OWCL%20%28CL%20Training%20%26%20Evaluation%29"

wait
echo "🎉 All RGB models completed!"
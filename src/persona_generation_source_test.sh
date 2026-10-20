#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -f

# Activate virtual env
source ../venv/bin/activate

# Load S3 configuration
if [ -f .aws_batch_config ]; then
    source .aws_batch_config
else
    echo "❌ Error: .aws_batch_config not found"
    exit 1
fi

# GSS 
domain='gss'
data_dir="/quick_start/${domain}_per_user_with_gt.json"

sim_model='claude3.7'
max_token=50000  #131072
num_history=all 
format='newfree' 
proc=15
feature_set='opinionqa'  # 'opinionqa', 'extended', 'all'
python persona_generation_source.py \
    --data_dir ${data_dir} \
    --domain ${domain} \
    --num_history ${num_history} \
    --sim_model ${sim_model} \
    --max_token ${max_token} \
    --format ${format} \
    --proc ${proc} \
    --feature_set ${feature_set} \

echo ""
echo "=========================================="
echo "✅ ${data_name} complete!"
echo "=========================================="
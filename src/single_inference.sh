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

sim_model='claude3.7' # claude3.7  llama3.3-70b nova-pro deepseek-r1 qwen3-32b
eval_only='false'
dataset="opinionqa" # simbench, socsci, gss, opinionqa

# Sub-options
simbench_name="opinionqa"        # opinionqa ESS Afrobarometer ISSP LatinoBarometro
socsci_name="nj5dx"              # mp3dr nj5dx
opinionqa_wave="29"

if [ "$dataset" = "simbench" ]; then
    data_name="${simbench_name}"
    data_dir="dir to SimBenchGrouped_${data_name}.json"

elif [ "$dataset" = "socsci" ]; then
    data_name="${socsci_name}"
    data_dir="dir to grouped_socsci_${data_name}.json"

elif [ "$dataset" = "gss" ]; then
    data_name="gss"
    data_dir="dir to ${data_name}_grouped.json"

elif [ "$dataset" = "opinionqa" ]; then
    wave="${opinionqa_wave}"
    data_name="opinionqa_W${wave}"
    data_dir="dir to ${data_name}_grouped.json"

else
    echo "Error: Unknown dataset '${dataset}'. Choose from: simbench, socsci, gss, opinionqa"
    exit 1
fi

echo "Dataset:  ${dataset}"
echo "Name:     ${data_name}"
echo "Path:     ${data_dir}"

output_type='verbalized' # token_prob
max_token=1024
check_interval=60
num_inference=1
temperature=0.0

echo "=========================================="
echo "Batch Inference Experiment"
echo "=========================================="
echo "Data: ${data_name}"
echo "Model: ${sim_model}"
echo "S3 Bucket: ${S3_BUCKET}"
echo "S3 Prefix: ${S3_PREFIX}/${data_name}"
echo "=========================================="

python single_inference_batch.py \
    --data_dir ${data_dir} \
    --sim_model ${sim_model} \
    --output_type ${output_type} \
    --max_token ${max_token} \
    --data_name ${data_name} \
    --eval_only ${eval_only} \
    --num_inference ${num_inference} \
    --temperature ${temperature} \
    --s3_bucket ${S3_BUCKET} \
    --s3_prefix ${S3_PREFIX}/${data_name} \
    --check_interval ${check_interval} \
    --aws_profile ${AWS_PROFILE} \

echo ""
echo "=========================================="
echo "✅ ${data_name} complete!"
echo "=========================================="
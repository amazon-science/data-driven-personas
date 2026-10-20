#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -f

# Activate virtual env
if [ ! -f ../venv/bin/activate ]; then
    echo "❌ Error: Virtual environment not found at ../venv/bin/activate"
    exit 1
fi
source ../venv/bin/activate || { echo "❌ Error: Failed to activate virtual environment"; exit 1; }

# Load S3 configuration
if [ -f .aws_batch_config ]; then
    source .aws_batch_config
else
    echo "❌ Error: .aws_batch_config not found"
    exit 1
fi

output_type='verbalized' # token_prob  verbalized 
sim_model='claude3.7' #llama3.3-70b claude3.7
max_token=300
eval_from='first'
check_interval=60
proc=30

# Inference
eval_only='false'
data_name='gss_80_q0' # gss_20_q gss_80_ind
persona_source="${data_name}" # movie.  survey_nj5dx.  opinionqa_W26.  gss

persona_prompt_type='bio_history' # bio  bio_dict  dict, natural

format="newfree" # datainfo_${data_name}   rubric  free
num_persona=30


# ########## Test DATA ##############
# GSS - Question
data_dir="dir to ${data_name}_grouped.json"
history_dir="dir to ${persona_source}_per_user_with_gt.json"
output_dir="output_folder"

# ########## Persona SORUCE ##############
persona_dir="dir to ${persona_source}_personas_opinionqafeatures_allreviews_${format}.json"


echo "=========================================="
echo "Batch Inference Experiment"
echo "=========================================="
echo "Data: ${data_name}"
echo "Model: ${sim_model}"
echo "S3 Bucket: ${S3_BUCKET}"
echo "S3 Prefix: ${S3_PREFIX}/${data_name}"
echo "=========================================="

python persona_inference_split.py \
    --data_dir ${data_dir} \
    --output_dir ${output_dir} \
    --history_dir ${history_dir} \
    --sim_model ${sim_model} \
    --output_type ${output_type} \
    --format ${format} \
    --persona_source ${persona_source} \
    --persona_prompt_type ${persona_prompt_type} \
    --max_token ${max_token} \
    --data_name ${data_name} \
    --eval_only ${eval_only} \
    --eval_from ${eval_from} \
    --num_persona ${num_persona} \
    --persona_dir ${persona_dir} \
    --proc ${proc} \
    --s3_bucket ${S3_BUCKET} \
    --s3_prefix ${S3_PREFIX}/${data_name} \
    --check_interval ${check_interval} \
    --aws_profile ${AWS_PROFILE} \

echo ""
echo "=========================================="
echo "✅ ${data_name} complete!"
echo "=========================================="
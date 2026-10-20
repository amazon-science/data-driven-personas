#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -f

source ../venv/bin/activate

if [ -f .aws_batch_config ]; then
    source .aws_batch_config
else
    echo "❌ Error: .aws_batch_config not found"
    exit 1
fi

output_type='verbalized'
sim_model='llama3.3-70b' # llama3.3-70b claude3.7 qwen3-32b
max_token=300
eval_from='first'
check_interval=60
proc=30

eval_only='false'

num_persona=30
output_dir="/path_to_local_output_dir"

persona_prompt_type='bio' # bio  bio_history
format="newfree" # newfree  newfree_gt

dataset="socsci"
persona_source="gss"

# ========== Test DATA ==========
if [ "$dataset" = "simbench" ]; then
    data_name='opinionqa'
    data_dir="dir to SimBenchGrouped_${data_name}.json"
elif [ "$dataset" = "socsci" ]; then
    data_name='nj5dx' #mp3dr nj5dx
    data_dir="dir to grouped_socsci_${data_name}.json"
elif [ "$dataset" = "gss" ]; then
    data_name='gss'
    data_dir="dir to ${data_name}_grouped.json"
elif [ "$dataset" == "opinionqa_W26" ]; then
    data_name='opinionqa_W26' # opinionqa_W26 opinionqa_W29
    data_dir="dir to ${data_name}_grouped.json"
elif [ "$dataset" == "opinionqa_W29" ]; then
    data_name='opinionqa_W29' # opinionqa_W26 opinionqa_W29
    data_dir="dir to ${data_name}_grouped.json"
else
    echo "Error: Unknown dataset '${dataset}'"
    exit 1
fi

# ========== Persona SOURCE ==========
if [ "$persona_source" = "claude37" ]; then
    persona_dir="dir to ${persona_source}_personas_100_newfree_simbench_opinionqa_simbench_direct.json"
    history_dir="/"
elif [ "$persona_source" = "gss" ]; then
    persona_dir="dir to ${persona_source}_personas_opinionqafeatures_allreviews_${format}.json"
    history_dir="dir to ${persona_source}_per_user_with_gt.json"
elif [ "$persona_source" = "opinionqa_W26" ]; then
    persona_dir="dir to ${persona_source}_personas_opinionqafeatures_allreviews_${format}.json"
    history_dir="dir to ${persona_source}_per_user_with_gt.json"
elif [ "$persona_source" = "opinionqa_W29" ]; then
    persona_dir="dir to ${persona_source}_personas_opinionqafeatures_allreviews_${format}.json"
    history_dir="dir to ${persona_source}_per_user_with_gt.json"
elif [ "$persona_source" = "shopping" ]; then
    persona_dir="dir to ${persona_source}_personas_opinionqafeatures_allreviews_${format}.json"
    history_dir="dir to amazon_purchase_per_user.json"
elif [ "$persona_source" = "movie" ]; then
    persona_dir="dir to ${persona_source}_personas_opinionqafeatures_allreviews_${format}.json"
    history_dir="dir to IMDB_per_user.json"
elif [ "$persona_source" = "socsci" ]; then
    socsci_sub="nj5dx"
    persona_source="${socsci_sub}"  # override to match original behavior
    persona_dir="dir to survey_${persona_source}_personas_opinionqafeatures_allreviews_${format}.json"
    history_dir="dir to socsci_${persona_source}_per_respondent.json"
else
    echo "Error: Unknown persona_source '${persona_source}'"
    exit 1
fi

echo "========== Settings =========="
echo "Dataset:        ${dataset}"
echo "Data name:      ${data_name}"
echo "Data dir:       ${data_dir}"
echo "Persona source: ${persona_source}"
echo "Persona dir:    ${persona_dir}"
echo "History dir:    ${history_dir}"
echo "=============================="


echo "=========================================="
echo "Batch Inference Experiment"
echo "=========================================="
echo "Data: ${data_name}"
echo "Model: ${sim_model}"
echo "S3 Bucket: ${S3_BUCKET}"
echo "S3 Prefix: ${S3_PREFIX}/${data_name}"
echo "=========================================="

python persona_inference_batch.py \
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
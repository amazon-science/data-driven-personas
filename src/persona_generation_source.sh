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

# Movie
# domain='movie' # shopping movie
# data_dir = "dir to IMDB_per_user.json"

# Shopping
# domain="shopping"
# data_dir = "dir to amazon_purchase_per_user.json"

# GSS 
# domain='gss' # opinionqa
# data_dir = "dir to ${domain}_per_user_with_gt.json"

# GSS Split - Individual
# percent='80' # 40% 60% 80%
# domain="gss_${percent}_ind2"
# data_dir = "dir to ${domain}_per_user_with_gt.json"

# GSS Split - Question
percent='80' # 40% 60% 80%
domain="gss_${percent}_q1"
data_dir = "dir to ${domain}_per_user_with_gt.json"

# OpinionQA
# domain='opinionqa_W29' # opinionqa
# wave="29"
# data_dir="dir to ${domain}_per_user_with_gt.json"


# Survey
# study_id='nj5dx' # nj5dx nb4xg
# domain="survey_${study_id}" # shopping movie
# data_dir="dir to socsci_${study_id}_per_respondent.json"

sim_model='claude3.7'
max_token=50000  #131072
num_history=all 
format='newfree_gt' # gtmatch
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
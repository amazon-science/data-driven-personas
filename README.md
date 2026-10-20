# Data-Driven Persona Simulation

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

A framework for generating and evaluating synthetic personas (induced from human behavior data) to simulate survey responses across demographic groups using Large Language Models (LLMs).

## Overview

This project enables researchers to:

- Generate diverse, realistic personas induced from human behavior data (e.g., movie reviews, shopping history, survey responses)
- Simulate survey responses using persona-conditioned LLMs
- Evaluate simulation quality using multiple statistical metrics (TVD, KL-divergence, Wasserstein distance)
- Analyze results across demographic groups and datasets

## Project Structure

```text
data-driven-personas/
├── src/                                 # Core source code and example run scripts
│   ├── persona_generation_source.py      # Generate personas from human behavior data
│   ├── persona_inference_batch.py        # Persona-conditioned batch inference + evaluation
│   ├── persona_inference_split.py        # Inference with personas for in-domain splits
│   ├── single_inference_batch.py         # Baseline (no-persona) inference
│   ├── prompt_util.py                    # Prompt generation utilities
│   ├── model_input_formatter.py          # Model-specific input formatting
│   ├── util.py                           # Helper functions (parsing, metrics)
│   ├── parallel_bedrock.py               # Parallel Amazon Bedrock inference
│   ├── batch_inference_utils.py          # Bedrock batch inference utilities
│   └── *.sh                              # Example run configurations
├── dataset_preprocess/                  # Dataset preprocessing notebooks
├── requirements.txt
└── pyproject.toml
```

Datasets are **not** included in this repository. Download them from the original sources listed under
[Dataset Preparation](#dataset-preparation) and place them under `dataset/`.

## Key Features

### 1. Persona Generation

Generate synthetic personas from diverse human behavior data:

- **free**: Flexible biographical descriptions
- **newfree**: Simulation-optimized personas for predictive fidelity
- **newfree_gt**: Simulation-optimized personas with added ground-truth (GT) features from behavior data  
  (Only available for OpinionQA and GSS)

### 2. Survey Simulation

Simulate survey responses using:

- **token_prob**: Direct answer selection
- **verbalized**: Distribution estimation over answer options

### 3. Evaluation Metrics

- **SimScore**: Similarity to the ground-truth distribution compared to a uniform baseline (SimBench; Hu et al., 2025)
- **TVD (Total Variation Distance)**: Statistical distance relative to the ground-truth distribution
- **KL-divergence**: Information-theoretic divergence relative to the ground-truth distribution
- **Wasserstein Distance (WD)**: Earth mover’s distance relative to the ground-truth distribution

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The example `.sh` scripts in `src/` activate `../venv` and read a `.aws_batch_config` file from the
directory they are run in. Create that file yourself with your own values, for example:

```bash
export S3_BUCKET=<your-bucket>
export S3_PREFIX=<your-prefix>
export AWS_PROFILE=<your-aws-profile>
export BEDROCK_BATCH_ROLE_ARN=<IAM role ARN that Bedrock batch inference can assume>
```

You need your own AWS account with access to the Amazon Bedrock models you want to use.

## Usage

### Generate Personas

```bash
python src/persona_generation_source.py \
  --data_dir dataset/gss/gss_per_user_with_gt.json \
  --domain <gss|opinionqa_W26|opinionqa_W29|movie|shopping> \
  --num_history <all|n> \
  --sim_model claude3.7 \
  --max_token 50000 \
  --format <newfree|newfree_gt> \
  --proc 15 \
  --feature_set opinionqa
```

Alternatively:

```bash
bash src/persona_generation_source.sh
```

> The `.sh` scripts include example configurations.

### Single Inference (Baseline)

```bash
python src/single_inference_batch.py \
  --data_dir dataset/gss/gss_grouped.json \
  --sim_model <claude3.7|llama3.3-70b|qwen3-32b> \
  --output_type <verbalized|token_prob> \
  --max_token 1024 \
  --data_name gss \
  --eval_only false \
  --num_inference 1 \
  --temperature 0.0 \
  --s3_bucket "${S3_BUCKET}" \
  --s3_prefix "${S3_PREFIX}/gss" \
  --check_interval 60 \
  --aws_profile "${AWS_PROFILE}"
```

Alternatively:

```bash
bash src/single_inference.sh
```

### Run Synthetic Persona-Conditioned Inference

```bash
python src/persona_inference_batch.py \
  --data_dir dataset/opinionqa/American_Trends_Panel_W26/opinionqa_W26_per_user_with_gt.json \
  --persona_dir output/personas/opinionqa_W26_personas.json \
  --output_dir output/results/ \
  --sim_model <claude3.7|llama> \
  --output_type verbalized \
  --format newfree \
  --num_persona 10 \
  --persona_prompt_type bio \
  --max_token 512 \
  --data_name opinionqa_W26 \
  --s3_bucket your-bucket \
  --proc 30
```

Alternatively:

```bash
bash src/persona_inference_batch.sh
```

### Evaluation Only

```bash
python src/persona_inference_batch.py \
  --data_dir output/results/opinionqa_W26_results.json \
  --persona_dir output/personas/opinionqa_W26_personas.json \
  --output_dir output/evaluation/ \
  --sim_model anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --output_type verbalized \
  --format newfree \
  --num_persona 10 \
  --eval_only true \
  --data_name opinionqa_W26 \
  --s3_bucket your-bucket \
  --proc 30
```

## Configuration

### Persona Prompt Types

- `bio`: Short biographical description
- `bio_history`: Biography + previous behavior data

### Output Types

- `verbalized`: Generate probability distributions over answer options
- `token_prob`: Select a single answer option

### Supported Datasets

- **OpinionQA**: American Trends Panel surveys (e.g., W26, W29, W32)
  - Behavior data: `dataset/opinionqa/grouped/opinionqa_W${wave}_per_user_with_gt.json`
  - Test data: `dataset/opinionqa/grouped/opinionqa_W${wave}_grouped.json`

- **GSS**: General Social Survey
  - Behavior data: `dataset/gss/gss_per_user_with_gt.json`
  - Test data: `dataset/gss/gss_grouped.json`

- **SocSci210**: Social science surveys
  - Behavior data: `dataset/socsci/socsci_${study_id}_per_respondent.json`
  - Test data: `dataset/socsci/grouped_socsci_${study_id}.json`

- **IMDb**: Movie reviews
  - Behavior data: `dataset/IMDB_naacl/IMDB_per_user.json`

- **E-commerce**: Amazon user purchase history
  - Behavior data: `dataset/amazon_purchase/amazon_purchase_per_user.json`

- **SimBench_opinionqa**: Multi-source benchmark (Afrobarometer, ESS, ISSP, LatinoBarometro)
  - Test data: `dataset/simbench/SimBenchGrouped_opinionqa.json`

## Evaluation Output

Results are saved in CSV format with:

- Overall metrics across all demographic groups
- Per-demographic-group breakdowns
- Top-5 best- and worst-performing personas

Example output structure:

| Method                         | Num Instances | SimScore |  TVD  | KL-divergence |  WD  | AVG None_cnt |
|-------------------------------|--------------:|---------:|------:|--------------:|-----:|-------------:|
| 10_bio_personas_verbalized_100% |           150 |     85.2 | 0.148 |         0.234 | 0.156 |          0.0 |

## Dataset Preparation

### GSS

- Original GSS data was downloaded from `https://gss.norc.org/get-the-data/spss.html`

### OpinionQA

- Original OpinionQA data was downloaded from `https://github.com/tatsu-lab/opinions_qa`

### SocSci210

- Original SocSci210 data was downloaded from `https://huggingface.co/datasets/socratesft/SocSci210`

### IMDb (Movie Reviews)

- Original IMDb data was downloaded from `https://github.com/granelle/naacl24-user-sim`

### E-commerce (Shopping History)

- Original e-commerce data was downloaded from `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY`

### Test Dataset Construction

- The GSS and OpinionQA datasets were preprocessed into group-level distributions using `dataset_preprocess/260113_opinionqa_gss_test_data_preprocess.ipynb`
- The SocSci210 dataset was preprocessed into group-level distributions using `dataset_preprocess/251114_SOCSCI_dataset.ipynb`

## Key Utilities

### `util.py`

- `parse_option()`: Extract an LLM answer from the output
- `parse_and_normalize_json_response()`: Parse probability distributions
- `simscore()`: Calculate similarity score
- `KL_divergence()`: KL-divergence calculation
- `Wasserstein_distance()`: Wasserstein distance calculation

### `prompt_util.py`

- `generate_prompt()`: Create survey prompts with persona context
- `generate_persona_prompt()`: Create persona generation prompts
- `get_dataset_context()`: Add dataset-specific context

## AWS Integration

The framework uses AWS Bedrock for LLM inference:

- Parallel processing with a configurable worker count
- Batch inference support
- S3 integration for large-scale experiments

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{lee2026datadriven,
      title={Data-Driven Personas for Survey Simulation: Insights into Simulation Alignment Across Data-Access Regimes}, 
      author={Dongryeol Lee and Weronika {\L}ajewska and Leonardo Perelli and Saab Mansour},
      booktitle={Proceedings of the Workshop on Research on Agent Language Models (REALM) at EMNLP 2026},
      year={2026},
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

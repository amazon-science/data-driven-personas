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

## Quick Start: Reproduce a Baseline

This walkthrough reproduces the B1 baseline (no demographic context) for Llama-3.3-70B on OpinionQA Wave 29, reported in Table 5 of the paper. It needs no persona generation. Run all commands from the repository root.

1. Install the dependencies (see [Installation](#installation)) and configure AWS credentials with Amazon Bedrock access in `us-east-1`.

2. Download the OpinionQA Wave 29 human responses (`info.csv`, `metadata.csv`, `responses.csv`) from the `human_resp` bundle of the [OpinionQA CodaLab worksheet](https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69):

   ```bash
   mkdir -p dataset/opinionqa/American_Trends_Panel_W29
   for f in info.csv metadata.csv responses.csv; do
     curl -L -o dataset/opinionqa/American_Trends_Panel_W29/$f \
       "https://worksheets.codalab.org/rest/bundles/0x050b7e72abb04d1f9b493c1743e580cf/contents/blob/human_resp/American_Trends_Panel_W29/$f"
   done
   ```

3. Build the per-user and grouped test files. Keep `wave='29'` (the default) and run, from the repository root:
   - the "OpinionQA preprocess" section of `dataset_preprocess/260107_gss_opinionqa_persona_preprocess.ipynb`, which writes `opinionqa_W29_per_user_with_gt.json`;
   - the "OpinionQA preprocess" section of `dataset_preprocess/260113_opinionqa_gss_test_data_preprocess.ipynb`, which writes `opinionqa_W29_grouped.json` (77 questions, 3,868 test instances).

4. Run baseline inference and evaluation:

   ```bash
   export BEDROCK_BATCH_ROLE_ARN=<IAM role ARN that Bedrock batch inference can assume>
   python src/single_inference_batch.py \
     --data_dir dataset/opinionqa/American_Trends_Panel_W29/opinionqa_W29_grouped.json \
     --sim_model llama3.3-70b --output_type verbalized --max_token 1024 \
     --data_name opinionqa_W29 --eval_only false --num_inference 1 --temperature 0.0 \
     --s3_bucket <your-bucket> --s3_prefix <your-prefix>/opinionqa_W29 \
     --check_interval 60 --aws_profile <your-aws-profile>
   ```

   The script calls Bedrock directly for the 77 B1 prompts, then submits one Bedrock batch job (3,868 prompts) for B2 and B3. The batch job typically finishes in 15 to 30 minutes and uses about 0.9M input and 0.1M output tokens.

5. Results are written to `out/b1_single/llama3.3-70b/simbench_opinionqa_W29_verbalized.csv`. Expected values (small run-to-run differences are possible):

   | Method | Paper TVD | Paper WD | Reproduced TVD | Reproduced WD |
   |---|---:|---:|---:|---:|
   | B1 (Without Description) | 0.200 | 0.295 | 0.199 | 0.297 |
   | B2 (General Description) | 0.188 | 0.279 | 0.193 | 0.283 |
   | B3 (Demographic Description) | 0.181 | 0.264 | 0.181 | 0.267 |

   Claude 3.7 Sonnet, the main simulator in the paper, has reached end of life on Amazon Bedrock, so Table 1 can no longer be reproduced exactly. Llama-3.3-70B and Qwen3-32B (Table 5) are still available.

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

- **OpinionQA**: American Trends Panel surveys (e.g., W26, W29)
  - Behavior data: `dataset/opinionqa/American_Trends_Panel_W${wave}/opinionqa_W${wave}_per_user_with_gt.json`
  - Test data: `dataset/opinionqa/American_Trends_Panel_W${wave}/opinionqa_W${wave}_grouped.json`

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

Example (`simbench_opinionqa_W29_verbalized.csv`, Llama-3.3-70B, from the [Quick Start](#quick-start-reproduce-a-baseline)):

| Method | SimScore | TVD | KL-divergence | WD |
|---|---:|---:|---:|---:|
| Without Description | 48.86 | 0.199 | 0.158 | 0.297 |
| General Description | 51.35 | 0.193 | 0.141 | 0.283 |
| Demographic Description | 53.39 | 0.181 | 0.154 | 0.267 |

## Dataset Preparation

### GSS

- Original GSS data was downloaded from `https://gss.norc.org/get-the-data/spss.html`

### OpinionQA

- Original OpinionQA data is described at `https://github.com/tatsu-lab/opinions_qa`. The per-wave human responses (`human_resp/American_Trends_Panel_W*/{info,metadata,responses}.csv`) are in the `human_resp` bundle of the [OpinionQA CodaLab worksheet](https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69). The [Quick Start](#quick-start-reproduce-a-baseline) has a download command.

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

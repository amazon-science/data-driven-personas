# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import boto3
import os
import json
import argparse
import random
from util import *
from prompt_util import *
from tqdm import tqdm
from pathlib import Path
import csv
from datetime import datetime
import numpy as np
from batch_inference_utils import run_batch_inference

# Use AWS profile authentication (set in bash script)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to input JSON data file')
    parser.add_argument('--sim_model', required=True, help='Bedrock model ID')
    parser.add_argument('--output_type', required=True, help='Output type for prompts')
    parser.add_argument('--max_token', type=int, required=True, help='Maximum tokens for generation')
    parser.add_argument('--data_name', required=True, help='Name for output files')
    parser.add_argument('--eval_only', default='false', help='Run evaluation only (true/false)')
    parser.add_argument('--num_inference', type=int, default=1, required=True, help='Number of inferences')
    parser.add_argument('--temperature', type=float, default=0.7, required=True, help='Temperature')
    # Batch inference arguments
    parser.add_argument('--s3_bucket', required=True, help='S3 bucket for batch inference')
    parser.add_argument('--s3_prefix', default='batch-inference/experiments', help='S3 prefix for batch files')
    parser.add_argument('--check_interval', type=int, default=60, help='Seconds between job status checks')
    parser.add_argument('--aws_profile', default='shopper-agents-role', help='AWS profile name')
    
    args = parser.parse_args()
    return args

model_id_to_bedrock_id = {
    "claude3.7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "llama3.1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
    "llama3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    'deepseek-r1': "us.deepseek.r1-v1:0",
    'qwen3-32b': "qwen.qwen3-32b-v1:0"
}


def main(args):
    print("="*60)
    print("Batch Inference Experiment")
    print("="*60)
    print(f"Data: {args.data_dir}")
    print(f"Model: {args.sim_model}")
    print(f"Output Type: {args.output_type}")
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"S3 Prefix: {args.s3_prefix}")
    print(f"Eval Only: {args.eval_only}")
    print("="*60)

    random.seed(1004)
    
    # Load data from file
    with open(args.data_dir, 'r') as f:
        data = json.load(f) 
    REGION = 'us-east-1'
    # Use AWS profile for authentication
    session = boto3.Session(profile_name=args.aws_profile)
    s3_client = session.client('s3', region_name=REGION)
    bedrock_client = session.client('bedrock', region_name=REGION)
    client = session.client(
        service_name='bedrock-runtime',
        region_name=REGION
    )          
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_idx=0
    for idx, d in enumerate(data):
        if len(d['group_prompt_variable_map']):
            start_idx=idx
            break
    
    print(f"{start_idx} unique questions")

    directory=os.getcwd()
    if args.eval_only.lower()=='false':
        # Without description
        # ============================================
        # Batch 2: Process first 20 items without description
        # ============================================
        for run in range(args.num_inference):
            print(f"\nProcessing batch 1: First {start_idx} items without description...")
            print("Generating prompts...")
            
            for d in data[:start_idx]:
                if '{}_response_noprompt'.format(args.sim_model) not in d.keys():
                    system_prompt, user_prompt = generate_prompt(d, 'verbalized')
                    if args.sim_model.startswith("claude"):
                        body = {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": args.max_token,
                            "temperature": args.temperature,
                            "system": "You are a group of individuals.",
                            "messages": [{"role": "user", "content": user_prompt}]
                        }
                        d['{}_response_noprompt_input'.format(args.sim_model)] = body
                        response = client.invoke_model(
                            modelId=model_id_to_bedrock_id[args.sim_model],
                            body=json.dumps(body)
                        )
                        response_body = json.loads(response['body'].read())
                        d['{}_response_noprompt_{}'.format(args.sim_model, run)] = response_body['content'][0]['text']
                        
                    elif args.sim_model.startswith("llama"):
                        body = {
                            "prompt": f"You are a group of individuals.\n\n{user_prompt}",
                            "max_gen_len": args.max_token,
                            "temperature": args.temperature,
                            "top_p": 0.9
                        }
                        d['{}_response_noprompt_input'.format(args.sim_model)] = body
                        response = client.invoke_model(
                            modelId=model_id_to_bedrock_id[args.sim_model],
                            body=json.dumps(body)
                        )
                        response_body = json.loads(response['body'].read())
                        d['{}_response_noprompt_{}'.format(args.sim_model, run)] = response_body['generation']
                        
                    elif args.sim_model.startswith("nova"):
                        messages = [
                            {
                                "role": "user",
                                "content": [{"text": user_prompt}]
                            }
                        ]
                        body = {
                            "messages": messages,
                            "system": [{"text": "You are a group of individuals."}],  # Top-level, not in messages
                            "inferenceConfig": {
                                "max_new_tokens": args.max_token,
                                "temperature": args.temperature,
                                "topP": 0.9
                            }
                        }
                        d['{}_response_noprompt_input'.format(args.sim_model)] = body
                        response = client.invoke_model(
                            modelId=model_id_to_bedrock_id[args.sim_model],
                            body=json.dumps(body)
                        )
                        response_body = json.loads(response['body'].read())
                        d['{}_response_noprompt_{}'.format(args.sim_model, run)] = response_body['output']['message']['content'][0]['text']

                    elif args.sim_model.startswith("qwen"):
                        # Qwen3 on Bedrock uses messages format
                        body = {
                            "messages": [
                                {"role": "system", "content": "You are a group of individuals."},
                                {"role": "user", "content": user_prompt}
                            ],
                            "max_tokens": args.max_token,
                            "temperature": args.temperature,
                            "top_p": 0.9
                        }
                        d['{}_response_noprompt_input'.format(args.sim_model)] = body
                        response = client.invoke_model(
                            modelId=model_id_to_bedrock_id[args.sim_model],
                            body=json.dumps(body)
                        )
                        response_body = json.loads(response['body'].read())
                        # Extract response - adjust path if needed
                        d['{}_response_noprompt_{}'.format(args.sim_model, run)] = response_body['choices'][0]['message']['content']


            with open('{}/out/{}'.format(directory, 'simbench_{}_{}.json'.format(args.data_name, args.output_type)), 'w') as f:
                json.dump(data, f)
            
            # ============================================
            # Batch 1: Process all data with default prompt
            # ============================================

            print("Processing batch 2: All data with default prompt...")
            print("Generating prompts...")
            
            system_prompts_batch1 = []
            user_prompts_batch1 = []
            record_ids_batch1 = []
            indices_batch1 = []
            
            print("Generating prompts...")
            for idx, d in enumerate(tqdm(data)):
                system_prompt, user_prompt = generate_prompt(d, args.output_type)
                system_prompts_batch1.append(system_prompt)
                user_prompts_batch1.append(user_prompt)
                record_ids_batch1.append(f"{idx}_{d.get('question_idx', idx)}")
                indices_batch1.append(idx)
            
            print(f"Found {len(system_prompts_batch1)} items to process")

            if system_prompts_batch1:
                results_map_batch1 = run_batch_inference(
                    s3_client=s3_client,
                    bedrock_client=bedrock_client,
                    system_prompts=system_prompts_batch1,
                    user_prompts=user_prompts_batch1,
                    record_ids=record_ids_batch1,
                    model_id=args.sim_model,
                    bucket=args.s3_bucket,
                    s3_prefix=args.s3_prefix,
                    max_tokens=args.max_token,
                    job_suffix=f'default_prompt_run{run}',
                    timestamp=timestamp,
                    check_interval=args.check_interval
                )
                
                # Merge results back to data
                print("Merging results...")
                for idx, record_id in zip(indices_batch1, record_ids_batch1):
                    if record_id in results_map_batch1:
                        data[idx]['{}_response_{}'.format(args.sim_model, run)] = results_map_batch1[record_id]
                print(f"✅ Batch 1 complete: {len(results_map_batch1)} responses")


            
            if not (Path.cwd() / 'out' / 'b1_single' / args.sim_model).exists():
                os.makedirs(('{}/out/b1_single/{}'.format(directory, args.sim_model)), exist_ok=True)
            with open('{}/out/b1_single/{}/{}'.format(directory, args.sim_model, 'simbench_{}_{}.json'.format(args.data_name, args.output_type)), 'w') as f:
                json.dump(data, f)

        
       

    # # Evaluation
    # # Without description
    # csv_header=[['Evaluator: {} result'.format(args.sim_model)]]
    # csv_header.append(['Method','SimScore', 'TVD', 'KL-divergence', "WD"])
    # base_distribution={}
    # for d in data[:start_idx]:
    #     expected_keys=list(d['human_answer'].keys())
    #     aggregated_dist = [0.0] * len(expected_keys)
    #     valid_count=0
    #     for run in range(args.num_inference):
    #         parsed=parse_and_normalize_json_response(d['{}_response_noprompt_{}'.format(args.sim_model, run)], expected_keys)
    #         if parsed is not None:
    #             assert(len(parsed)==len(aggregated_dist))
    #             valid_count+=1
    #             for i in range(len(parsed)):
    #                 aggregated_dist[i] += parsed[i]
    #     pred_distribution=[val / valid_count for val in aggregated_dist]
    #     base_distribution[d['question_idx']]=pred_distribution


    # sim_score=0
    # tvd_tot=0
    # kl_tot=0
    # wd_tot=0
    # none_cnt=0
    # for idx, d in enumerate(data[start_idx:]):
    #     expected_keys=list(d['human_answer'].keys())
    #     gt_distribution=[]
    #     for key in d['human_answer'].keys():
    #         gt_distribution.append(d['human_answer'][key]/100)
    #     pred_distribution=base_distribution[d['question_idx']]
    #     if pred_distribution is not None:
    #         tvd, sim= simscore(gt_distribution, pred_distribution)
    #         kl=KL_divergence(gt_distribution, pred_distribution)
    #         wd=Wasserstein_distance(gt_distribution, pred_distribution)
    #         kl_tot+=kl
    #         sim_score+=sim
    #         tvd_tot+=tvd
    #         wd_tot+=wd

    #     else:
    #         none_cnt+=1
    # print('Without Description')
    # print('SimScore: {}'.format(round(sim_score/(len(data[start_idx:])-none_cnt), 2)))
    # print('TVD: {}'.format(round(tvd_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('KL-divergence: {}'.format(round(kl_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('Wasserstein Distance: {}'.format(round(wd_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('Number of Errors: {}'.format(none_cnt))
    # print()
    # csv_header.append(['Without Description', round(sim_score/(len(data[start_idx:])-none_cnt), 2), round(tvd_tot/(len(data[start_idx:])-none_cnt), 3), round(kl_tot/(len(data[start_idx:])-none_cnt), 3), round(wd_tot/(len(data[start_idx:])-none_cnt), 3)])

    # # With general description
    # base_distribution={}
    # for idx, d in enumerate(data[:start_idx]):
    #     expected_keys=list(d['human_answer'].keys())
    #     aggregated_dist = [0.0] * len(expected_keys)
    #     valid_count=0
    #     for run in range(args.num_inference):
    #         parsed=parse_and_normalize_json_response(d['{}_response_{}'.format(args.sim_model, run)], expected_keys)
    #         if parsed is not None:
    #             assert(len(parsed)==len(aggregated_dist))
    #             valid_count+=1
    #             for i in range(len(parsed)):
    #                 aggregated_dist[i] += parsed[i]
    #     pred_distribution=[val / valid_count for val in aggregated_dist]
    #     base_distribution[d['question_idx']]=pred_distribution

    # sim_score=0
    # tvd_tot=0
    # kl_tot=0
    # wd_tot=0
    # none_cnt=0

    # for d in data[start_idx:]:
    #     expected_keys=list(d['human_answer'].keys())
    #     gt_distribution=[]
    #     for key in d['human_answer'].keys():
    #         gt_distribution.append(d['human_answer'][key]/100)
    #     pred_distribution=base_distribution[d['question_idx']]
    #     if pred_distribution is not None:
    #         tvd, sim= simscore(gt_distribution, pred_distribution)
    #         kl=KL_divergence(gt_distribution, pred_distribution)
    #         wd=Wasserstein_distance(gt_distribution, pred_distribution)
    #         kl_tot+=kl
    #         sim_score+=sim
    #         tvd_tot+=tvd
    #         wd_tot+=wd
    #     else:
    #         none_cnt+=1
        
    # print('General Description')
    # print('SimScore: {}'.format(round(sim_score/(len(data[start_idx:])-none_cnt), 2)))
    # print('TVD: {}'.format(round(tvd_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('KL-divergence: {}'.format(round(kl_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('Wasserstein Distance: {}'.format(round(wd_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('Number of Errors: {}'.format(none_cnt))

    # print()
    # csv_header.append(['General Description', round(sim_score/(len(data[start_idx:])-none_cnt), 2), round(tvd_tot/(len(data[start_idx:])-none_cnt), 3), round(kl_tot/(len(data[start_idx:])-none_cnt), 3), round(wd_tot/(len(data[start_idx:])-none_cnt), 3)])

    # # With demographic description
    # sim_score=0
    # tvd_tot=0
    # kl_tot=0
    # wd_tot=0
    # none_cnt=0
    # for d in data[start_idx:]:
    #     expected_keys=list(d['human_answer'].keys())
    #     gt_distribution=[]
    #     for key in d['human_answer'].keys():
    #         gt_distribution.append(d['human_answer'][key]/100)

    #     aggregated_dist = [0.0] * len(expected_keys)
    #     valid_count=0
    #     for run in range(args.num_inference):
    #         parsed=parse_and_normalize_json_response(d['{}_response_{}'.format(args.sim_model, run)], expected_keys)
    #         if parsed is not None:
    #             assert(len(parsed)==len(aggregated_dist))
    #             valid_count+=1
    #             for i in range(len(parsed)):
    #                 aggregated_dist[i] += parsed[i]
    #     if valid_count==0:
    #         pred_distribution=None
    #     else:
    #         pred_distribution=[val / valid_count for val in aggregated_dist]
        
    #     if pred_distribution is not None:
    #         tvd, sim= simscore(gt_distribution, pred_distribution)
    #         kl=KL_divergence(gt_distribution, pred_distribution)
    #         wd=Wasserstein_distance(gt_distribution, pred_distribution)
    #         kl_tot+=kl
    #         sim_score+=sim
    #         tvd_tot+=tvd
    #         wd_tot+=wd
    #     else:
    #         none_cnt+=1
        
    # print('Demographic Description')
    # print('SimScore: {}'.format(round(sim_score/(len(data[start_idx:])-none_cnt), 2)))
    # print('TVD: {}'.format(round(tvd_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('KL-divergence: {}'.format(round(kl_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('Wasserstein Distance: {}'.format(round(wd_tot/(len(data[start_idx:])-none_cnt), 3)))
    # print('Number of Errors: {}'.format(none_cnt))
    # csv_header.append(['Demographic Description', round(sim_score/(len(data[start_idx:])-none_cnt), 2), round(tvd_tot/(len(data[start_idx:])-none_cnt), 3), round(kl_tot/(len(data[start_idx:])-none_cnt), 3), round(wd_tot/(len(data[start_idx:])-none_cnt), 3)])   
    # with open('{}/out/b1_single/{}/{}'.format(directory, args.sim_model, 'simbench_{}_{}.csv'.format(args.data_name, args.output_type)), 'w', newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(csv_header)
    # Helper functions
    def get_demographic_key(group_prompt_variable_map):
        """Create a string key from demographic variables"""
        if not group_prompt_variable_map:
            return "No Demographics"
        sorted_items = sorted(group_prompt_variable_map.items())
        return ', '.join([f"{k}:{v}" for k, v in sorted_items])

    def calculate_shannon_entropy(distribution):
        """Calculate Shannon entropy of a probability distribution"""
        distribution = np.array(distribution)
        distribution = distribution[distribution > 0]
        if len(distribution) == 0:
            return 0.0
        return -np.sum(distribution * np.log2(distribution))

    # ============================================================================
    # COLLECT ALL DATA FIRST
    # ============================================================================

    demographic_results_noprompt = {}
    demographic_counts_noprompt = {}
    demographic_results_general = {}
    demographic_counts_general = {}
    demographic_results_demo = {}
    demographic_counts_demo = {}
    demographic_group_sizes = {}
    demographic_human_answers = {}
    demographic_key_group_sizes = {}

    # ============================================================================
    # WITHOUT DESCRIPTION - Data Collection
    # ============================================================================
    base_distribution_noprompt = {}
    for d in data[:start_idx]:
        expected_keys = list(d['human_answer'].keys())
        aggregated_dist = [0.0] * len(expected_keys)
        valid_count = 0
        for run in range(args.num_inference):
            parsed = parse_and_normalize_json_response(d['{}_response_noprompt_{}'.format(args.sim_model, run)], expected_keys)
            if parsed is not None:
                assert(len(parsed) == len(aggregated_dist))
                valid_count += 1
                for i in range(len(parsed)):
                    aggregated_dist[i] += parsed[i]
        pred_distribution = [val / valid_count for val in aggregated_dist] if valid_count > 0 else None
        base_distribution_noprompt[d['question_idx']] = pred_distribution

    sim_score_noprompt = 0
    tvd_tot_noprompt = 0
    kl_tot_noprompt = 0
    wd_tot_noprompt = 0
    none_cnt_noprompt = 0

    for idx, d in enumerate(data[start_idx:]):
        expected_keys = list(d['human_answer'].keys())
        gt_distribution = []
        for key in d['human_answer'].keys():
            gt_distribution.append(d['human_answer'][key] / 100)
        pred_distribution = base_distribution_noprompt[d['question_idx']]
        
        demo_key = get_demographic_key(d.get('group_prompt_variable_map', {}))
        
        if demo_key not in demographic_results_noprompt:
            demographic_results_noprompt[demo_key] = {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0}
            demographic_counts_noprompt[demo_key] = 0
            demographic_group_sizes[demo_key] = d.get('group_size', 'N/A')
            demographic_human_answers[demo_key] = []
        
        demographic_human_answers[demo_key].append(d['human_answer'])
        
        demo_map = d.get('group_prompt_variable_map', {})
        for key_name, key_value in demo_map.items():
            if key_name not in demographic_key_group_sizes:
                demographic_key_group_sizes[key_name] = {}
            if key_value not in demographic_key_group_sizes[key_name]:
                demographic_key_group_sizes[key_name][key_value] = d.get('group_size', 0)
        
        if pred_distribution is not None:
            tvd, sim = simscore(gt_distribution, pred_distribution)
            kl = KL_divergence(gt_distribution, pred_distribution)
            wd = Wasserstein_distance(gt_distribution, pred_distribution)
            kl_tot_noprompt += kl
            sim_score_noprompt += sim
            tvd_tot_noprompt += tvd
            wd_tot_noprompt += wd
            
            demographic_results_noprompt[demo_key]['sim'] += sim
            demographic_results_noprompt[demo_key]['tvd'] += tvd
            demographic_results_noprompt[demo_key]['kl'] += kl
            demographic_results_noprompt[demo_key]['wd'] += wd
            demographic_counts_noprompt[demo_key] += 1
        else:
            none_cnt_noprompt += 1
            demographic_results_noprompt[demo_key]['none_cnt'] += 1

    # ============================================================================
    # GENERAL DESCRIPTION - Data Collection
    # ============================================================================
    base_distribution_general = {}
    for idx, d in enumerate(data[:start_idx]):
        expected_keys = list(d['human_answer'].keys())
        aggregated_dist = [0.0] * len(expected_keys)
        valid_count = 0
        for run in range(args.num_inference):
            parsed = parse_and_normalize_json_response(d['{}_response_{}'.format(args.sim_model, run)], expected_keys)
            if parsed is not None:
                assert(len(parsed) == len(aggregated_dist))
                valid_count += 1
                for i in range(len(parsed)):
                    aggregated_dist[i] += parsed[i]
        pred_distribution = [val / valid_count for val in aggregated_dist] if valid_count > 0 else None
        base_distribution_general[d['question_idx']] = pred_distribution

    sim_score_general = 0
    tvd_tot_general = 0
    kl_tot_general = 0
    wd_tot_general = 0
    none_cnt_general = 0

    for d in data[start_idx:]:
        expected_keys = list(d['human_answer'].keys())
        gt_distribution = []
        for key in d['human_answer'].keys():
            gt_distribution.append(d['human_answer'][key] / 100)
        pred_distribution = base_distribution_general[d['question_idx']]
        
        demo_key = get_demographic_key(d.get('group_prompt_variable_map', {}))
        
        if demo_key not in demographic_results_general:
            demographic_results_general[demo_key] = {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0}
            demographic_counts_general[demo_key] = 0
        
        if pred_distribution is not None:
            tvd, sim = simscore(gt_distribution, pred_distribution)
            kl = KL_divergence(gt_distribution, pred_distribution)
            wd = Wasserstein_distance(gt_distribution, pred_distribution)
            kl_tot_general += kl
            sim_score_general += sim
            tvd_tot_general += tvd
            wd_tot_general += wd
            
            demographic_results_general[demo_key]['sim'] += sim
            demographic_results_general[demo_key]['tvd'] += tvd
            demographic_results_general[demo_key]['kl'] += kl
            demographic_results_general[demo_key]['wd'] += wd
            demographic_counts_general[demo_key] += 1
        else:
            none_cnt_general += 1
            demographic_results_general[demo_key]['none_cnt'] += 1

    # ============================================================================
    # DEMOGRAPHIC DESCRIPTION - Data Collection
    # ============================================================================
    sim_score_demo = 0
    tvd_tot_demo = 0
    kl_tot_demo = 0
    wd_tot_demo = 0
    none_cnt_demo = 0

    for d in data[start_idx:]:
        expected_keys = list(d['human_answer'].keys())
        gt_distribution = []
        for key in d['human_answer'].keys():
            gt_distribution.append(d['human_answer'][key] / 100)

        aggregated_dist = [0.0] * len(expected_keys)
        valid_count = 0
        for run in range(args.num_inference):
            parsed = parse_and_normalize_json_response(d['{}_response_{}'.format(args.sim_model, run)], expected_keys)
            if parsed is not None:
                assert(len(parsed) == len(aggregated_dist))
                valid_count += 1
                for i in range(len(parsed)):
                    aggregated_dist[i] += parsed[i]
        if valid_count == 0:
            pred_distribution = None
        else:
            pred_distribution = [val / valid_count for val in aggregated_dist]
        
        demo_key = get_demographic_key(d.get('group_prompt_variable_map', {}))
        
        if demo_key not in demographic_results_demo:
            demographic_results_demo[demo_key] = {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0}
            demographic_counts_demo[demo_key] = 0
        
        if pred_distribution is not None:
            tvd, sim = simscore(gt_distribution, pred_distribution)
            kl = KL_divergence(gt_distribution, pred_distribution)
            wd = Wasserstein_distance(gt_distribution, pred_distribution)
            kl_tot_demo += kl
            sim_score_demo += sim
            tvd_tot_demo += tvd
            wd_tot_demo += wd
            
            demographic_results_demo[demo_key]['sim'] += sim
            demographic_results_demo[demo_key]['tvd'] += tvd
            demographic_results_demo[demo_key]['kl'] += kl
            demographic_results_demo[demo_key]['wd'] += wd
            demographic_counts_demo[demo_key] += 1
        else:
            none_cnt_demo += 1
            demographic_results_demo[demo_key]['none_cnt'] += 1

    # # ============================================================================
    # # CALCULATE ENTROPY FOR ALL DEMOGRAPHICS
    # # ============================================================================
    # demographic_entropy = {}
    # for demo_key in demographic_human_answers.keys():
    #     if len(demographic_human_answers[demo_key]) > 0:
    #         answer_keys = list(demographic_human_answers[demo_key][0].keys())
    #         avg_distribution = {}
    #         for answer_key in answer_keys:
    #             avg_distribution[answer_key] = np.mean([
    #                 ha[answer_key] for ha in demographic_human_answers[demo_key]
    #             ])
    #         prob_distribution = [avg_distribution[k] / 100.0 for k in answer_keys]
    #         entropy = calculate_shannon_entropy(prob_distribution)
    #         demographic_entropy[demo_key] = entropy

    # ============================================================================
    # AGGREGATE PER-KEY RESULTS
    # ============================================================================

    demographic_key_results = {}
    demographic_key_human_answers = {}

    for d in data[start_idx:]:
        demo_map = d.get('group_prompt_variable_map', {})
        if not demo_map:
            continue
        for key_name in demo_map.keys():
            if key_name not in demographic_key_human_answers:
                demographic_key_human_answers[key_name] = []
            demographic_key_human_answers[key_name].append(d['human_answer'])

    for demo_key in demographic_results_demo.keys():
        if demo_key == "No Demographics":
            continue
        demo_dict = {}
        for item in demo_key.split(', '):
            if ':' in item:
                k, v = item.split(':', 1)
                demo_dict[k] = v
        
        for key_name in demo_dict.keys():
            if key_name not in demographic_key_results:
                demographic_key_results[key_name] = {
                    'Without Description': {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0},
                    'General Description': {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0},
                    'Demographic Description': {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0}
                }
            
            if demo_key in demographic_results_noprompt:
                demographic_key_results[key_name]['Without Description']['sim'] += demographic_results_noprompt[demo_key]['sim']
                demographic_key_results[key_name]['Without Description']['tvd'] += demographic_results_noprompt[demo_key]['tvd']
                demographic_key_results[key_name]['Without Description']['kl'] += demographic_results_noprompt[demo_key]['kl']
                demographic_key_results[key_name]['Without Description']['wd'] += demographic_results_noprompt[demo_key]['wd']
                demographic_key_results[key_name]['Without Description']['none_cnt'] += demographic_results_noprompt[demo_key]['none_cnt']
                demographic_key_results[key_name]['Without Description']['count'] += demographic_counts_noprompt.get(demo_key, 0)
            
            if demo_key in demographic_results_general:
                demographic_key_results[key_name]['General Description']['sim'] += demographic_results_general[demo_key]['sim']
                demographic_key_results[key_name]['General Description']['tvd'] += demographic_results_general[demo_key]['tvd']
                demographic_key_results[key_name]['General Description']['kl'] += demographic_results_general[demo_key]['kl']
                demographic_key_results[key_name]['General Description']['wd'] += demographic_results_general[demo_key]['wd']
                demographic_key_results[key_name]['General Description']['none_cnt'] += demographic_results_general[demo_key]['none_cnt']
                demographic_key_results[key_name]['General Description']['count'] += demographic_counts_general.get(demo_key, 0)
            
            demographic_key_results[key_name]['Demographic Description']['sim'] += demographic_results_demo[demo_key]['sim']
            demographic_key_results[key_name]['Demographic Description']['tvd'] += demographic_results_demo[demo_key]['tvd']
            demographic_key_results[key_name]['Demographic Description']['kl'] += demographic_results_demo[demo_key]['kl']
            demographic_key_results[key_name]['Demographic Description']['wd'] += demographic_results_demo[demo_key]['wd']
            demographic_key_results[key_name]['Demographic Description']['none_cnt'] += demographic_results_demo[demo_key]['none_cnt']
            demographic_key_results[key_name]['Demographic Description']['count'] += demographic_counts_demo.get(demo_key, 0)

    # demographic_key_entropy = {}
    # for key_name in demographic_key_human_answers.keys():
    #     if len(demographic_key_human_answers[key_name]) > 0:
    #         answer_keys = list(demographic_key_human_answers[key_name][0].keys())
    #         avg_distribution = {}
    #         for answer_key in answer_keys:
    #             avg_distribution[answer_key] = np.mean([
    #                 ha[answer_key] for ha in demographic_key_human_answers[key_name]
    #             ])
    #         prob_distribution = [avg_distribution[k] / 100.0 for k in answer_keys]
    #         entropy = calculate_shannon_entropy(prob_distribution)
    #         demographic_key_entropy[key_name] = entropy

    demographic_key_total_group_size = {}
    for key_name in demographic_key_group_sizes.keys():
        total_size = sum(demographic_key_group_sizes[key_name].values())
        demographic_key_total_group_size[key_name] = total_size

    sorted_demo_keys = sorted(demographic_results_demo.keys())

    # ============================================================================
    # SAVE MAIN CSV FILE (OVERALL RESULTS ONLY)
    # ============================================================================
    csv_header = [['Evaluator: {} result'.format(args.sim_model)]]
    csv_header.append(['Method', 'SimScore', 'TVD', 'KL-divergence', "WD"])

    csv_header.append(['Without Description', 
                    round(sim_score_noprompt / (len(data[start_idx:]) - none_cnt_noprompt), 2), 
                    round(tvd_tot_noprompt / (len(data[start_idx:]) - none_cnt_noprompt), 3), 
                    round(kl_tot_noprompt / (len(data[start_idx:]) - none_cnt_noprompt), 3), 
                    round(wd_tot_noprompt / (len(data[start_idx:]) - none_cnt_noprompt), 3)])

    csv_header.append(['General Description', 
                    round(sim_score_general / (len(data[start_idx:]) - none_cnt_general), 2), 
                    round(tvd_tot_general / (len(data[start_idx:]) - none_cnt_general), 3), 
                    round(kl_tot_general / (len(data[start_idx:]) - none_cnt_general), 3), 
                    round(wd_tot_general / (len(data[start_idx:]) - none_cnt_general), 3)])

    csv_header.append(['Demographic Description', 
                    round(sim_score_demo / (len(data[start_idx:]) - none_cnt_demo), 2), 
                    round(tvd_tot_demo / (len(data[start_idx:]) - none_cnt_demo), 3), 
                    round(kl_tot_demo / (len(data[start_idx:]) - none_cnt_demo), 3), 
                    round(wd_tot_demo / (len(data[start_idx:]) - none_cnt_demo), 3)])

    csv_file = '{}/out/b1_single/{}/{}'.format(directory, args.sim_model, 'simbench_{}_{}.csv'.format(args.data_name, args.output_type))

    with open(csv_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_header)

    # ============================================================================
    # SAVE PER-DEMOGRAPHIC CSV FILE
    # ============================================================================
    demo_csv_header = [['Per-Demographic Results - Evaluator: {} result'.format(args.sim_model)]]

    # Without Description - Per-Demographic
    demo_csv_header.append([])
    demo_csv_header.append(['=== WITHOUT DESCRIPTION - PER-DEMOGRAPHIC ==='])
    demo_csv_header.append(['Demographic', 'Group Size', 'Num Instances',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'None Count'])

    for demo_key in sorted_demo_keys:
        if demo_key in demographic_results_noprompt and demographic_counts_noprompt.get(demo_key, 0) > 0:
            demo_count = demographic_counts_noprompt[demo_key]
            group_size = demographic_group_sizes.get(demo_key, 'N/A')
            
            avg_sim = round(demographic_results_noprompt[demo_key]['sim'] / demo_count, 2)
            avg_tvd = round(demographic_results_noprompt[demo_key]['tvd'] / demo_count, 3)
            avg_kl = round(demographic_results_noprompt[demo_key]['kl'] / demo_count, 3)
            avg_wd = round(demographic_results_noprompt[demo_key]['wd'] / demo_count, 3)
            none_count = demographic_results_noprompt[demo_key]['none_cnt']
            
            demo_csv_header.append([demo_key, group_size, demo_count,  avg_sim, avg_tvd, avg_kl, avg_wd, none_count])

    # General Description - Per-Demographic
    demo_csv_header.append([])
    demo_csv_header.append(['=== GENERAL DESCRIPTION - PER-DEMOGRAPHIC ==='])
    demo_csv_header.append(['Demographic', 'Group Size', 'Num Instances',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'None Count'])

    for demo_key in sorted_demo_keys:
        if demo_key in demographic_results_general and demographic_counts_general.get(demo_key, 0) > 0:
            demo_count = demographic_counts_general[demo_key]
            group_size = demographic_group_sizes.get(demo_key, 'N/A')
            
            avg_sim = round(demographic_results_general[demo_key]['sim'] / demo_count, 2)
            avg_tvd = round(demographic_results_general[demo_key]['tvd'] / demo_count, 3)
            avg_kl = round(demographic_results_general[demo_key]['kl'] / demo_count, 3)
            avg_wd = round(demographic_results_general[demo_key]['wd'] / demo_count, 3)
            none_count = demographic_results_general[demo_key]['none_cnt']
            
            demo_csv_header.append([demo_key, group_size, demo_count,  avg_sim, avg_tvd, avg_kl, avg_wd, none_count])

    # Demographic Description - Per-Demographic
    demo_csv_header.append([])
    demo_csv_header.append(['=== DEMOGRAPHIC DESCRIPTION - PER-DEMOGRAPHIC ==='])
    demo_csv_header.append(['Demographic', 'Group Size', 'Num Instances',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'None Count'])

    for demo_key in sorted_demo_keys:
        if demo_key in demographic_results_demo and demographic_counts_demo.get(demo_key, 0) > 0:
            demo_count = demographic_counts_demo[demo_key]
            group_size = demographic_group_sizes.get(demo_key, 'N/A')
            
            avg_sim = round(demographic_results_demo[demo_key]['sim'] / demo_count, 2)
            avg_tvd = round(demographic_results_demo[demo_key]['tvd'] / demo_count, 3)
            avg_kl = round(demographic_results_demo[demo_key]['kl'] / demo_count, 3)
            avg_wd = round(demographic_results_demo[demo_key]['wd'] / demo_count, 3)
            none_count = demographic_results_demo[demo_key]['none_cnt']
            
            demo_csv_header.append([demo_key, group_size, demo_count,  avg_sim, avg_tvd, avg_kl, avg_wd, none_count])

    # Without Description - Per-Demographic-Key
    demo_csv_header.append([])
    demo_csv_header.append(['=== WITHOUT DESCRIPTION - PER-DEMOGRAPHIC-KEY ==='])
    demo_csv_header.append(['Demographic Key', 'Total Group Size', 'Group Size Breakdown', 'Num Instances',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'None Count'])

    for key_name in sorted(demographic_key_results.keys()):
        count = demographic_key_results[key_name]['Without Description']['count']
        if count > 0:
            total_group_size = demographic_key_total_group_size.get(key_name, 'N/A')
            group_size_breakdown = ""
            if key_name in demographic_key_group_sizes:
                breakdown_items = [f"{v}:{sz}" for v, sz in sorted(demographic_key_group_sizes[key_name].items())]
                group_size_breakdown = "; ".join(breakdown_items)
            
            avg_sim = round(demographic_key_results[key_name]['Without Description']['sim'] / count, 2)
            avg_tvd = round(demographic_key_results[key_name]['Without Description']['tvd'] / count, 3)
            avg_kl = round(demographic_key_results[key_name]['Without Description']['kl'] / count, 3)
            avg_wd = round(demographic_key_results[key_name]['Without Description']['wd'] / count, 3)
            none_count = demographic_key_results[key_name]['Without Description']['none_cnt']
            
            demo_csv_header.append([key_name, total_group_size, group_size_breakdown, count,  avg_sim, avg_tvd, avg_kl, avg_wd, none_count])

    # General Description - Per-Demographic-Key
    demo_csv_header.append([])
    demo_csv_header.append(['=== GENERAL DESCRIPTION - PER-DEMOGRAPHIC-KEY ==='])
    demo_csv_header.append(['Demographic Key', 'Total Group Size', 'Group Size Breakdown', 'Num Instances',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'None Count'])

    for key_name in sorted(demographic_key_results.keys()):
        count = demographic_key_results[key_name]['General Description']['count']
        if count > 0:
            total_group_size = demographic_key_total_group_size.get(key_name, 'N/A')
            group_size_breakdown = ""
            if key_name in demographic_key_group_sizes:
                breakdown_items = [f"{v}:{sz}" for v, sz in sorted(demographic_key_group_sizes[key_name].items())]
                group_size_breakdown = "; ".join(breakdown_items)
            
            avg_sim = round(demographic_key_results[key_name]['General Description']['sim'] / count, 2)
            avg_tvd = round(demographic_key_results[key_name]['General Description']['tvd'] / count, 3)
            avg_kl = round(demographic_key_results[key_name]['General Description']['kl'] / count, 3)
            avg_wd = round(demographic_key_results[key_name]['General Description']['wd'] / count, 3)
            none_count = demographic_key_results[key_name]['General Description']['none_cnt']
            
            demo_csv_header.append([key_name, total_group_size, group_size_breakdown, count,  avg_sim, avg_tvd, avg_kl, avg_wd, none_count])

    # Demographic Description - Per-Demographic-Key
    demo_csv_header.append([])
    demo_csv_header.append(['=== DEMOGRAPHIC DESCRIPTION - PER-DEMOGRAPHIC-KEY ==='])
    demo_csv_header.append(['Demographic Key', 'Total Group Size', 'Group Size Breakdown', 'Num Instances',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'None Count'])

    for key_name in sorted(demographic_key_results.keys()):
        count = demographic_key_results[key_name]['Demographic Description']['count']
        if count > 0:
            total_group_size = demographic_key_total_group_size.get(key_name, 'N/A')
            group_size_breakdown = ""
            if key_name in demographic_key_group_sizes:
                breakdown_items = [f"{v}:{sz}" for v, sz in sorted(demographic_key_group_sizes[key_name].items())]
                group_size_breakdown = "; ".join(breakdown_items)
            
            avg_sim = round(demographic_key_results[key_name]['Demographic Description']['sim'] / count, 2)
            avg_tvd = round(demographic_key_results[key_name]['Demographic Description']['tvd'] / count, 3)
            avg_kl = round(demographic_key_results[key_name]['Demographic Description']['kl'] / count, 3)
            avg_wd = round(demographic_key_results[key_name]['Demographic Description']['wd'] / count, 3)
            none_count = demographic_key_results[key_name]['Demographic Description']['none_cnt']
            
            demo_csv_header.append([key_name, total_group_size, group_size_breakdown, count, avg_sim, avg_tvd, avg_kl, avg_wd, none_count])

    demo_csv_file = '{}/out/b1_single/{}/{}'.format(directory, args.sim_model, 'simbench_{}_{}_per_demographic.csv'.format(args.data_name, args.output_type))

    with open(demo_csv_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(demo_csv_header)
        
if __name__ == '__main__':
    args = get_args()
    main(args)
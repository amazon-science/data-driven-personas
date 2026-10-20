# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import boto3
import os
import json
import argparse
import random
from util import *
from tqdm import tqdm
from pathlib import Path
import csv
from datetime import datetime
from batch_inference_utils import run_batch_inference, run_batch_inference_chunked
from parallel_bedrock import *
from prompt_util import *
import numpy as np
from collections import Counter
import re

# Use AWS profile authentication (set in bash script)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to input JSON data file')
    parser.add_argument('--output_dir', required=True, help='Path to output dir')
    parser.add_argument('--history_dir', default='')
    parser.add_argument('--persona_prompt_type', default='bio')
    parser.add_argument('--sim_model', required=True, help='Bedrock model ID')
    parser.add_argument('--output_type', required=True, help='Output type for prompts')
    parser.add_argument('--format', required=True)
    parser.add_argument('--persona_source', default='movie')
    parser.add_argument('--max_token', type=int, required=True, help='Maximum tokens for generation')
    parser.add_argument('--data_name', required=True, help='Name for output files')
    parser.add_argument('--eval_only', default='false', help='Run evaluation only (true/false)')
    parser.add_argument('--eval_from', default='first', help='Run evaluation from first / last persona')
    parser.add_argument('--num_persona', default=10, help='Number of generated personas')
    parser.add_argument('--persona_dir', required=True)
    parser.add_argument('--proc', type=int, required=True, default=30)
    # Batch inference arguments
    parser.add_argument('--s3_bucket', required=True, help='S3 bucket for batch inference')
    parser.add_argument('--s3_prefix', default='batch-inference/experiments', help='S3 prefix for batch files')
    parser.add_argument('--check_interval', type=int, default=60, help='Seconds between job status checks')
    parser.add_argument('--aws_profile',  help='AWS profile name')
    
    args = parser.parse_args()
    return args


def calculate_shannon_entropy(distribution):
    """
    Calculate Shannon entropy for a probability distribution.
    H = -Σ(p_i * log2(p_i))
    
    Args:
        distribution: list or array of probabilities (should sum to 1.0)
    
    Returns:
        entropy value (float)
    """
    dist = np.array(distribution)
    # Remove zero probabilities to avoid log(0)
    dist = dist[dist > 0]
    entropy = -np.sum(dist * np.log2(dist))
    return entropy


def calculate_tvd(pred_dist, gt_dist):
    """Calculate Total Variation Distance between two distributions."""
    return 0.5 * np.sum(np.abs(np.array(pred_dist) - np.array(gt_dist)))


def calculate_kl_divergence(pred_dist, gt_dist):
    """Calculate KL divergence with smoothing to avoid log(0)."""
    pred = np.array(pred_dist) + 1e-10
    gt = np.array(gt_dist) + 1e-10
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)
    return np.sum(gt * np.log(gt / pred))


def evaluate_persona_ranking(d, output_type, sim_model, expected_keys, gt_distribution, top_k=5):
    """
    Evaluate individual personas and return metrics for top-k best and worst personas.
    Uses pred_label which contains JSON strings of distributions.
    
    Returns:
        (best_metrics, worst_metrics) where each is (sim, tvd, kl, wd, none_cnt) or (None, ...) if not enough personas
    """
    persona_outputs = d.get('persona_outputs_{}'.format(output_type), [])
    
    # Dynamically adjust top_k based on available personas
    if len(persona_outputs) < 3:
        return (None, None, None, None, None), (None, None, None, None, None)
    
    # Use smaller k if we don't have enough personas
    actual_k = min(top_k, len(persona_outputs) // 2)
    actual_k = max(actual_k, 1)
    
    # Calculate TVD for each persona
    persona_tvds = []
    for idx, persona_output in enumerate(persona_outputs):
        pred_label = persona_output.get('pred_label', None)
        
        # Parse pred_label (it's a JSON string like ' {"A": 82, "B": 12, "C": 6}')
        pred_dist = None
        if pred_label is not None:
            if isinstance(pred_label, str):
                try:
                    pred_label = json.loads(pred_label.strip())
                except:
                    pass
            
            if isinstance(pred_label, dict):
                # Normalize to probabilities
                total = sum(pred_label.values())
                if total > 0:
                    pred_dist = [pred_label.get(key, 0) / total for key in expected_keys]
        
        # If parsing failed or invalid, use uniform distribution (high TVD)
        if pred_dist is None:
            pred_dist = [1.0 / len(expected_keys)] * len(expected_keys)
        
        # Calculate TVD for this persona
        tvd = calculate_tvd(pred_dist, gt_distribution)
        persona_tvds.append((tvd, pred_dist, idx))
    
    # Sort by TVD (ascending - best personas have lowest TVD)
    sorted_personas = sorted(persona_tvds, key=lambda x: x[0])
    
    # Get top-k best (lowest TVD) and worst (highest TVD)
    best_indices = [item[2] for item in sorted_personas[:actual_k]]
    worst_indices = [item[2] for item in sorted_personas[-actual_k:]]
    
    def evaluate_subset(indices, subset_name="subset"):
        # Aggregate distributions for this subset
        aggregated_dist = [0.0] * len(expected_keys)
        valid_count = 0
        
        for idx in indices:
            pred_label = persona_outputs[idx].get('pred_label', None)
            
            # Parse pred_label (JSON string)
            if pred_label is not None:
                if isinstance(pred_label, str):
                    try:
                        pred_label = json.loads(pred_label.strip())
                    except:
                        pred_label = None
                
                if isinstance(pred_label, dict):
                    total = sum(pred_label.values())
                    if total > 0:
                        for i, key in enumerate(expected_keys):
                            aggregated_dist[i] += pred_label.get(key, 0) / total
                        valid_count += 1
        
        if valid_count == 0:
            return 0.0, 0.5, 999.0, 999.0, len(indices)
        
        # Normalize aggregated distribution
        pred_dist = [val / valid_count for val in aggregated_dist]
        none_cnt = len(indices) - valid_count
        
        # Calculate metrics using same functions as regular evaluation
        tvd, sim = simscore(gt_distribution, pred_dist)
        kl = KL_divergence(gt_distribution, pred_dist)
        wd = Wasserstein_distance(gt_distribution, pred_dist)
        
        return sim, tvd, kl, wd, none_cnt
    
    best_metrics = evaluate_subset(best_indices, "BEST")
    worst_metrics = evaluate_subset(worst_indices, "WORST")
    
    return best_metrics, worst_metrics


def has_valid_persona(d, persona_indexing):
    """Check if a data entry has a matching persona in the persona file."""
    data_key = tuple(sorted(d['group_prompt_variable_map'].items()))
    return data_key in persona_indexing
def _norm_str(x: str) -> str:
    """Normalize strings for robust matching (case/whitespace)."""
    if x is None:
        return ""
    # keep it conservative: lower + strip + collapse spaces
    s = str(x).strip().lower()
    s = " ".join(s.split())
    return s

def normalize_demo_map(demo_map: dict) -> dict:
    """
    Normalize demographic map for matching between persona and target data.
    - keys: lower/strip
    - values:
      - strings: lower/strip/collapse spaces
      - numbers/bools: keep as-is
      - None: empty string
    """
    out = {}
    for k, v in (demo_map or {}).items():
        nk = _norm_str(k)
        if isinstance(v, str) or v is None:
            nv = _norm_str(v)
        else:
            # keep numeric/bool as-is (but still stringify? usually not needed)
            nv = v
        out[nk] = nv
    return out

def make_persona_key(demo_map: dict):
    """Canonical key for persona/data matching."""
    nm = normalize_demo_map(demo_map)
    # sort to make deterministic tuple key
    return tuple(sorted(nm.items(), key=lambda x: x[0]))


def main(args):
    print(args)
    print("="*60)
    print("Batch Inference Experiment")
    print("="*60)
    print(f"Data: {args.data_dir}")
    print(f"Model: {args.sim_model}")
    print(f"Output Type: {args.output_type}")
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"S3 Prefix: {args.s3_prefix}")
    print(f"Eval Only: {args.eval_only}")
    print(f"Num Persona: {args.num_persona}")
    print("="*60)
    REGION = 'us-east-1'
    # Use AWS profile for authentication
    session = boto3.Session(profile_name=args.aws_profile)
    s3_client = session.client('s3', region_name=REGION)
    bedrock_client = session.client('bedrock', region_name=REGION)
    random.seed(1004)
    
    # Load data from file
    with open(args.data_dir, 'r') as f:
        data = json.load(f) 
    with open(args.persona_dir, 'r') as f:
        personas = json.load(f)
    
    if 'history' in args.persona_prompt_type:
        with open(args.history_dir, 'r') as f:
            history =json.load(f)
    REGION = 'us-east-1'
    # Use profile if specified
    session = boto3.Session(profile_name=args.aws_profile)
    use_all_personas = (str(args.num_persona).lower() == 'all')
    fixed_num_persona = None if use_all_personas else int(args.num_persona)
    
    # Build persona indexing
    persona_indexing = {}
    for p_idx, p in enumerate(personas):
        #key = tuple(sorted(p['group_prompt_variable_map'].items()))
        key = make_persona_key(p.get('group_prompt_variable_map', {}))

        persona_indexing[key] = p_idx
    
    # ============================================
    # Identify missing demographics (do this ONCE, before inference or eval)
    # ============================================
    for idx, d in enumerate(data):
        if len(d['group_prompt_variable_map']):
            start_idx = idx
            break
    end_idx = len(data)
    
    # Find all missing demographic keys
    missing_demographic_keys = set()
    missing_demographic_counts = {}
    valid_data_count = 0
    
    for d in data[start_idx:]:
        #data_key = tuple(sorted(d['group_prompt_variable_map'].items()))
        data_key = make_persona_key(d.get('group_prompt_variable_map', {}))

        if data_key not in persona_indexing.keys():
            missing_demographic_keys.add(data_key)
            key_str = str(dict(data_key))
            missing_demographic_counts[key_str] = missing_demographic_counts.get(key_str, 0) + 1
        else:
            valid_data_count += 1
    
    # Print missing demographics summary
    if missing_demographic_keys:
        total_missing = sum(missing_demographic_counts.values())
        print("\n" + "="*60)
        print(f"⚠️  MISSING DEMOGRAPHICS DETECTED")
        print("="*60)
        print(f"Total entries with missing personas: {total_missing}")
        print(f"Total entries with valid personas: {valid_data_count}")
        print(f"\nMissing demographic breakdown:")
        for demo, count in sorted(missing_demographic_counts.items()):
            print(f"   {demo}: {count} entries")
        print(f"\n⚠️  These {total_missing} entries will be SKIPPED in inference and evaluation.")
        print("="*60 + "\n")
    else:
        print(f"\n✅ All {valid_data_count} data entries have matching personas.\n")
    
    if args.eval_only.lower() == 'false':
        print("Processing batch with persona-conditioned prompts...")
        print("Generating prompts...")
        
        system_prompts_batch1 = []
        user_prompts_batch1 = []
        record_ids = []  # For batch inference tracking
        
        # Track persona counts and which instances to skip
        persona_counts_per_instance = []
        instance_to_process = []
        skipped_instances = []

        # Generate prompts for each data instance
        for idx, d in enumerate(tqdm(data[start_idx:end_idx], desc="Generating prompts")):
            #persona_key = tuple(sorted(d['group_prompt_variable_map'].items()))
            persona_key = make_persona_key(d.get('group_prompt_variable_map', {}))

            # ============================================
            # SKIP if no matching persona exists
            # ============================================
            if persona_key not in persona_indexing:
                skipped_instances.append({
                    'index': idx,
                    'demographic': get_demographic_key(d['group_prompt_variable_map']),
                    'reason': 'No matching persona in persona file'
                })
                continue
            
            persona_data = personas[persona_indexing[persona_key]]
            
            # Determine number of personas for this instance
            if use_all_personas:
                num_personas_for_instance = len(persona_data['generated_persona'])
            else:
                num_personas_for_instance = min(fixed_num_persona, len(persona_data['generated_persona']))
            
            # Skip instances with 0 personas
            if num_personas_for_instance == 0:
                skipped_instances.append({
                    'index': idx,
                    'demographic': get_demographic_key(d['group_prompt_variable_map']),
                    'reason': 'No personas available'
                })
                continue
            
            persona_counts_per_instance.append(num_personas_for_instance)
            instance_to_process.append(idx)
            
            des = " ".join(get_dataset_context(args.data_name, d['group_prompt_template']))
            des += " "
            using_persona = persona_data['generated_persona'][:num_personas_for_instance]
            for i in range(num_personas_for_instance):
                user_behavior=""
                question_idx=0
                if 'history' in args.persona_prompt_type:
                    if 'opinionqa' in args.persona_source or 'gss' in args.persona_source:
                        user_idx=using_persona[i]['user_idx']
                        for review in history[user_idx]['responses']:
                            if len(review['Response']):
                                user_behavior += f"Survey Question {question_idx+1}: {review['Question']}\n"
                                user_behavior += f"Options for Question {question_idx+1}: {review['Possible_Answers']}\n"
                                user_behavior += f"User Response to Question {question_idx+1}: {review['Response']}\n"            
                                question_idx+=1
                        system_prompt, user_prompt = generate_prompt(
                            d, 
                            args.output_type, 
                            user_behavior,
                            using_persona[i],
                            'bio_history',
                            des
                        )
                    elif 'mp3dr' in args.persona_source or 'nj5dx' in args.persona_source:
                        user_idx=using_persona[i]['user_idx']
                        for review in history[user_idx]:
                            user_behavior += f"Survey Question {question_idx+1}: {review['stimuli']}\n"
                            user_behavior += f"User Response to Question {question_idx+1}: {review['response']}\n\n"            
                            question_idx+=1
                        system_prompt, user_prompt = generate_prompt(
                            d, 
                            args.output_type, 
                            user_behavior,
                            using_persona[i],
                            'bio_history',
                            des
                        )
                    elif 'shopping' in args.persona_source:
                        user_idx=using_persona[i]['user_idx']
                        num_reviews=min(50, len(history[user_idx]))
                        for his_idx in range(1, num_reviews+1):
                            his = history[user_idx][-his_idx]
                            user_behavior += f"[Recent Purchase - {his_idx}]\n"
                            user_behavior += f"Purchase Item: {his['Title']}\n"
                            user_behavior += f"Quantity: {his['Quantity']}\n"
                            user_behavior += f"Purchase Price Per Unit: {his['Purchase Price Per Unit']}\n"
                            user_behavior += f"Shipping Address State: {his['Shipping Address State']}\n\n"
                        system_prompt, user_prompt = generate_prompt(
                            d, 
                            args.output_type, 
                            user_behavior,
                            using_persona[i],
                            'bio_history_shopping',
                            des
                        )
                    elif 'movie' in args.persona_source:
                        user_idx=using_persona[i]['user_idx']
                        num_reviews=min(10, len(history[user_idx]))
                        for his_idx in range(num_reviews):
                            his = history[user_idx][his_idx]
                            user_behavior += f"Movie Title: {his['movie_title']}\n"
                            user_behavior += f"Review Title: {his['review_title']}\n"
                            user_behavior += f"Review: {his['review_content']}\n\n"
                        system_prompt, user_prompt = generate_prompt(
                            d, 
                            args.output_type, 
                            user_behavior,
                            using_persona[i],
                            'bio_history_movie',
                            des
                        )
                    else:
                        raise ValueError("Unknown persona source")
                else:
                    system_prompt, user_prompt = generate_prompt(
                        d, 
                        args.output_type, 
                        user_behavior,
                        using_persona[i],
                        args.persona_prompt_type,
                        des
                    )
                system_prompts_batch1.append(system_prompt)
                user_prompts_batch1.append(user_prompt)
                # Create unique record ID for batch inference
                record_ids.append(f"{idx}_{i}")
        # Print statistics about skipped instances
        if skipped_instances:
            print(f"\n⚠️  Skipped {len(skipped_instances)} instances total:")
            
            # Group by reason
            reason_counts = Counter(s['reason'] for s in skipped_instances)
            for reason, count in reason_counts.items():
                print(f"   - {reason}: {count} instances")
            
            # Group by demographic for missing persona reason
            missing_persona_skips = [s for s in skipped_instances if 'No matching persona' in s['reason']]
            if missing_persona_skips:
                print(f"\n   Demographics without personas:")
                demo_skip_counts = Counter(s['demographic'] for s in missing_persona_skips)
                for demo, count in sorted(demo_skip_counts.items()):
                    print(f"      - {demo}: {count} instances")
        
        # Print statistics about persona counts
        if persona_counts_per_instance:
            print(f"\n📊 Persona Statistics (excluding skipped instances):")
            print(f"   Instances to process: {len(persona_counts_per_instance)}")
            print(f"   Instances skipped: {len(skipped_instances)}")
            if use_all_personas:
                print(f"   Min personas per instance: {min(persona_counts_per_instance)}")
                print(f"   Max personas per instance: {max(persona_counts_per_instance)}")
                print(f"   Avg personas per instance: {sum(persona_counts_per_instance)/len(persona_counts_per_instance):.2f}")
            print(f"   Total prompts: {sum(persona_counts_per_instance)}")
        else:
            print("\n⚠️  WARNING: No instances to process! All instances were skipped.")
            return

        # Initialize AWS clients for batch inference
        print("\n" + "="*60)
        print("Initializing batch inference...")
        print("="*60)
        
        # Configure batch inference parameters
        bucket = args.s3_bucket
        s3_prefix = getattr(args, 's3_prefix', None) or f"batch-inference/{args.data_name}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_suffix = f"{args.data_name}_{args.persona_source}_{args.output_type}_{args.persona_prompt_type}"
        
        # Run batch inference
        print(f"\nRunning batch inference on {len(system_prompts_batch1)} prompts...")
        print(f"Model: {args.sim_model}")
        print(f"S3 Bucket: {bucket}")
        print(f"S3 Prefix: {s3_prefix}")
        
        results_map = run_batch_inference_chunked(
            s3_client=s3_client,
            bedrock_client=bedrock_client,
            system_prompts=system_prompts_batch1,
            user_prompts=user_prompts_batch1,
            record_ids=record_ids,
            model_id=args.sim_model,
            bucket=bucket,
            s3_prefix=s3_prefix,
            max_tokens=args.max_token,
            job_suffix=job_suffix,
            timestamp=timestamp,
            check_interval=getattr(args, 'batch_check_interval', 60)
        )

        # Process and merge batch inference results
        print("\n" + "="*60)
        print("Processing and merging results...")
        print("="*60)
        
        result_idx = 0
        missing_results = []
        
        for processed_idx, original_idx in enumerate(tqdm(instance_to_process, desc="Merging results")):
            d = data[start_idx + original_idx]
            
            # Initialize storage for persona outputs if not exists
            if 'persona_outputs_{}'.format(args.output_type) not in d:
                d['persona_outputs_{}'.format(args.output_type)] = []
            
            num_personas_for_instance = persona_counts_per_instance[processed_idx]
            
            # Collect outputs for this data instance
            for persona_idx in range(num_personas_for_instance):
                record_id = record_ids[result_idx]
                
                # Get result from batch inference results
                if record_id in results_map:
                    raw_output = results_map[record_id]
                    
                    output_data = {
                        'persona_idx': persona_idx,
                        'system_prompt': system_prompts_batch1[result_idx],
                        'user_message': user_prompts_batch1[result_idx],
                        'pred_label': raw_output,
                        '{}_output'.format(args.sim_model): raw_output,
                        'input_tokens': 0,  
                        'output_tokens': 0
                    }
                    d['persona_outputs_{}'.format(args.output_type)].append(output_data)
                else:
                    missing_results.append(record_id)
                    print(f"⚠️  Warning: Missing result for record_id: {record_id}")
                
                result_idx += 1
        
        # Mark skipped instances explicitly
        for skip_info in skipped_instances:
            d = data[start_idx + skip_info['index']]
            d['persona_outputs_{}'.format(args.output_type)] = []
            d['skipped_reason'] = skip_info['reason']
        
        # Verify all results were processed
        expected_results = len(system_prompts_batch1)
        received_results = len(results_map)
        
        print(f"\n📊 Inference Statistics:")
        print(f"   Total data instances: {len(data[start_idx:end_idx])}")
        print(f"   Instances processed: {len(instance_to_process)}")
        print(f"   Instances skipped: {len(skipped_instances)}")
        print(f"   Total inference calls: {expected_results}")
        print(f"   Results received: {received_results}")
        if missing_results:
            print(f"   ⚠️  Missing results: {len(missing_results)}")
        if use_all_personas:
            print(f"   Personas per instance: Variable (see above)")
        else:
            print(f"   Personas per instance: Up to {fixed_num_persona}")
        
        print(f"   Note: Token counts not available with batch inference")

        # Save results
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)
            
        output_file = output_dir / f'simbench_{args.data_name}_{args.num_persona}_{args.format}_{args.persona_source}_personas_{args.persona_prompt_type}_prompt_{args.output_type}_conf.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Results saved to: {output_file}")

    else:
        # Eval only mode
        print("\n" + "="*60)
        print("EVAL ONLY MODE - Validating existing outputs...")
        print("="*60)

     # ============================================
    # EVALUATION (with missing demographics skipped)
    # ============================================
    eval_configs = [
        (1.0, '100%')
    ]
    
    # Initialize results storage with per-percentage counts
    results = {name: {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0} 
               for _, name in eval_configs}
    
    # Initialize top-5 and worst-5 results
    top5_results = {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0}
    worst5_results = {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0}
    
    # Initialize demographic results with per-percentage counts
    demographic_results = {}
    demographic_counts = {}
    demographic_group_sizes = {}  # Store group_size for each demographic
    demographic_human_answers = {}  # Store human answers for entropy calculation
    demographic_key_group_sizes = {}  # Store group sizes per key-value
    
    # Track skipped evaluations
    skipped_eval_count = 0
    skipped_eval_demographics = {}
    skipped_missing_persona_count = 0
    skipped_missing_persona_demographics = {}
    num_used_personas = 0
    
    # Run evaluation
    top5_call_count = 0
    persona_count_distribution = {}
    
    for idx, d in enumerate(tqdm(data[start_idx:], desc="Evaluating")):
        # ============================================
        # SKIP if no matching persona exists
        # ============================================
        #persona_key = tuple(sorted(d['group_prompt_variable_map'].items()))
        persona_key = make_persona_key(d.get('group_prompt_variable_map', {}))

        if persona_key not in persona_indexing:
            demo_key = get_demographic_key(d['group_prompt_variable_map'])
            skipped_missing_persona_count += 1
            skipped_missing_persona_demographics[demo_key] = skipped_missing_persona_demographics.get(demo_key, 0) + 1
            continue
        
        # Handle num_persona setting
        if args.num_persona == 'all':
            num_personas_this_instance = len(d.get('persona_outputs_{}'.format(args.output_type), []))
        else:
            if len(d.get('persona_outputs_{}'.format(args.output_type), [])) > int(args.num_persona):
                d['persona_outputs_{}'.format(args.output_type)] = random.sample(
                    d['persona_outputs_{}'.format(args.output_type)], int(args.num_persona)
                )
                num_personas_this_instance = int(args.num_persona)
            else:
                num_personas_this_instance = len(d.get('persona_outputs_{}'.format(args.output_type), []))
        
        # Track persona count distribution
        persona_count_distribution[num_personas_this_instance] = persona_count_distribution.get(num_personas_this_instance, 0) + 1
        
        num_used_personas += num_personas_this_instance
        
        # Skip instances with 0 personas
        if num_personas_this_instance == 0:
            demo_key = get_demographic_key(d['group_prompt_variable_map'])
            skipped_eval_count += 1
            skipped_eval_demographics[demo_key] = skipped_eval_demographics.get(demo_key, 0) + 1
            continue
        
        expected_keys = list(d['human_answer'].keys())
        gt_distribution = [d['human_answer'][key] / 100 for key in expected_keys]
        
        # Get demographic key for this instance
        demo_key = get_demographic_key(d['group_prompt_variable_map'])
        
        # Initialize demographic tracking if needed
        if demo_key not in demographic_results:
            demographic_results[demo_key] = {
                name: {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0}
                for _, name in eval_configs
            }
            demographic_counts[demo_key] = {}
            for _, name in eval_configs:
                demographic_counts[demo_key][name] = 0
            # Store group_size and human_answers
            demographic_group_sizes[demo_key] = d.get('group_size', 'N/A')
            demographic_human_answers[demo_key] = []
        
        # Store human answer for entropy calculation
        demographic_human_answers[demo_key].append(d['human_answer'])
        
        # Collect group sizes per key-value
        demo_map = d.get('group_prompt_variable_map', {})
        for key_name, key_value in demo_map.items():
            if key_name not in demographic_key_group_sizes:
                demographic_key_group_sizes[key_name] = {}
            if key_value not in demographic_key_group_sizes[key_name]:
                demographic_key_group_sizes[key_name][key_value] = d.get('group_size', 0)
        
        # Evaluate regular percentages
        for fraction, name in eval_configs:
            result = evaluate_persona_subset(
                d, 
                args.output_type, 
                args.sim_model, 
                expected_keys,
                gt_distribution,
                num_personas_this_instance,
                fraction,
                args.eval_from
            )
            
            if result[0] is not None:  # Valid result
                sim, tvd, kl, wd, none_cnt = result
                
                # Add to overall results with count
                results[name]['sim'] += sim
                results[name]['tvd'] += tvd
                results[name]['kl'] += kl
                results[name]['wd'] += wd
                results[name]['none_cnt'] += none_cnt
                results[name]['count'] += 1
                
                # Add to demographic-specific results with count
                demographic_results[demo_key][name]['sim'] += sim
                demographic_results[demo_key][name]['tvd'] += tvd
                demographic_results[demo_key][name]['kl'] += kl
                demographic_results[demo_key][name]['wd'] += wd
                demographic_results[demo_key][name]['none_cnt'] += none_cnt
                demographic_results[demo_key][name]['count'] += 1
                demographic_counts[demo_key][name] += 1
        
        # Evaluate top-5 best and worst-5 personas
        top5_call_count += 1
        best_metrics, worst_metrics = evaluate_persona_ranking(
            d, args.output_type, args.sim_model, expected_keys, gt_distribution, top_k=5
        )
        
        if best_metrics[0] is not None:
            sim, tvd, kl, wd, none_cnt = best_metrics
            top5_results['sim'] += sim
            top5_results['tvd'] += tvd
            top5_results['kl'] += kl
            top5_results['wd'] += wd
            top5_results['none_cnt'] += none_cnt
            top5_results['count'] += 1
        
        if worst_metrics[0] is not None:
            sim, tvd, kl, wd, none_cnt = worst_metrics
            worst5_results['sim'] += sim
            worst5_results['tvd'] += tvd
            worst5_results['kl'] += kl
            worst5_results['wd'] += wd
            worst5_results['none_cnt'] += none_cnt
            worst5_results['count'] += 1
    
    # Print persona count distribution
    print(f"\n=== PERSONA COUNT DISTRIBUTION ===")
    for count in sorted(persona_count_distribution.keys()):
        print(f"Instances with {count} personas: {persona_count_distribution[count]}")
    
    # Print skipped statistics
    print("\n" + "="*60)
    print("EVALUATION SKIP SUMMARY")
    print("="*60)
    
    if skipped_missing_persona_count > 0:
        print(f"\n⚠️  Skipped {skipped_missing_persona_count} instances (NO MATCHING PERSONA):")
        for demo, count in sorted(skipped_missing_persona_demographics.items()):
            print(f"   - {demo}: {count} instances")
    
    if skipped_eval_count > 0:
        print(f"\n⚠️  Skipped {skipped_eval_count} instances (0 PERSONAS AVAILABLE):")
        for demo, count in sorted(skipped_eval_demographics.items()):
            print(f"   - {demo}: {count} instances")
    
    total_skipped = skipped_missing_persona_count + skipped_eval_count
    total_evaluated = len(data[start_idx:]) - total_skipped
    print(f"\n📊 Evaluation Summary:")
    print(f"   Total data instances: {len(data[start_idx:])}")
    print(f"   Skipped (no matching persona): {skipped_missing_persona_count}")
    print(f"   Skipped (0 personas): {skipped_eval_count}")
    print(f"   Total skipped: {total_skipped}")
    print(f"   Total evaluated: {total_evaluated}")
    
    # Check if we have any data to evaluate
    if all(results[name]['count'] == 0 for _, name in eval_configs):
        print("\n⚠️  WARNING: No instances to evaluate! All instances were skipped.")
        return
    
    # Print overall results
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print(f"AVG Number of Persona used for prediction: {round(num_used_personas/max(total_evaluated, 1), 2)}")
    print("="*60)
    
    csv_header = [['Evaluator: {} result'.format(args.sim_model)]]
    csv_header.append(['Method', 'Num Instances', 'SimScore', 'TVD', 'KL-divergence', 'WD', 'AVG None_cnt'])
    
    # Add regular results
    for fraction, name in eval_configs:
        num_valid = results[name]['count']
        
        if num_valid == 0:
            print(f'\n===== {name} PERSONAS =====')
            print('No valid results')
            continue
            
        print(f'\n===== {name} PERSONAS =====')
        print(f'Valid instances: {num_valid}')
        
        avg_sim = round(results[name]['sim'] / num_valid, 2)
        avg_tvd = round(results[name]['tvd'] / num_valid, 3)
        avg_kl = round(results[name]['kl'] / num_valid, 3)
        avg_wd = round(results[name]['wd'] / num_valid, 3)
        avg_none = round(results[name]['none_cnt'] / num_valid, 3)
        
        print(f'SimScore: {avg_sim}')
        print(f'TVD: {avg_tvd}')
        print(f'KL-divergence: {avg_kl}')
        print(f'Wasserstein Distance: {avg_wd}')
        print(f'AVG unanswered responses: {avg_none}')
        
        # Add to CSV
        method_name = f'{args.num_persona}_{args.persona_prompt_type}_personas_{args.output_type}_{name}'
        csv_header.append([method_name, num_valid, avg_sim, avg_tvd, avg_kl, avg_wd, avg_none])
    
    # Add top-5 best personas results
    if top5_results['count'] > 0:
        print(f'\n===== TOP-5 BEST PERSONAS (Lowest TVD) =====')
        print(f'Valid instances: {top5_results["count"]}')
        
        avg_sim = round(top5_results['sim'] / top5_results['count'], 2)
        avg_tvd = round(top5_results['tvd'] / top5_results['count'], 3)
        avg_kl = round(top5_results['kl'] / top5_results['count'], 3)
        avg_wd = round(top5_results['wd'] / top5_results['count'], 3)
        avg_none = round(top5_results['none_cnt'] / top5_results['count'], 3)
        
        print(f'SimScore: {avg_sim}')
        print(f'TVD: {avg_tvd}')
        print(f'KL-divergence: {avg_kl}')
        print(f'Wasserstein Distance: {avg_wd}')
        print(f'AVG unanswered responses: {avg_none}')
        
        method_name = f'{args.num_persona}_{args.persona_prompt_type}_personas_{args.output_type}_top5_best'
        csv_header.append([method_name, top5_results['count'], avg_sim, avg_tvd, avg_kl, avg_wd, avg_none])
    else:
        print(f'\n⚠️  WARNING: No valid top-5 results to report!')
    
    # Add worst-5 personas results
    if worst5_results['count'] > 0:
        print(f'\n===== WORST-5 PERSONAS (Highest TVD) =====')
        print(f'Valid instances: {worst5_results["count"]}')
        
        avg_sim = round(worst5_results['sim'] / worst5_results['count'], 2)
        avg_tvd = round(worst5_results['tvd'] / worst5_results['count'], 3)
        avg_kl = round(worst5_results['kl'] / worst5_results['count'], 3)
        avg_wd = round(worst5_results['wd'] / worst5_results['count'], 3)
        avg_none = round(worst5_results['none_cnt'] / worst5_results['count'], 3)
        
        print(f'SimScore: {avg_sim}')
        print(f'TVD: {avg_tvd}')
        print(f'KL-divergence: {avg_kl}')
        print(f'Wasserstein Distance: {avg_wd}')
        print(f'AVG unanswered responses: {avg_none}')
        
        method_name = f'{args.num_persona}_{args.persona_prompt_type}_personas_{args.output_type}_worst5'
        csv_header.append([method_name, worst5_results['count'], avg_sim, avg_tvd, avg_kl, avg_wd, avg_none])
    else:
        print(f'\n⚠️  WARNING: No valid worst-5 results to report!')
    
    # Save overall CSV
    csv_dir = Path(args.output_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir / f'simbench_{args.data_name}_{args.num_persona}_{args.format}_{args.persona_source}_personas_{args.persona_prompt_type}_prompt_{args.output_type}_conf.csv'
    
    with open(csv_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_header)
    
    print(f"\n💾 Overall CSV results saved to: {csv_file}")
    
    # # ============================================================================
    # # CALCULATE ENTROPY FOR ALL DEMOGRAPHICS
    # # ============================================================================
    # demographic_entropy = {}
    # for demo_key in demographic_human_answers.keys():
    #     if len(demographic_human_answers[demo_key]) > 0:
    #         answer_keys = list(demographic_human_answers[demo_key][0].keys())
    #         print(answer_keys)
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
    
    # Collect human answers for each demographic key
    for idx, d in enumerate(data[start_idx:]):
        #persona_key = tuple(sorted(d['group_prompt_variable_map'].items()))
        persona_key = make_persona_key(d.get('group_prompt_variable_map', {}))

        if persona_key not in persona_indexing:
            continue
        
        if len(d.get('persona_outputs_{}'.format(args.output_type), [])) == 0:
            continue
        
        demo_map = d.get('group_prompt_variable_map', {})
        for key_name in demo_map.keys():
            if key_name not in demographic_key_human_answers:
                demographic_key_human_answers[key_name] = []
            demographic_key_human_answers[key_name].append(d['human_answer'])
    
    # Aggregate results from existing demographic combinations
    for demo_key in demographic_results.keys():
        demo_dict = {}
        for item in demo_key.split(', '):
            if ':' in item:
                k, v = item.split(':', 1)
                demo_dict[k] = v
        
        for key_name in demo_dict.keys():
            if key_name not in demographic_key_results:
                demographic_key_results[key_name] = {
                    name: {'sim': 0, 'tvd': 0, 'kl': 0, 'wd': 0, 'none_cnt': 0, 'count': 0}
                    for _, name in eval_configs
                }
            
            for fraction, name in eval_configs:
                demographic_key_results[key_name][name]['sim'] += demographic_results[demo_key][name]['sim']
                demographic_key_results[key_name][name]['tvd'] += demographic_results[demo_key][name]['tvd']
                demographic_key_results[key_name][name]['kl'] += demographic_results[demo_key][name]['kl']
                demographic_key_results[key_name][name]['wd'] += demographic_results[demo_key][name]['wd']
                demographic_key_results[key_name][name]['none_cnt'] += demographic_results[demo_key][name]['none_cnt']
                demographic_key_results[key_name][name]['count'] += demographic_results[demo_key][name]['count']
    
    # Calculate Shannon entropy for each demographic key
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
    
    # Calculate total group size per key
    demographic_key_total_group_size = {}
    for key_name in demographic_key_group_sizes.keys():
        total_size = sum(demographic_key_group_sizes[key_name].values())
        demographic_key_total_group_size[key_name] = total_size
    
    sorted_demo_keys = sorted(demographic_results.keys())
    
    # ============================================================================
    # SAVE PER-DEMOGRAPHIC CSV FILE
    # ============================================================================
    demo_csv_header = [['Per-Demographic Results - Evaluator: {} result'.format(args.sim_model)]]
    
    # Per-Demographic Results
    demo_csv_header.append([])
    demo_csv_header.append(['=== PER-DEMOGRAPHIC ==='])
    demo_csv_header.append(['Demographic', 'Group Size', 'Num Instances', 'Num Personas',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'AVG None_cnt'])
    
    for demo_key in sorted_demo_keys:
        # Get persona count for this demographic
        demo_persona_count = None
        for d in data[start_idx:]:
            if get_demographic_key(d['group_prompt_variable_map']) == demo_key:
                demo_persona_count = len(d.get('persona_outputs_{}'.format(args.output_type), []))
                if demo_persona_count > 0:
                    break
        
        for fraction, name in eval_configs:
            demo_count = demographic_counts[demo_key][name]
            
            if demo_count > 0 and name == '100%':
                group_size = demographic_group_sizes.get(demo_key, 'N/A')
                
                avg_sim = round(demographic_results[demo_key][name]['sim'] / demo_count, 2)
                avg_tvd = round(demographic_results[demo_key][name]['tvd'] / demo_count, 3)
                avg_kl = round(demographic_results[demo_key][name]['kl'] / demo_count, 3)
                avg_wd = round(demographic_results[demo_key][name]['wd'] / demo_count, 3)
                avg_none = round(demographic_results[demo_key][name]['none_cnt'] / demo_count, 3)
                
                demo_csv_header.append([
                    demo_key,
                    group_size,
                    demo_count,
                    demo_persona_count if demo_persona_count else 'N/A',
                    avg_sim, 
                    avg_tvd, 
                    avg_kl, 
                    avg_wd, 
                    avg_none
                ])
    
    # Per-Demographic-Key Results
    demo_csv_header.append([])
    demo_csv_header.append(['=== PER-DEMOGRAPHIC-KEY ==='])
    demo_csv_header.append(['Demographic Key', 'Total Group Size', 'Group Size Breakdown', 'Num Instances',  'SimScore', 'TVD', 'KL-divergence', 'WD', 'AVG None_cnt'])
    
    for key_name in sorted(demographic_key_results.keys()):
        for fraction, name in eval_configs:
            count = demographic_key_results[key_name][name]['count']
            
            if count > 0 and name == '100%':
                total_group_size = demographic_key_total_group_size.get(key_name, 'N/A')
                group_size_breakdown = ""
                if key_name in demographic_key_group_sizes:
                    breakdown_items = [f"{v}:{sz}" for v, sz in sorted(demographic_key_group_sizes[key_name].items())]
                    group_size_breakdown = "; ".join(breakdown_items)
                
                avg_sim = round(demographic_key_results[key_name][name]['sim'] / count, 2)
                avg_tvd = round(demographic_key_results[key_name][name]['tvd'] / count, 3)
                avg_kl = round(demographic_key_results[key_name][name]['kl'] / count, 3)
                avg_wd = round(demographic_key_results[key_name][name]['wd'] / count, 3)
                avg_none = round(demographic_key_results[key_name][name]['none_cnt'] / count, 3)
                
                demo_csv_header.append([
                    key_name,
                    total_group_size,
                    group_size_breakdown,
                    count,
                    avg_sim,
                    avg_tvd,
                    avg_kl,
                    avg_wd,
                    avg_none
                ])
    
    # Save per-demographic CSV
    demographic_csv_file = csv_dir / f'simbench_{args.data_name}_{args.num_persona}_{args.format}_{args.persona_source}_personas_{args.persona_prompt_type}_prompt_{args.output_type}_per_demographic_conf.csv'
    
    with open(demographic_csv_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(demo_csv_header)
    
    print(f"\n💾 Per-demographic CSV results saved to: {demographic_csv_file}")

if __name__ == '__main__':
    args = get_args()
    main(args)
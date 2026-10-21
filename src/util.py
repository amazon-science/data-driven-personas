# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Helper function to extract JSON from response
def extract_json_from_response(text):
    """
    Extract JSON content from text that may contain explanatory text or markdown.
    """
    # Try to find JSON between code fences
    json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find JSON object or array without code fences
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    return text.strip()


def parse_persona_response(response_text, expected_count=5):
    """
    Parse persona response and extract personas list.
    Each persona should have the structure:
    {
        "group_prompt_variable_map": {...},
        "persona_id": "P_01",
        "core_constraints": {...},
        "attributes": {...},
        "short_bio": "..."
    }
    """
    try:
        # Extract JSON from response
        json_text = extract_json_from_response(response_text)
        persona_dict = json.loads(json_text)
        
        # Extract personas (handle both list and dict formats)
        if isinstance(persona_dict, list):
            return persona_dict
        elif isinstance(persona_dict, dict):
            # Try common keys
            cycle_personas = persona_dict.get('personas', 
                            persona_dict.get('persona_list', 
                            persona_dict.get('data', 
                            persona_dict.get('results', []))))
            
            # If no personas found in dict, check if dict itself is a single persona
            if not cycle_personas and 'persona_id' in persona_dict:  # Changed from 'name' to 'persona_id'
                return [persona_dict]
            return cycle_personas
        return []
    except Exception as e:
        print(f"⚠️ Error parsing persona response: {e}")
        print(f"Response preview: {response_text[:500]}")  # Add preview for debugging
        return []
    
def parse_option(llm_output, valid_options):
    """
    Parse LLM output to extract the selected answer.
    
    Args:
        llm_output: Raw output from LLM
        valid_options: List of valid answer options (e.g., ['A', 'B', 'C', 'D'])
    
    Returns:
        Parsed answer string or None if no valid answer found
    """
    if not llm_output or llm_output.strip() == "":
        return None
    
    # Clean the output
    output = llm_output.strip()
    
    # Method 1: Look for **Answer**: pattern
    answer_pattern = r'\*\*Answer\*\*:\s*\(?([^)]+)\)?'
    match = re.search(answer_pattern, output, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        # Check if it's a valid option
        for option in valid_options:
            if option.lower() == candidate.lower():
                return option
    
    # Method 2: Look for answer in parentheses
    paren_pattern = r'\(([^)]+)\)'
    matches = re.findall(paren_pattern, output)
    for match in matches:
        candidate = match.strip()
        for option in valid_options:
            if option.lower() == candidate.lower():
                return option
    
    # Method 3: Check if output directly contains one of the options
    for option in valid_options:
        # Exact match (case-insensitive)
        if output.lower() == option.lower():
            return option
        
        # Check if option appears as a standalone word
        pattern = r'\b' + re.escape(option) + r'\b'
        if re.search(pattern, output, re.IGNORECASE):
            return option
    
    # Method 4: Check first word/token
    first_token = output.split()[0].strip('(),.:;!?')
    for option in valid_options:
        if first_token.lower() == option.lower():
            return option
    
    # If no match found, return None or the original output
    return None

def parse_and_normalize_json_response(response, expected_keys):
    try:
        match = re.search(r'\{[\s\S]*?\}', response)
        if match:
            json_str = match.group(0)
            try:
                response_parsed = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = re.sub(r'%', '', json_str)
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                json_str = re.sub(r'([{,]\s*)([^"\s]+)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r':\s*([^",}\]\s]+)([,}\]])', r':"\1"\2', json_str)
                json_str = re.sub(r':"(-?\d+\.?\d*)"', r':\1', json_str)
                json_str = re.sub(r'([^"])"([,}])', r'\1\2', json_str)
                json_str = json_str.strip()
                if not json_str.startswith('{'): json_str = '{' + json_str
                if not json_str.endswith('}'): json_str = json_str + '}'
                response_parsed = json.loads(json_str)
            if not all(key in response_parsed for key in expected_keys):
                return None
            total_prob = sum(response_parsed.values())
            if total_prob > 0:
                normalized_probs = [v / total_prob for v in response_parsed.values()]
                return normalized_probs
            else:
                None
        else:
            return None
    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError, ZeroDivisionError) as e:
        return None

def simscore(gt_distribution: list, pred_distribution: list):
    assert(len(gt_distribution)== len(pred_distribution))
    uniform_dis=float(1/len(gt_distribution))

    num=0
    den=0
    for idx, (gt, pred) in enumerate(zip(gt_distribution, pred_distribution)):
        num+=abs(gt-pred)
        den+=abs(gt-uniform_dis)

    tvd = num / 2  # TVD is half the sum of absolute differences
    if den==0.0:
        den=0.01
    return tvd, 100*(1-num/den)

def KL_divergence(gt_distribution: list, pred_distribution: list) -> float:
    P = np.array(gt_distribution) + 1e-10
    Q = np.array(pred_distribution) + 1e-10
    return np.sum(P * np.log(P / Q ))

def Wasserstein_distance(gt_distribution: list, pred_distribution: list) -> float:
    """
    Calculate 1-Wasserstein Distance (Earth Mover's Distance) for ordered discrete distributions.
    
    Measures the minimum "cost" to transform one distribution into another,
    considering the distance between categories.
    
    For ordered categories: W1 = Σ |CDF_P(i) - CDF_Q(i)|
    
    Args:
        gt_distribution: Ground truth / reference distribution
        pred_distribution: Predicted / observed distribution
    
    Returns:
        1-Wasserstein distance (normalized by number of positions)
    
    Example:
        Categories: [Very favorable, Somewhat favorable, Somewhat unfavorable, Very unfavorable]
        GT:   [0.4, 0.3, 0.2, 0.1]
        Pred: [0.3, 0.3, 0.2, 0.2]
    """
    P = np.array(gt_distribution, dtype=np.float64)
    Q = np.array(pred_distribution, dtype=np.float64)
    
    # Validation
    if len(P) != len(Q):
        raise ValueError(f"Distributions must have same length: {len(P)} vs {len(Q)}")
    
    # if not np.isclose(P.sum(), 1.0, atol=1e-6):
    #     raise ValueError(f"gt_distribution must sum to 1, got {P.sum()}")
    
    # if not np.isclose(Q.sum(), 1.0, atol=1e-6):
    #     raise ValueError(f"pred_distribution must sum to 1, got {Q.sum()}")
    
    # Calculate cumulative distributions
    cdf_P = np.cumsum(P)
    cdf_Q = np.cumsum(Q)
    
    # 1-Wasserstein distance is the L1 distance between CDFs
    wasserstein = np.sum(np.abs(cdf_P - cdf_Q))
    
    return wasserstein

def get_demographic_key(group_prompt_variable_map):
    """Create a readable string key from demographic features"""
    sorted_items = sorted(group_prompt_variable_map.items())
    return "|".join([f"{k}:{v}" for k, v in sorted_items])


def evaluate_persona_subset(data_instance, output_type, sim_model, expected_keys, 
                            gt_distribution, num_persona, sample_fraction, eval_from='first'):
    """
    Unified evaluation logic for any percentage of personas.
    
    Args:
        data_instance: Single data instance dict
        output_type: 'verbalized' or 'token_prob'
        sim_model: Model name string
        expected_keys: List of valid answer keys
        gt_distribution: Ground truth probability distribution
        num_persona: Total number of personas available
        sample_fraction: Fraction of personas to use (e.g., 0.5 for 50%)
        eval_from: 'first' or 'last' - which personas to sample
    
    Returns:
        tuple: (sim_score, tvd, kl_divergence, wasserstein_distance, none_count)
    """
    num_samples = int(num_persona * sample_fraction)
    none_cnt = 0
    output_key = f'persona_outputs_{output_type}'
    model_output_key = f'{sim_model}_output'
    
    if output_type == 'verbalized':
        pred_distribution = [0.0] * len(gt_distribution)
        
        for i in range(num_samples):
            idx = i if eval_from == 'first' else num_persona - 1 - i
            parsed_distribution = parse_and_normalize_json_response(
                data_instance[output_key][idx][model_output_key], 
                expected_keys
            )
            
            if parsed_distribution is None:
                none_cnt += 1
            else:
                pred_distribution = [pred_distribution[j] + parsed_distribution[j] 
                                   for j in range(len(pred_distribution))]
        
        # Normalize by valid responses
        if num_samples - none_cnt > 0:
            pred_distribution = [p / (num_samples - none_cnt) for p in pred_distribution]
        else:
            return None, None, None, None, none_cnt
    
    elif output_type == 'token_prob':
        option_cnt = {key: 0 for key in expected_keys}
        
        for i in range(num_samples):
            idx = i if eval_from == 'first' else num_persona - 1 - i
            parsed_option = parse_option(
                data_instance[output_key][idx][model_output_key],
                expected_keys
            )
            
            if parsed_option is None:
                none_cnt += 1
            else:
                option_cnt[parsed_option] += 1
        
        # Convert counts to distribution
        if num_samples - none_cnt > 0:
            pred_distribution = [option_cnt[key] / (num_samples - none_cnt) 
                               for key in expected_keys]
        else:
            return None, None, None, None, none_cnt
    
    else:
        raise ValueError(f"Unknown output_type: {output_type}")
    
    # Calculate metrics
    tvd, sim = simscore(gt_distribution, pred_distribution)
    kl = KL_divergence(gt_distribution, pred_distribution)
    wd = Wasserstein_distance(gt_distribution, pred_distribution)
    
    return sim, tvd, kl, wd, none_cnt


def validate_eval_only_mode(data, output_type, num_persona, start_idx):
    """Validate that data has required outputs for eval_only mode."""
    output_key = f'persona_outputs_{output_type}'
    
    for idx, d in enumerate(data[start_idx:], start=start_idx):
        if output_key not in d:
            raise ValueError(
                f"Data instance {idx} missing '{output_key}'. "
                f"Cannot run eval_only without prior inference results."
            )
        if len(d[output_key]) != num_persona:
            raise ValueError(
                f"Data instance {idx} has {len(d[output_key])} outputs, "
                f"expected {num_persona}"
            )
    print(f"✓ Validation passed: All {len(data[start_idx:])} instances have valid outputs")


def diagnostic_analysis(data, args, start_idx=20):
    """
    Analyze per-question performance across different persona percentages.
    Identifies questions where fewer personas unexpectedly have lower TVD.
    """
    print("\n" + "="*60)
    print("RUNNING DIAGNOSTIC ANALYSIS")
    print("="*60)
    
    eval_configs = [
        (1.0, '100%'),
        (0.5, '50%'),
        (0.3, '30%'),
        (0.2, '20%'),
        (0.1, '10%'),
        (0.05, '5%')
    ]
    
    question_analysis = []
    
    for idx, d in enumerate(tqdm(data[start_idx:], desc="Analyzing questions")):
        # Skip instances with no personas
        num_personas_this_instance = len(d.get('persona_outputs_{}'.format(args.output_type), []))
        if num_personas_this_instance == 0:
            continue
        
        # Get basic info
        expected_keys = list(d['human_answer'].keys())
        gt_distribution = [d['human_answer'][key] / 100 for key in expected_keys]
        demographic = get_demographic_key(d['group_prompt_variable_map'])
        
        # Calculate question difficulty (TVD from uniform)
        uniform_dist = [1.0/len(gt_distribution)] * len(gt_distribution)
        tv_uniform = 0.5 * sum(abs(g-u) for g,u in zip(gt_distribution, uniform_dist))
        
        # Calculate denominator used in SimBench
        den = sum(abs(g-u) for g,u in zip(gt_distribution, uniform_dist))
        
        # Store metrics for this question
        question_data = {
            'question_idx': start_idx + idx,
            'demographic': demographic,
            'num_options': len(expected_keys),
            'tv_uniform': tv_uniform,
            'denominator': den,
            'num_personas': num_personas_this_instance,
            'gt_distribution': str(gt_distribution),
        }
        
        # Evaluate at different persona percentages
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
            
            if result[0] is not None:
                sim, tvd, kl, wd, none_cnt = result
                question_data[f'simbench_{name}'] = round(sim, 2)
                question_data[f'tvd_{name}'] = round(tvd, 4)
                question_data[f'kl_{name}'] = round(kl, 4)
                question_data[f'wd_{name}'] = round(wd, 4)
            else:
                question_data[f'simbench_{name}'] = None
                question_data[f'tvd_{name}'] = None
                question_data[f'kl_{name}'] = None
                question_data[f'wd_{name}'] = None
        
        question_analysis.append(question_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(question_analysis)
    
    # Calculate derived metrics
    df['tvd_diff_100_5'] = df['tvd_100%'] - df['tvd_5%']  # Positive = 5% is better (unexpected)
    df['simbench_diff_100_5'] = df['simbench_100%'] - df['simbench_5%']  # Positive = 100% is better (expected)
    
    # Identify anomalous questions (5% has lower TVD than 100%)
    df['anomalous'] = df['tvd_diff_100_5'] > 0
    
    # Save to CSV
    output_dir = Path(args.output_dir)
    diagnostic_file = output_dir / f'diagnostic_{args.data_name}_{args.num_persona}_{args.persona_prompt_type}_personas_{args.output_type}.csv'
    df.to_csv(diagnostic_file, index=False)
    print(f"\n💾 Diagnostic CSV saved to: {diagnostic_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    print(f"\nTotal questions analyzed: {len(df)}")
    print(f"Questions where 5% has LOWER TVD than 100%: {df['anomalous'].sum()} ({100*df['anomalous'].mean():.1f}%)")
    
    if df['anomalous'].sum() > 0:
        print("\n--- Anomalous Questions (5% has lower TVD) ---")
        print(f"Average TV_Uniform (difficulty): {df[df['anomalous']]['tv_uniform'].mean():.4f}")
        print(f"Average denominator: {df[df['anomalous']]['denominator'].mean():.4f}")
        print(f"Average TVD difference (100%-5%): {df[df['anomalous']]['tvd_diff_100_5'].mean():.4f}")
        
        print("\n--- Normal Questions (100% has lower TVD) ---")
        print(f"Average TV_Uniform (difficulty): {df[~df['anomalous']]['tv_uniform'].mean():.4f}")
        print(f"Average denominator: {df[~df['anomalous']]['denominator'].mean():.4f}")
        print(f"Average TVD difference (100%-5%): {df[~df['anomalous']]['tvd_diff_100_5'].mean():.4f}")
    
    # Correlation analysis
    print("\n--- Correlation Analysis ---")
    print("\nCorrelation between question difficulty and TVD improvements:")
    for fraction, name in eval_configs[1:]:  # Skip 100%
        if f'tvd_{name}' in df.columns:
            corr = df['tv_uniform'].corr(df[f'tvd_{name}'])
            print(f"  TV_Uniform vs TVD_{name}: {corr:.3f}")
    
    print("\nCorrelation between question difficulty and SimBench scores:")
    for fraction, name in eval_configs:
        if f'simbench_{name}' in df.columns:
            corr = df['tv_uniform'].corr(df[f'simbench_{name}'])
            print(f"  TV_Uniform vs SimBench_{name}: {corr:.3f}")
    
    # Per-demographic breakdown
    print("\n--- Per-Demographic Breakdown ---")
    demo_stats = df.groupby('demographic').agg({
        'anomalous': ['count', 'sum', 'mean'],
        'tv_uniform': 'mean',
        'tvd_100%': 'mean',
        'tvd_5%': 'mean',
        'simbench_100%': 'mean',
        'simbench_5%': 'mean'
    }).round(4)
    
    print("\nDemographics with highest anomaly rates:")
    demo_stats_sorted = demo_stats.sort_values(('anomalous', 'mean'), ascending=False)
    print(demo_stats_sorted.head(10).to_string())
    
    # Save demographic summary
    demographic_summary_file = output_dir / f'diagnostic_demographic_{args.data_name}_{args.num_persona}_{args.persona_prompt_type}_personas_{args.output_type}.csv'
    demo_stats.to_csv(demographic_summary_file)
    print(f"\n💾 Demographic summary saved to: {demographic_summary_file}")
    
    # Create visualization-ready summary
    summary_data = []
    for _, name in eval_configs:
        summary_data.append({
            'persona_percentage': name,
            'avg_tvd': df[f'tvd_{name}'].mean(),
            'avg_simbench': df[f'simbench_{name}'].mean(),
            'std_tvd': df[f'tvd_{name}'].std(),
            'std_simbench': df[f'simbench_{name}'].std(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f'diagnostic_summary_{args.data_name}_{args.num_persona}_{args.persona_prompt_type}_personas_{args.output_type}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary statistics saved to: {summary_file}")
    
    return df


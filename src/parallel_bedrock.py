# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
import time
import boto3
import json
from botocore.config import Config
from model_input_formatter import format_model_input, extract_response_text, extract_token_usage

# AWS Configuration
config = Config(
    read_timeout=900,
    connect_timeout=900,
    retries={'max_attempts': 10, 'mode': 'standard'}
)

model_id_to_bedrock_id = {
    "claude3.7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude4.0": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "llama3.1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
    "claude4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-pro-standard": "amazon.nova-pro-v1:0"
}

def call_bedrock_model(model_id, system_prompt, user_prompt, param_dict, region_name):
    """
    Call AWS Bedrock model with unified formatting.
    Uses the same format as batch inference.
    """
    # Uses the standard AWS credential chain (AWS_PROFILE, env vars, instance role)
    session = boto3.Session()
    client = session.client(
        service_name='bedrock-runtime',
        region_name=region_name,
        config=config
    )
    
    # Format input using shared formatter (same as batch inference)
    body = format_model_input(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_id=model_id,
        max_tokens=param_dict.get('max_tokens', 4096),
        temperature=param_dict.get('temperature', 0.7),
        top_p=param_dict.get('top_p', 1.0)
    )
    
    # Call model
    response = client.invoke_model(
        modelId=model_id_to_bedrock_id[model_id],
        body=json.dumps(body)
    )
    
    # Parse response
    response_body = json.loads(response['body'].read())
    
    # Extract text using shared extractor (same as batch inference)
    response_text = extract_response_text(response_body, model_id)
    
    # Extract token usage using shared extractor
    input_tokens, output_tokens = extract_token_usage(response_body, model_id)
    
    return response_text, input_tokens, output_tokens

def call_llm_api(model_id, system_prompt, user_prompt, param_dict, region_name="us-east-1"):
    """Call LLM API with retry logic."""
    sleep_duration = 60
    max_time = 600
    counter = 0
    init_time = time.perf_counter()

    while counter < 10 and (time.perf_counter() - init_time) < max_time:
        try:
            return call_bedrock_model(model_id, system_prompt, user_prompt, param_dict, region_name)
        except Exception as e:
            if "ThrottlingException" in str(e) or "TooManyRequestsException" in str(e):
                time.sleep(sleep_duration * 2)
            else:
                time.sleep(sleep_duration)
        counter += 1
    return "", 0, 0

def get_llm_prediction_with_retries(row, llm_params, model_id, region_name):
    """Get prediction from LLM with retries."""
    total_input_tokens = 0
    total_output_tokens = 0
    num_retries = 6
    
    while num_retries > 0:
        num_retries -= 1
        llm_output, input_tokens, output_tokens = call_llm_api(
            model_id=model_id,
            system_prompt=row["system_prompt"],
            user_prompt=row["user_message"],
            param_dict=llm_params,
            region_name=region_name
        )
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        if llm_output:  # Valid response
            return {
                "system_prompt": row.get("system_prompt", ""),
                "user_message": row.get("user_message", ""),
                "pred_label": llm_output, 
                "raw_llm_output": llm_output,
                "input_tokens": total_input_tokens, 
                "output_tokens": total_output_tokens
            }
    
    return {
        "system_prompt": row.get("system_prompt", ""),
        "user_message": row.get("user_message", ""),
        "pred_label": "", 
        "raw_llm_output": "",
        "input_tokens": total_input_tokens, 
        "output_tokens": total_output_tokens
    }

def parallelize_on_rows(data, func, num_proc=8):
    """Apply function to each row in parallel."""
    if num_proc == 1:
        results = []
        for idx in tqdm(range(len(data)), desc="Processing rows"):
            row = data.iloc[idx]
            results.append(func(row))
        return pd.DataFrame(results)
    
    with Pool(num_proc) as pool:
        results = list(tqdm(
            pool.imap(func, [data.iloc[i] for i in range(len(data))]), 
            total=len(data), 
            desc="Processing rows"
        ))
    
    return pd.DataFrame(results)

def process_dataset_parallel(dataset, model_id, llm_params, region_name, num_proc=30):
    """Main function to process dataset with parallel LLM inference."""
    from functools import partial
    
    # Create partial function with fixed parameters
    func = partial(
        get_llm_prediction_with_retries,
        llm_params=llm_params,
        model_id=model_id,
        region_name=region_name
    )
    
    if num_proc == 1:
        # Sequential processing
        results = []
        for idx in tqdm(range(len(dataset)), desc="Processing rows"):
            row = dataset.iloc[idx]
            result = get_llm_prediction_with_retries(row, llm_params, model_id, region_name)
            results.append(result)
        result_df = pd.DataFrame(results)
    else:
        # Parallel processing
        print(f"Processing {len(dataset)} rows with {num_proc} processes...")
        result_df = parallelize_on_rows(dataset, func, num_proc=num_proc)
    
    return result_df

def create_dataset_from_prompts(system_prompts, user_messages):
    """
    Helper function to convert system prompts and user messages to required format.
    
    Args:
        system_prompts: List of system prompts (one for each input)
        user_messages: List of user messages (one for each input)
    
    Returns:
        DataFrame with prompts ready for processing
    """
    if len(system_prompts) != len(user_messages):
        raise ValueError(f"Length mismatch: system_prompts ({len(system_prompts)}) and user_messages ({len(user_messages)}) must have the same length")
    
    data = []
    for system_prompt, user_message in zip(system_prompts, user_messages):
        data.append({
            "system_prompt": system_prompt,
            "user_message": user_message
        })
    return pd.DataFrame(data)

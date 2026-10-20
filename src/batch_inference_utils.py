# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import boto3
import json
import time
import os
from datetime import datetime
from model_input_formatter import format_model_input, extract_response_text

model_id_to_bedrock_id = {
    "claude3.7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "llama3.1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
    "llama3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    'deepseek-r1': "us.deepseek.r1-v1:0",
    'qwen3-32b': "qwen.qwen3-32b-v1:0"
}


# def prepare_batch_input(system_prompts, user_prompts, record_ids=None, max_tokens=1000, model_id="claude3.7"):
#     """Prepare batch inference input in JSONL format
    
#     Args:
#         system_prompts: List of system prompts
#         user_prompts: List of user prompts
#         record_ids: List of record IDs (optional, will generate sequential IDs if not provided)
#         max_tokens: Maximum tokens for generation
#         model_id: Model identifier (e.g., "claude3.7", "llama3.1-70b", "nova-pro")
    
#     Returns:
#         List of batch request dictionaries
#     """
#     if len(system_prompts) != len(user_prompts):
#         raise ValueError(f"Length mismatch: {len(system_prompts)} system prompts vs {len(user_prompts)} user prompts")
    
#     batch_requests = []
    
#     for idx, (system_prompt, user_prompt) in enumerate(zip(system_prompts, user_prompts)):
#         record_id = record_ids[idx] if record_ids else str(idx)
        
#         # Create model-specific request format
#         if model_id.startswith("claude"):
#             model_input = {
#                 "anthropic_version": "bedrock-2023-05-31",
#                 "max_tokens": max_tokens,
#                 "system": system_prompt,
#                 "messages": [{"role": "user", "content": user_prompt}]
#             }
#         elif model_id.startswith("llama"):
#             # Combine system and user prompts for Llama
#             full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
#             model_input = {
#                 "prompt": full_prompt,
#                 "max_gen_len": max_tokens,
#                 "temperature": 0.5,
#                 "top_p": 0.9
#             }
#         elif model_id.startswith("nova"):
#             # Nova models use top-level system parameter (not in messages array)
#             messages = [
#                 {
#                     "role": "user",
#                     "content": [{"text": user_prompt}]
#                 }
#             ]
            
#             model_input = {
#                 "messages": messages,
#                 "inferenceConfig": {
#                     "max_new_tokens": max_tokens,
#                     "temperature": 0.7,
#                     "topP": 0.9
#                 }
#             }
            
#             # Add system prompt at top level if provided
#             if system_prompt:
#                 model_input["system"] = [{"text": system_prompt}]
#         else:
#             raise ValueError(f"Unsupported model: {model_id}")
        
#         request = {
#             "recordId": record_id,
#             "modelInput": model_input
#         }
#         batch_requests.append(request)
    
#     return batch_requests
def prepare_batch_input(system_prompts, user_prompts, record_ids=None, max_tokens=1000, model_id="claude3.7", temperature=0.0, top_p=1.0):
    """Prepare batch inference input in JSONL format - uses shared formatter"""
    if len(system_prompts) != len(user_prompts):
        raise ValueError(f"Length mismatch: {len(system_prompts)} system prompts vs {len(user_prompts)} user prompts")
    
    batch_requests = []
    
    for idx, (system_prompt, user_prompt) in enumerate(zip(system_prompts, user_prompts)):
        record_id = record_ids[idx] if record_ids else str(idx)
        
        # Use shared formatter - ensures consistency with parallel inference
        model_input = format_model_input(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        request = {
            "recordId": record_id,
            "modelInput": model_input
        }
        batch_requests.append(request)
    
    return batch_requests


def upload_to_s3(s3_client, bucket, key, data):
    """Upload JSONL data to S3
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        data: List of dictionaries to upload as JSONL
    
    Returns:
        S3 URI string
    """
    jsonl_content = '\n'.join([json.dumps(item) for item in data])
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=jsonl_content.encode('utf-8')
    )
    return f"s3://{bucket}/{key}"


def create_batch_job(bedrock_client, model_id, input_s3_uri, output_s3_uri, job_name, role_arn=None):
    """Create a batch inference job
    
    Args:
        bedrock_client: Boto3 Bedrock client
        model_id: Model identifier (short name like "claude3.7")
        input_s3_uri: S3 URI for input data
        output_s3_uri: S3 URI for output data
        job_name: Name for the batch job
        role_arn: IAM role ARN (optional, will use environment variable if not provided)
    
    Returns:
        Job ARN string
    """
    if role_arn is None:
        role_arn = os.environ.get('BEDROCK_BATCH_ROLE_ARN')
        if not role_arn:
            raise ValueError("BEDROCK_BATCH_ROLE_ARN must be set in environment or passed as parameter")
    
    # Convert short model ID to full Bedrock model ID
    bedrock_model_id = model_id_to_bedrock_id.get(model_id, model_id)
    
    response = bedrock_client.create_model_invocation_job(
        jobName=job_name,
        roleArn=role_arn,
        modelId=bedrock_model_id,
        inputDataConfig={
            's3InputDataConfig': {
                's3Uri': input_s3_uri
            }
        },
        outputDataConfig={
            's3OutputDataConfig': {
                's3Uri': output_s3_uri
            }
        }
    )
    return response['jobArn']


def wait_for_batch_job(bedrock_client, job_arn, check_interval=60, verbose=True):
    """Wait for batch job to complete
    
    Args:
        bedrock_client: Boto3 Bedrock client
        job_arn: Job ARN to monitor
        check_interval: Seconds between status checks
        verbose: Whether to print status updates
    
    Returns:
        Job response dictionary
    
    Raises:
        Exception: If job fails or is stopped
    """
    if verbose:
        print(f"Waiting for batch job to complete: {job_arn}")
    
    start_time = time.time()
    
    while True:
        response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        status = response['status']
        
        if verbose:
            elapsed = int(time.time() - start_time)
            print(f"Job status: {status} (elapsed: {elapsed}s)")
        
        if status == 'Completed':
            if verbose:
                total_time = int(time.time() - start_time)
                print(f"✅ Job completed in {total_time}s")
            return response
        elif status in ['Failed', 'Stopped']:
            failure_message = response.get('message', 'No error message available')
            raise Exception(f"Batch job failed with status: {status}. Message: {failure_message}")
        
        time.sleep(check_interval)


def download_batch_results(s3_client, output_s3_uri):
    """Download and parse batch inference results
    
    Args:
        s3_client: Boto3 S3 client
        output_s3_uri: S3 URI where results are stored
    
    Returns:
        List of result dictionaries
    """
    # Parse S3 URI
    parts = output_s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    # List all output files
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        raise Exception(f"No output files found in {output_s3_uri}")
    
    results = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.jsonl.out'):
            # Download the file
            file_response = s3_client.get_object(Bucket=bucket, Key=obj['Key'])
            content = file_response['Body'].read().decode('utf-8')
            
            # Parse JSONL
            for line in content.strip().split('\n'):
                if line:
                    results.append(json.loads(line))
    
    if not results:
        raise Exception(f"No results found in output files at {output_s3_uri}")
    
    return results


# def merge_batch_results_by_record_id(batch_results, model_id="claude3.7"):
#     """Convert batch results to a dictionary mapping record_id to response text
    
#     Args:
#         batch_results: List of batch result dictionaries
#         model_id: Model identifier to determine response format
    
#     Returns:
#         Dictionary mapping record_id to response text
#     """
#     results_map = {}
    
#     for result in batch_results:
#         record_id = result['recordId']
        
#         if 'modelOutput' in result:
#             try:
#                 # Parse response based on model type
#                 if model_id.startswith("claude"):
#                     response_text = result['modelOutput']['content'][0]['text']
#                 elif model_id.startswith("llama"):
#                     response_text = result['modelOutput']['generation']
#                 elif model_id.startswith("nova"):
#                     # Nova response format: output -> message -> content -> text
#                     response_text = result['modelOutput']['output']['message']['content'][0]['text']
#                 else:
#                     # Fallback: try multiple formats
#                     if 'content' in result['modelOutput']:
#                         response_text = result['modelOutput']['content'][0]['text']
#                     elif 'generation' in result['modelOutput']:
#                         response_text = result['modelOutput']['generation']
#                     elif 'output' in result['modelOutput']:
#                         response_text = result['modelOutput']['output']['message']['content'][0]['text']
#                     else:
#                         print(f"⚠️  Warning: Unknown response format for record {record_id}")
#                         continue
                
#                 results_map[record_id] = response_text
                
#             except (KeyError, IndexError) as e:
#                 print(f"⚠️  Warning: Error parsing result for record {record_id}: {e}")
#                 print(f"    Result structure: {json.dumps(result, indent=2)}")
#         else:
#             # Log if there's an error in the result
#             if 'error' in result:
#                 print(f"⚠️  Warning: Error in result for record {record_id}: {result.get('error')}")
    
#     return results_map
def merge_batch_results_by_record_id(batch_results, model_id="claude3.7"):
    """Convert batch results using shared extractor"""
    results_map = {}
    
    for result in batch_results:
        record_id = result['recordId']
        
        if 'modelOutput' in result:
            try:
                # Use shared extractor - ensures consistency
                response_text = extract_response_text(result['modelOutput'], model_id)
                results_map[record_id] = response_text
                
            except (KeyError, IndexError) as e:
                print(f"⚠️  Warning: Error parsing result for record {record_id}: {e}")
                print(f"    Result structure: {json.dumps(result, indent=2)}")
        else:
            if 'error' in result:
                print(f"⚠️  Warning: Error in result for record {record_id}: {result.get('error')}")
    
    return results_map


def run_batch_inference(s3_client, bedrock_client, system_prompts, user_prompts, 
                       record_ids, model_id, bucket, s3_prefix, max_tokens,
                       job_suffix='', timestamp=None, check_interval=60):
    """Complete batch inference pipeline
    
    Args:
        s3_client: Boto3 S3 client
        bedrock_client: Boto3 Bedrock client
        system_prompts: List of system prompts
        user_prompts: List of user prompts
        record_ids: List of record IDs
        model_id: Model identifier (e.g., "claude3.7", "llama3.1-70b", "nova-pro")
        bucket: S3 bucket name
        s3_prefix: S3 prefix for files
        max_tokens: Maximum tokens for generation
        job_suffix: Suffix for job naming
        timestamp: Optional timestamp string for file naming
        check_interval: Seconds between status checks
    
    Returns:
        Dictionary mapping record_id to response text
    """
    import hashlib
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not system_prompts:
        print(f"No requests to process for job {job_suffix}")
        return {}
    
    print(f"\n{'='*60}")
    print(f"Starting batch inference: {job_suffix}")
    print(f"Model: {model_id}")
    print(f"Total requests: {len(system_prompts)}")
    print(f"{'='*60}\n")
    
    # Prepare batch requests
    batch_requests = prepare_batch_input(
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        record_ids=record_ids,
        max_tokens=max_tokens,
        model_id=model_id
    )
    
    # Upload input to S3
    input_key = f"{s3_prefix}/input/{job_suffix}_{timestamp}.jsonl"
    input_s3_uri = upload_to_s3(s3_client, bucket, input_key, batch_requests)
    print(f"✅ Uploaded batch input to: {input_s3_uri}")
    
    # Create batch job
    output_s3_uri = f"s3://{bucket}/{s3_prefix}/output/{job_suffix}_{timestamp}/"
    
    # Create unique job name - ensure uniqueness even after truncation
    # Use hash of full identifier to guarantee uniqueness
    full_identifier = f"{job_suffix}_{timestamp}"
    unique_hash = hashlib.md5(full_identifier.encode()).hexdigest()[:12]
    
    # Format: batch-<hash>-<timestamp> to ensure uniqueness
    timestamp_short = timestamp.replace('_', '-')
    job_name = f"batch-{unique_hash}-{timestamp_short}"
    
    # Should already be under 63, but safety check
    if len(job_name) > 63:
        job_name = job_name[:63]
    
    print(f"📋 Job name: {job_name}")

    try:
        job_arn = create_batch_job(
            bedrock_client,
            model_id,
            input_s3_uri,
            output_s3_uri,
            job_name
        )
        print(f"✅ Created batch job: {job_arn}")
    except Exception as e:
        print(f"❌ Failed to create batch job: {str(e)}")
        raise
    
    # Wait for completion
    try:
        wait_for_batch_job(bedrock_client, job_arn, check_interval=check_interval)
    except Exception as e:
        print(f"❌ Batch job failed: {str(e)}")
        raise
    
    # Download and process results
    try:
        batch_results = download_batch_results(s3_client, output_s3_uri)
        results_map = merge_batch_results_by_record_id(batch_results, model_id=model_id)
        print(f"✅ Processed {len(results_map)} responses")
        
        if len(results_map) < len(system_prompts):
            print(f"⚠️  Warning: Expected {len(system_prompts)} results but got {len(results_map)}")
        
        return results_map
    except Exception as e:
        print(f"❌ Failed to download/process results: {str(e)}")
        raise

def run_batch_inference_chunked(
    s3_client,
    bedrock_client,
    system_prompts,
    user_prompts,
    record_ids,
    model_id,
    bucket,
    s3_prefix,
    max_tokens,
    job_suffix,
    timestamp,
    check_interval=60,
    max_records_per_batch=50000
):
    """
    Run batch inference with automatic chunking for large datasets.
    AWS Bedrock has a limit of 50,000 records per batch job.
    """
    total_prompts = len(system_prompts)
    
    if total_prompts <= max_records_per_batch:
        # Single batch - use original function
        return run_batch_inference(
            s3_client=s3_client,
            bedrock_client=bedrock_client,
            system_prompts=system_prompts,
            user_prompts=user_prompts,
            record_ids=record_ids,
            model_id=model_id,
            bucket=bucket,
            s3_prefix=s3_prefix,
            max_tokens=max_tokens,
            job_suffix=job_suffix,
            timestamp=timestamp,
            check_interval=check_interval
        )
    
    # Need to split into multiple batches
    num_batches = (total_prompts + max_records_per_batch - 1) // max_records_per_batch
    print(f"\n⚠️  Total prompts ({total_prompts}) exceeds limit ({max_records_per_batch})")
    print(f"   Splitting into {num_batches} batches...")
    
    all_results = {}
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_records_per_batch
        end_idx = min((batch_idx + 1) * max_records_per_batch, total_prompts)
        
        batch_system_prompts = system_prompts[start_idx:end_idx]
        batch_user_prompts = user_prompts[start_idx:end_idx]
        batch_record_ids = record_ids[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx + 1}/{num_batches}")
        print(f"Records: {start_idx} to {end_idx - 1} ({len(batch_record_ids)} prompts)")
        print(f"{'='*60}")
        
        batch_job_suffix = f"{job_suffix}_batch{batch_idx + 1}of{num_batches}"
        batch_timestamp = f"{timestamp}_{batch_idx + 1}"
        
        batch_results = run_batch_inference(
            s3_client=s3_client,
            bedrock_client=bedrock_client,
            system_prompts=batch_system_prompts,
            user_prompts=batch_user_prompts,
            record_ids=batch_record_ids,
            model_id=model_id,
            bucket=bucket,
            s3_prefix=s3_prefix,
            max_tokens=max_tokens,
            job_suffix=batch_job_suffix,
            timestamp=batch_timestamp,
            check_interval=check_interval
        )
        
        # Merge results
        all_results.update(batch_results)
        print(f"✅ Batch {batch_idx + 1}/{num_batches} complete. Total results so far: {len(all_results)}")
    
    print(f"\n{'='*60}")
    print(f"ALL BATCHES COMPLETE")
    print(f"Total results collected: {len(all_results)}")
    print(f"{'='*60}")
    
    return all_results

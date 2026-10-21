# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model input formatting for batch and parallel inference"""

def format_model_input(system_prompt, user_prompt, model_id, max_tokens, temperature=0.7, top_p=1.0):
    """
    Format model input consistently across batch and parallel inference.
    
    Args:
        system_prompt: System prompt text
        user_prompt: User prompt text
        model_id: Model identifier (e.g., "claude3.7", "llama3.1-70b", "nova-pro")
        max_tokens: Maximum tokens for generation
        temperature: Temperature parameter
        top_p: Top-p parameter
    
    Returns:
        Dictionary with model-specific input format
    """
    if model_id.startswith("claude"):
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "top_p": top_p
        }

    elif model_id.startswith("llama"):
    # Llama 3.x Instruct chat template format
        if system_prompt:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        return {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
    
    elif model_id.startswith("nova"):
        # Nova models use top-level system parameter
        messages = [
            {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
        ]
        
        model_input = {
            "messages": messages,
            "inferenceConfig": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "topP": top_p
            }
        }
        
        # Add system prompt at top level if provided
        if system_prompt:
            model_input["system"] = [{"text": system_prompt}]
        
        return model_input
    
    elif model_id.startswith("qwen"):
        # Qwen3 on Bedrock uses messages format
        messages = [{"role": "user", "content": user_prompt}]
        
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        return {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }


    
    else:
        raise ValueError(f"Unsupported model: {model_id}")


def extract_response_text(response_body, model_id):
    """
    Extract response text from model output.
    
    Args:
        response_body: Response body from model
        model_id: Model identifier
    
    Returns:
        Response text string
    """
    if model_id.startswith("claude"):
        return response_body['content'][0]['text']
    
    elif model_id.startswith("llama"):
        return response_body['generation']
    
    elif model_id.startswith("nova"):
        return response_body['output']['message']['content'][0]['text']
    
    elif model_id.startswith("qwen"):
            # Response structure:
            # {
            #     "choices": [
            #         {
            #             "message": {
            #                 "role": "assistant",
            #                 "content": "..."
            #             },
            #             "finish_reason": "stop",
            #             "index": 0
            #         }
            #     ],
            #     "usage": {
            #         "prompt_tokens": X,
            #         "completion_tokens": Y,
            #         "total_tokens": Z
            #     }
            # }
            choices = response_body.get("choices", [])
            
            if not choices:
                raise ValueError("Qwen response has empty choices array")
            
            message = choices[0].get("message", {})
            content = message.get("content")
            
            if content is None:
                raise ValueError("Qwen response missing 'content' in message")
            
            return content

    else:
        raise ValueError(f"Unsupported model: {model_id}")


def extract_token_usage(response_body, model_id):
    """
    Extract token usage from model response.
    
    Args:
        response_body: Response body from model
        model_id: Model identifier
    
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    if model_id.startswith("claude"):
        usage = response_body.get('usage', {})
        return usage.get('input_tokens', 0), usage.get('output_tokens', 0)
    
    elif model_id.startswith("llama"):
        # Llama includes token counts differently
        return (
            response_body.get('prompt_token_count', 0),
            response_body.get('generation_token_count', 0)
        )
    
    elif model_id.startswith("nova"):
        usage = response_body.get('usage', {})
        return usage.get('inputTokens', 0), usage.get('outputTokens', 0)
    
    else:
        return 0, 0

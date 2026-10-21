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
import math
from datetime import datetime
from batch_inference_utils import run_batch_inference
from parallel_bedrock import *
from collections import defaultdict
import time
import re
from botocore.config import Config 
from prompt_util import *
import ast

# ============================================================
# EXTENDED FEATURE VALUES DICTIONARY
# ============================================================
FEATURE_VALUES = {
    'CREGION': ['Midwest', 'Northeast', 'South', 'West'],
    
    'AGE': ['18-29', '30-49', '50-64', '65+', '', 'Refused'],
    
    'SEX': ['Female', 'Male', 'In some other way', 'Refused'],
    
    'EDUCATION': [
        'Less than high school',
        'High school graduate',
        'Some college, no degree',
        "Associate's degree",
        'College graduate/some postgrad',
        'Postgraduate',
        'Refused'
    ],
    
    'MARITAL': [
        'Married',
        'Living with a partner',
        'Divorced',
        'Separated',
        'Widowed',
        'Never been married',
        '',
        'Refused'
    ],
    
    'RELIG': [
        'Protestant',
        'Roman Catholic',
        'Mormon',
        'Orthodox',
        'Jewish',
        'Muslim',
        'Buddhist',
        'Hindu',
        'Atheist',
        'Agnostic',
        'Nothing in particular',
        'Christian',
        'Unitarian',
        'Other',
        '',
        'Refused'
    ],
    
    'RELIGATTEND': [
        'More than once a week',
        'Once a week',
        'Once or twice a month',
        'A few times a year',
        'Seldom',
        'Never',
        '',
        'Refused'
    ],
    
    'POLPARTY': ['Democrat', 'Republican', 'Independent', 'Other', '', 'Refused'],
    
    'INCOME': [
        'Less than $30,000',
        '$30,000-$50,000',
        '$50,000-$75,000',
        '$75,000-$100,000',
        '$100,000 or more',
        '',
        'Refused'
    ],
    
    'POLIDEOLOGY': [
        'Very liberal',
        'Liberal',
        'Moderate',
        'Conservative',
        'Very conservative',
        '',
        'Refused'
    ],
    
    'RACE': ['White', 'Black', 'Asian', 'Hispanic', 'Mixed Race', 'Other', 'Refused']
}

# ============================================================
# NORMALIZATION UTILITIES FOR CASE-INSENSITIVE MATCHING
# ============================================================
def normalize_value(val):
    """Normalize a value for case-insensitive comparison."""
    if val is None:
        return ''
    return str(val).lower().strip()

def create_normalization_map():
    """
    Create a mapping from normalized (lowercase) values to canonical values.
    This handles the case mismatch between TARGET_FEATURES and actual data.
    """
    norm_map = {}
    for field, values in FEATURE_VALUES.items():
        norm_map[field] = {}
        for val in values:
            norm_map[field][normalize_value(val)] = val
    return norm_map

NORMALIZATION_MAP = create_normalization_map()

def get_canonical_value(field, value):
    """
    Get the canonical (properly cased) value for a field.
    Returns the original value if no mapping found.
    """
    if field in NORMALIZATION_MAP:
        normalized = normalize_value(value)
        if normalized in NORMALIZATION_MAP[field]:
            return NORMALIZATION_MAP[field][normalized]
    return value

# ============================================================
# GENERATE ALL TARGET FEATURES FROM FEATURE_VALUES
# ============================================================
def generate_all_target_features(feature_values_dict, exclude_empty=True, exclude_refused=True):
    """
    Generate a list of all feature-value pairs from the feature values dictionary.
    
    Args:
        feature_values_dict: Dictionary mapping field names to lists of possible values
        exclude_empty: If True, exclude empty string values
        exclude_refused: If True, exclude 'Refused' values
    
    Returns:
        List of dictionaries, each with a single feature-value pair
    """
    target_features = []
    
    for field, values in feature_values_dict.items():
        for value in values:
            # Skip empty values if requested
            if exclude_empty and value == '':
                continue
            # Skip 'Refused' values if requested
            if exclude_refused and value == 'Refused':
                continue
            
            target_features.append({field: value})
    
    return target_features

# Generate all target features (excluding empty and refused by default)
TARGET_FEATURES_ALL = generate_all_target_features(FEATURE_VALUES)

# Original 50 features for backward compatibility (with normalized casing)
TARGET_FEATURES_OPINIONQA = [
    {'POLPARTY': 'Democrat'},
    {'MARITAL': 'Divorced'},
    {'EDUCATION': 'High school graduate'},
    {'RELIG': 'Atheist'},
    {'POLPARTY': 'Republican'},
    {'CREGION': 'West'},
    {'RACE': 'Black'},
    {'SEX': 'Male'},
    {'MARITAL': 'Living with a partner'},
    {'RELIGATTEND': 'Never'},
    {'INCOME': '$75,000-$100,000'},
    {'POLIDEOLOGY': 'Very liberal'},
    {'CREGION': 'South'},
    {'MARITAL': 'Widowed'},
    {'EDUCATION': 'Some college, no degree'},
    {'AGE': '30-49'},
    {'RELIG': 'Agnostic'},
    {'RELIGATTEND': 'More than once a week'},
    {'RELIGATTEND': 'Once or twice a month'},
    {'RELIG': 'Roman Catholic'},
    {'INCOME': '$50,000-$75,000'},
    {'RELIGATTEND': 'Seldom'},
    {'RELIGATTEND': 'Once a week'},
    {'RACE': 'Asian'},
    {'SEX': 'Female'},
    {'INCOME': 'Less than $30,000'},
    {'AGE': '50-64'},
    {'INCOME': '$100,000 or more'},
    {'MARITAL': 'Never been married'},
    {'AGE': '18-29'},
    {'MARITAL': 'Married'},
    {'RELIG': 'Protestant'},
    {'EDUCATION': "Associate's degree"},
    {'POLPARTY': 'Independent'},
    {'POLIDEOLOGY': 'Conservative'},
    {'RELIGATTEND': 'A few times a year'},
    {'EDUCATION': 'College graduate/some postgrad'},
    {'RACE': 'White'},
    {'EDUCATION': 'Less than high school'},
    {'RELIG': 'Nothing in particular'},
    {'INCOME': '$30,000-$50,000'},
    {'EDUCATION': 'Postgraduate'},
    {'RACE': 'Hispanic'},
    {'POLIDEOLOGY': 'Very conservative'},
    {'POLIDEOLOGY': 'Moderate'},
    {'POLPARTY': 'Other'},
    {'CREGION': 'Midwest'},
    {'AGE': '65+'},
    {'CREGION': 'Northeast'},
    {'POLIDEOLOGY': 'Liberal'}
]

# Extended features including new values (CITIZEN, Mixed Race, etc.)
TARGET_FEATURES_EXTENDED = [
    # All from original 50
    *TARGET_FEATURES_OPINIONQA,
    # Additional features
    {'RACE': 'Mixed Race'},
    {'RACE': 'Other'},
    {'SEX': 'In some other way'},
    {'MARITAL': 'Separated'},
    {'RELIG': 'Mormon'},
    {'RELIG': 'Jewish'},
    {'RELIG': 'Hindu'},
    {'RELIG': 'Buddhist'},
    {'RELIG': 'Muslim'},
    {'RELIG': 'Orthodox'},
    {'RELIG': 'Christian'},
    {'RELIG': 'Unitarian'},
    {'RELIG': 'Other'},
]

# ============================================================
# EXPECTED FIELDS - Now includes 
# ============================================================
EXPECTED_FIELDS = [
    'CREGION', 'AGE', 'SEX', 'EDUCATION',  'MARITAL', 
    'RELIG', 'RELIGATTEND', 'POLPARTY', 'POLIDEOLOGY', 
    'INCOME', 'RACE'
]

# Use AWS profile authentication (set in bash script)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to input JSON data file')
    parser.add_argument('--domain', required=True, help='Domain of persona source')
    parser.add_argument('--num_history', required=True, help='Number of sources used for generation')
    parser.add_argument('--sim_model', required=True, help='Bedrock model ID')
    parser.add_argument('--max_token', type=int, required=True, help='Maximum tokens for generation')
    parser.add_argument('--format', default='free', help='Persona format fixed / free')
    parser.add_argument('--proc', default=30, type=int, help='Number of parallel run.')
    parser.add_argument('--feature_set', default='opinionqa', 
                       choices=['opinionqa', 'extended', 'all'],
                       help='Which feature set to use: opinionqa (50), extended, or all')
    args = parser.parse_args()
    return args


def parse_llm_persona_response(raw_response, expected_fields=None):
    """
    Robust parser for LLM persona generation responses.
    Returns structure with 'user_profile' field to match expected interface.
    Now includes CITIZEN field and proper case normalization.
    """
    if expected_fields is None:
        expected_fields = EXPECTED_FIELDS
    
    try:
        cleaned = raw_response.strip()
        json_str = None
        
        # Step 1: Check if entire response is a Python string literal
        is_python_literal = False
        if (cleaned.startswith("'") and cleaned.endswith("'") and len(cleaned) > 2) or \
           (cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2):
            try:
                cleaned = ast.literal_eval(cleaned)
                is_python_literal = True
            except:
                cleaned = cleaned[1:-1]
                cleaned = cleaned.replace("\\'", "'").replace('\\"', '"')
                cleaned = cleaned.replace("\\n", "\n").replace("\\t", "\t")
                is_python_literal = True
        
        # Step 2: Extract JSON from <output> tags if present
        output_match = re.search(r'<output>\s*(\{.*?\})\s*</output>', cleaned, re.DOTALL)
        
        if output_match:
            json_str = output_match.group(1).strip()
        else:
            # Check if the cleaned content itself looks like JSON
            cleaned_stripped = cleaned.strip()
            if cleaned_stripped.startswith('{'):
                # Find matching closing brace
                brace_count = 0
                start_idx = 0
                for i in range(len(cleaned_stripped)):
                    if cleaned_stripped[i] == '{':
                        brace_count += 1
                    elif cleaned_stripped[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = cleaned_stripped[start_idx:i+1]
                            break
            
            # If still no JSON found, search for JSON block
            if not json_str:
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
        
        if not json_str:
            return {
                'analysis': '',
                'short_bio': '',
                'user_profile': {}
            }
        
        # Step 3: Try to parse JSON
        parsed_data = None
        parse_error = None
        
        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            parse_error = str(e)
            
            # Try to fix common issues if it came from a Python literal
            if is_python_literal:
                try:
                    json_str_fixed = json_str.replace("\\'", "'")
                    parsed_data = json.loads(json_str_fixed)
                except:
                    pass
            
            # If still failing, try regex extraction as fallback
            if not parsed_data:
                try:
                    parsed_data = {}
                    
                    # Extract analysis
                    analysis_match = re.search(r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,', json_str, re.DOTALL)
                    if analysis_match:
                        parsed_data['analysis'] = analysis_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                    
                    # Extract short_bio
                    bio_match = re.search(r'"short_bio"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL)
                    if bio_match:
                        parsed_data['short_bio'] = bio_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                    
                    # Extract user_profile section
                    profile_match = re.search(r'"user_profile"\s*:\s*\{(.*?)\}(?=\s*,\s*"short_bio"|\s*\})', json_str, re.DOTALL)
                    
                    if profile_match:
                        profile_content = profile_match.group(1)
                        user_profile = {}
                        
                        # Extract each field - UPDATED TO INCLUDE CONFIDENCE
                        for field in expected_fields:
                            # Pattern that handles value, reasoning, and confidence
                            field_pattern = rf'"{field}"\s*:\s*\{{\s*"value"\s*:\s*"([^"]*)"[^}}]*"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"[^}}]*(?:"confidence"\s*:\s*"([^"]*)")?'
                            field_match = re.search(field_pattern, profile_content, re.DOTALL)
                            
                            if field_match:
                                user_profile[field] = {
                                    'value': field_match.group(1),
                                    'reasoning': field_match.group(2).replace('\\"', '"').replace('\\n', '\n'),
                                    'confidence': field_match.group(3) if field_match.group(3) else 'medium'
                                }
                        
                        parsed_data['user_profile'] = user_profile
                    
                except Exception as regex_error:
                    print(f"Regex extraction also failed: {regex_error}")
                    pass
        
        if not parsed_data:
            return {
                'analysis': '',
                'short_bio': '',
                'user_profile': {}
            }
        
        # Step 4: Return the result in expected format with confidence
        # Apply case normalization to values
        result = {
            'analysis': parsed_data.get('analysis', ''),
            'short_bio': parsed_data.get('short_bio', ''),
            'user_profile': {}
        }
        
        # Ensure each field has confidence (add default if missing)
        # Also normalize values to canonical casing
        user_profile = parsed_data.get('user_profile', {})
        for field, field_data in user_profile.items():
            if isinstance(field_data, dict):
                raw_value = field_data.get('value', 'Not-deducible')
                # Normalize the value to canonical casing
                canonical_value = get_canonical_value(field, raw_value)
                
                result['user_profile'][field] = {
                    'value': canonical_value,
                    'reasoning': field_data.get('reasoning', ''),
                    'confidence': field_data.get('confidence', 'medium')
                }
            else:
                # Handle case where field_data is not a dict
                canonical_value = get_canonical_value(field, str(field_data))
                result['user_profile'][field] = {
                    'value': canonical_value,
                    'reasoning': '',
                    'confidence': 'medium'
                }
        
        return result
        
    except Exception as e:
        print(f"Parser exception for response preview: {raw_response[:200]}...")
        print(f"Error: {type(e).__name__}: {str(e)}")
        return {
            'analysis': '',
            'short_bio': '',
            'user_profile': {}
        }


def group_personas_by_target_features(results, target_features):
    """
    Group personas by target feature-value pairs.
    Uses case-insensitive matching for robustness.
    Returns a list of components, each representing one target feature.
    Personas are ordered by confidence (high > medium > low) of the target feature.
    """
    # Confidence ordering
    confidence_order = {'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
    
    # Initialize list for components (one per target feature)
    grouped_output = []
    
    for target_feature in target_features:
        # Get the field name and value
        field_name = list(target_feature.keys())[0]
        field_value = target_feature[field_name]
        
        # Normalize target value for comparison
        normalized_target = normalize_value(field_value)
        
        # Find all personas matching this feature-value pair
        matching_personas = []
        
        for result in results:
            if 'persona' in result and field_name in result['persona']:
                persona_value = result['persona'][field_name].get('value', 'Not-deducible')
                
                # Use case-insensitive comparison
                if normalize_value(persona_value) == normalized_target:
                    # Get confidence for this specific field
                    field_confidence = result['persona'][field_name].get('confidence', 'medium')
                    
                    # Create persona entry with all fields
                    attributes = {}
                    for attr_field in EXPECTED_FIELDS:
                        attributes[attr_field] = result['persona'].get(attr_field, {}).get('value', 'Not-deducible')
                    
                    persona_entry = {
                        'group_prompt_variable_map': target_feature,
                        'persona_id': f"P_{len(matching_personas) + 1:02d}",
                        'user_idx': result.get('user_idx', -1),
                        'core_constraints': {
                            'as_text': 'User is from the dataset.',
                            'variables': target_feature
                        },
                        'attributes': attributes,
                        'short_bio': result.get('short_bio', ''),
                        'reasoning': result['persona'][field_name].get('reasoning', ''),
                        'confidence': field_confidence
                    }
                    
                    matching_personas.append(persona_entry)
        
        # Sort personas by confidence (high > medium > low)
        matching_personas.sort(
            key=lambda x: confidence_order.get(x.get('confidence', 'medium').lower(), 0),
            reverse=True
        )
        
        # Update persona_id after sorting
        for idx, persona in enumerate(matching_personas, 1):
            persona['persona_id'] = f"P_{idx:02d}"
        
        # Create group entry for this target feature
        group_entry = {
            'group_prompt_variable_map': target_feature,
            'total_personas_generated': len(matching_personas),
            'expected_personas': len(matching_personas),
            'num_cycles': 1,
            'generated_persona': matching_personas
        }
        
        grouped_output.append(group_entry)
    
    return grouped_output


def log_partial_parse_details(user_idx, parsed_response, expected_fields):
    """
    Log details about what's missing in a partial parse.
    """
    user_profile = parsed_response.get('user_profile', {})
    missing_fields = [f for f in expected_fields if f not in user_profile]
    present_fields = [f for f in expected_fields if f in user_profile]
    
    details = {
        'user_idx': user_idx,
        'total_fields_expected': len(expected_fields),
        'fields_found': len(present_fields),
        'fields_missing': len(missing_fields),
        'missing_field_names': missing_fields,
        'has_analysis': bool(parsed_response.get('analysis', '')),
        'has_short_bio': bool(parsed_response.get('short_bio', ''))
    }
    return details


def get_target_features_by_set(feature_set):
    """
    Get the appropriate target features based on the feature set argument.
    """
    if feature_set == 'opinionqa':
        return TARGET_FEATURES_OPINIONQA
    elif feature_set == 'extended':
        return TARGET_FEATURES_EXTENDED
    elif feature_set == 'all':
        return TARGET_FEATURES_ALL
    else:
        return TARGET_FEATURES_OPINIONQA


def print_feature_coverage_report(grouped_personas_list, feature_set_name):
    """
    Print a detailed report of feature coverage.
    """
    print("\n" + "="*60)
    print(f"Feature Coverage Report ({feature_set_name})")
    print("="*60)
    
    # Group by field for analysis
    field_stats = defaultdict(lambda: {'total_personas': 0, 'values': {}})
    
    for group in grouped_personas_list:
        feature_map = group['group_prompt_variable_map']
        field = list(feature_map.keys())[0]
        value = feature_map[field]
        count = group['total_personas_generated']
        
        field_stats[field]['total_personas'] += count
        field_stats[field]['values'][value] = count
    
    # Print per-field summary
    for field in EXPECTED_FIELDS:
        if field in field_stats:
            stats = field_stats[field]
            print(f"\n{field}:")
            print(f"  Total personas: {stats['total_personas']}")
            print(f"  Value distribution:")
            for value, count in sorted(stats['values'].items(), key=lambda x: x[1], reverse=True):
                print(f"    - {value}: {count}")
        else:
            print(f"\n{field}: No target features defined")

def generate_persona_prompt(domain, format):
    if domain=="movie":
        if format=='free':
            system_prompt="""You are an AI assistant specialized in creating detailed personas that will be used to simulate survey responses. Your personas must enable accurate simulation of how specific individuals would answer social surveys across demographics, politics, values, lifestyle, and diverse social issues.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to survey questions. Downstream applications will use your persona to answer surveys on their behalf across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues. Your persona must contain sufficient detail and internal consistency to enable realistic simulation of their survey answering behavior.

TASK CONTEXT:
You will receive movie reviews written by a single person. Your task is to analyze these reviews to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately. The persona you create will be presented with survey questions, and must be able to generate responses that authentically reflect how this specific individual would answer.

THE CHALLENGE:
Movie reviews provide a unique but limited window into someone's identity. While people reveal their values, worldview, education, personality, and life circumstances through how they write about films, you must avoid the trap of assuming their entire identity revolves around cinema. The reviews are DATA POINTS that reveal patterns about their broader life - their language reveals education level, their values reveal political orientation, their references reveal age and cultural background, their analytical approach reveals personality traits.

YOUR APPROACH:
Use a two-step methodology:
1) First, build a holistic biographical understanding by analyzing writing style, tone, values, life stage indicators, and worldview revealed in the reviews
2) Then, systematically extract specific demographic and psychographic characteristics that would predict survey responses

KEY PRINCIPLES:
- Think beyond movies: Use reviews to infer how they would answer surveys on politics, science, healthcare, education, social issues, etc.
- Look for patterns: Language sophistication → education level; social consciousness → political values; generational references → age group; lifestyle mentions → income/region
- Build coherence: All characteristics should form a realistic, consistent person whose survey responses would align
- Distinguish evidence levels: Clearly separate strong evidence from educated guesses based on demographic patterns
- Predict survey behavior: For each characteristic, consider "How would this person answer survey questions about this topic?"

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields (region, age, gender, education, marital status, religion, religious attendance, political party, political ideology, income, race)
- Select values ONLY from provided options for each field
- Provide detailed reasoning FIRST before each value determination
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics beyond the required 11
- Create a realistic biographical narrative in second-person perspective

SUCCESS CRITERIA:
Your persona should be detailed and coherent enough to accurately predict this person's responses across diverse social survey topics - not just about entertainment, but about climate change, healthcare policy, social issues, economic views, scientific topics, and cultural values."""

            prompt="""The user's data is provided within <user_data></user_data> XML tags.
The user's data includes movie reviews inside <movie_reviews> XML tag:

<user_data>
<movie_reviews>
{movie_reviews}
</movie_reviews>
</user_data>

Please create a detailed user persona profile based on the movie reviews provided.

IMPORTANT: This persona will be used to predict how this person would respond to social surveys on demographics, politics, values, and lifestyle. Your goal is to infer their real-world characteristics as a whole person, NOT just their movie preferences. Use the reviews as a window into their broader identity, but avoid overfitting to movie-specific behaviors.

Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the movie reviews holistically to understand this person as a complete individual. Consider:
- Overall writing style, tone, and sophistication
- General life stage and lifestyle indicators
- Broad values, worldview, and personality
- Life circumstances and context clues
- Cultural and social positioning

Think beyond just their movie tastes - what kind of person writes these reviews? What does their language, references, and perspective reveal about their broader life?

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates inferable characteristics through specific details
- Feels like a realistic description of a complete person (not just a moviegoer)
- Uses narrative style with concrete details about their life beyond movies
- Implicitly reveals demographics through lifestyle, behaviors, and circumstances
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the movie reviews AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the reviews and the bio
2. Then select your answer ONLY from the provided options listed in brackets
3. Assess confidence level (high/medium/low)
4. Consider: How would this person answer survey questions about this field?
5. If uncertain, make your best educated guess based on available information

ADDITIONALLY, you may identify any other inferable characteristics beyond these 11 required fields, such as:
- Personality traits and temperament
- Professional/occupational background
- Lifestyle patterns and hobbies
- Social values and attitudes
- Cultural orientation
- Any other relevant characteristics

For each additional characteristic you identify:
- Create a clear, descriptive feature name
- Provide detailed reasoning FIRST based on evidence from BOTH the reviews and the bio
- Then determine the specific value or description
- Assess confidence level (high/medium/low)

=== ANALYSIS GUIDELINES ===

For both steps, think through:

1. Review Analysis:
- Language patterns, vocabulary, and sophistication
- Cultural and generational references
- Values expressed in opinions about social issues in films
- Life stage indicators (family, career, lifestyle references)
- Geographic or regional linguistic markers
- Political and social commentary
- Religious or spiritual perspectives
- Socioeconomic indicators (viewing habits, travel, lifestyle)

2. Coherence Check:
- Do multiple pieces of evidence support each inference?
- Are the characteristics mutually consistent?
- Does the profile form a realistic, holistic person?
- Would this person's survey responses align with this profile?

3. Avoiding Movie Overfit:
- Don't assume someone's entire identity revolves around cinema
- Use movie reviews as DATA POINTS about their broader worldview
- Consider what their movie opinions reveal about their real-life attitudes
- Think about how they would behave and respond in non-movie contexts

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:

{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across reviews? What do the language, tone, and content reveal about this person's broader life, values, and identity? What life stage and lifestyle indicators are present?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, and context as a complete individual (not just as a movie reviewer)",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the reviews and bio together inform each characteristic? How would this person answer survey questions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===

<output>
{{
"step1_analysis": "Reviewing the movie reviews, I notice a sophisticated, analytical writing style with academic vocabulary and complex sentence structures, suggesting higher education. The user references films from the 1980s-90s with personal nostalgia ('reminds me of my teenage years,' 'watched this in college'), indicating they're likely in their 40s. The tone is earnest and socially conscious, with consistent emphasis on representation, environmental themes, and social justice. References to 'my partner' and 'weekend trips' suggest a stable relationship without young children. The writing reveals someone thoughtful, progressive, and culturally engaged who likely works in education or a knowledge-based field. Their lifestyle includes regular cultural activities and they have the disposable income and time for film festivals. This appears to be an educated, urban professional with progressive values in a comfortable life stage.",

"short_bio": "You are a 44-year-old high school English teacher living in Portland, Oregon, with your partner of eight years. You hold a master's degree in education and spend your free time attending independent film screenings at local theaters, volunteering with environmental nonprofits, and hiking in the Columbia River Gorge. Your colleagues know you as the go-to person for thoughtful discussions about social issues, and you're actively involved in your teachers' union. You and your partner enjoy a comfortable lifestyle, often hosting dinner parties where conversations drift from current events to film analysis, and you're planning a trip to an international documentary festival next summer.",

"step2_analysis": "Based on the reviews and bio, I can now extract specific characteristics. The nostalgic references to 80s/90s films combined with 'my college years' context places this person at approximately 42-48 years old, fitting the 30-49 age bracket. The sophisticated analytical style, teaching references, and master's degree indicate 'postgraduate' education. The 'my partner' language with no child references suggests 'living with a partner.' Progressive film preferences (social justice, environmentalism, LGBTQ+ representation) strongly indicate Democrat party affiliation and liberal ideology. Portland location places them in the West region. Regular film festival attendance and comfortable dual-professional lifestyle suggest '$75,000-$100,000' or possibly '$100,000 or more' household income. The secular humanist values and lack of religious references suggest 'agnostic' or 'nothing in particular' with 'never' attendance. Cultural references and perspective align with white American experiences. Gender appears female based on identification with female characters and feminist analytical lens.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio places person in Portland, Oregon. Review references to outdoor activities, progressive cultural environment, and urban Pacific Northwest lifestyle markers align with West region demographics. References to hiking in Columbia River Gorge confirm Pacific Northwest location.",
        "value": "West",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Nostalgic references to 80s/90s films during 'teenage years' and 'college years' indicate birth in late 1970s/early 1980s. Bio specifies 44 years old. Life stage indicators (established career, no young children, stable long-term relationship, homeownership/stability) confirm middle adulthood in 30-49 bracket.",
        "value": "30-49",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Reviews show feminist analytical lens, particular attention to female character development and representation, identification with female protagonists. Writing patterns and social consciousness align with educated female progressive perspectives. Bio describes lifestyle typical of female educators in urban progressive communities.",
        "value": "female",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Bio explicitly indicates master's degree in education. Reviews demonstrate graduate-level analytical skills, academic vocabulary, sophisticated critical thinking, and theoretical frameworks. Teaching profession typically requires advanced degrees. Clear postgraduate education level for survey response.",
        "value": "postgraduate",
        "confidence": "high"
    }},
    "MARITAL": {{
        "reasoning": "Bio mentions 'partner of eight years' rather than 'spouse' or 'husband,' suggesting unmarried cohabitation. Reviews reference 'my partner' without marriage indicators. No mention of wedding, marriage, or spouse terminology. For survey purposes, 'living with a partner' is most accurate response.",
        "value": "living with a partner",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Reviews show secular humanist values with no religious references or faith-based perspectives. Appreciates spiritual/philosophical themes in films but analyzes them academically rather than personally. Progressive social values and scientific worldview suggest agnostic rather than atheist (which would be more absolute rejection). Bio shows values-driven life without religious framework.",
        "value": "agnostic",
        "confidence": "medium"
    }},
    "RELIGATTEND": {{
        "reasoning": "No references to religious practices, services, faith communities, or spiritual observances in reviews or bio. Lifestyle centered entirely on secular activities (film, hiking, social activism, education). As someone who appears agnostic with no religious framework or practice, survey response would definitively be 'never.'",
        "value": "never",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Strong progressive indicators throughout: environmental nonprofit volunteering, teachers' union involvement, social justice emphasis in reviews, LGBTQ+ advocacy, praise for progressive films. Portland residence in progressive cultural environment. Consistent criticism of conservative messaging in films. Would definitely identify as Democrat in surveys.",
        "value": "Democrat",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Consistently progressive values but with nuanced, educated approach rather than radical positions. Values social justice, environmentalism, and representation, but analyzes films critically rather than ideologically. Teacher identity and academic orientation suggest pragmatic progressivism. Survey response likely 'liberal' rather than 'very liberal' due to measured, thoughtful approach.",
        "value": "liberal",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "Dual income household with both partners as professionals. Master's degree holder in teaching profession. Regular film festival attendance, international travel planning, comfortable lifestyle, likely homeowner or stable renter in Portland. Combined income of two educated professionals suggests $75,000-$100,000 range, possibly higher but this is most conservative estimate.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Cultural references, unstated normative perspective, and cultural touchstones align with white American middle-class experience. Portland demographics (predominantly white), teaching profession demographics, outdoor recreation patterns (hiking), and indie film culture participation all suggest white identity. No indicators of minority racial identity or multicultural perspective. Based on statistical likelihood and cultural patterns.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_traits": {{
        "reasoning": "Reviews demonstrate intellectual curiosity, analytical depth, empathy for characters, and systematic thinking. Bio describes community involvement, professional dedication, relationship stability, and cultural engagement. Personality profile aligns with educated progressive professionals in helping professions.",
        "value": "High openness to experience, high conscientiousness, high agreeableness - intellectually curious, empathetic, organized, values-driven",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states high school English teacher with master's in education. Reviews show pedagogical thinking, literary analysis skills, and educational framing. References to 'my students' and 'using in my class' confirm teaching profession.",
        "value": "Education (high school English teacher)",
        "confidence": "high"
    }},
    "lifestyle_category": {{
        "reasoning": "Bio and reviews reveal active cultural participation, environmental activism, urban living with access to arts venues, professional dual-income couple without children. Lifestyle centers on intellectual/cultural engagement, social activism, and outdoor recreation rather than traditional family-centered activities.",
        "value": "Urban progressive professional, culturally engaged, environmentally conscious, DINK (Dual Income No Kids)",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided movie reviews and generate the complete user persona profile following this two-step process. Remember to include ALL 11 REQUIRED demographic fields, plus any additional characteristics you can infer."""

        elif format == "newfree":
            system_prompt = """You are an AI assistant specialized in creating simulation-ready personas from movie reviews.
These personas will be used to simulate how the same individual would answer NEW survey questions.

PRIMARY OBJECTIVE:
Create a persona that maximizes predictive fidelity for future survey responses across politics, values, morality, institutions, lifestyle choices, and social issues, using only the signals available in movie-review behavior.

KEY PRINCIPLES:
- Treat movie reviews as behavioral evidence: preferences, aversions, moral judgments, and recurring themes are signals of stable traits.
- Think holistically: infer stable drivers (ideology tendencies, religiosity cues, trust in institutions, moral reasoning style, risk tolerance, openness, authoritarian vs anti-authoritarian lean, etc.).
- Look for patterns: repeated opinions across different films imply stable stances; mixed reactions imply conditional rules.
- Build coherence: your persona must form a realistic, internally consistent person.
- Distinguish evidence levels: separate direct evidence from educated inference.
- Avoid fabrication: do not invent vivid biographical specifics unless directly supported by the reviews/metadata.

MOVIE-REVIEW SIGNALS TO USE (examples):
- Genre/theme preferences (e.g., documentaries, war films, romance, satire) and what the user values (realism, escapism, craft, message).
- Moral judgments: condemnation/approval of characters’ actions; views on justice, punishment, loyalty, authority, fairness, harm, purity.
- Social attitudes: reactions to representation, diversity, sexuality/violence, class issues, immigration, policing, war, religion portrayed in films.
- Institutional trust cues: attitudes toward government, media, science, courts, police, corporations as inferred from recurring praise/critique.
- Disposition/style: cynicism vs optimism, tolerance for ambiguity, preference for order vs freedom, empathy vs toughness.
- Cultural/lifestyle hints ONLY when directly suggested (e.g., references to parenting, religious practice, rural/urban life), and keep them general.

CRITICAL REQUIREMENTS:
1) Output MUST be valid JSON and MUST contain EXACTLY four top-level keys:
   - "step1_analysis", "short_bio", "step2_analysis", "user_profile"
   Do NOT output any other keys.
2) SECOND-PERSON ONLY + no name:
   - The "short_bio" MUST be written in SECOND-PERSON perspective and refer to the person as "you".
   - Do NOT use any specific name. Address the person directly as "you".
3) "short_bio" is FREE-FORM persona_text for simulation:
   - No length constraint. Include all details you judge useful for downstream survey simulation.
   - It does NOT have to mention lifestyle/life stage/interests.
   - Focus on predictive traits, stances, conditional rules, attitude strength, and likely answer tendencies.
4) You MUST extract ALL 11 REQUIRED demographic fields:
   - CREGION, AGE, SEX, EDUCATION, MARITAL, RELIG, RELIGATTEND, POLPARTY, POLIDEOLOGY, INCOME, RACE
   - For each field, "value" MUST be selected ONLY from the provided options.
   - Provide "reasoning" BEFORE the value decision and include a "confidence" (high/medium/low).
   - If evidence is weak, still choose a value from the options, but make uncertainty explicit and set confidence to low.
5) No fabricated specifics:
   - Do NOT invent a job title, city, named organizations, number of children, or life events unless explicitly supported by the reviews/metadata.
   - Do NOT invent movie-watching behavior (e.g., “you go to theaters weekly”) unless directly supported.
   - If you include an inference, keep it general and lower confidence accordingly.
6) You may include additional characteristics in "user_profile" beyond the 11 required fields, if they improve simulation fidelity.
   - For any non-required extra fields, "value" may be free-form, but must remain grounded in evidence and avoid fabricated specifics.

IMPORTANT ABOUT ANALYSIS TEXT:
- "step1_analysis" and "step2_analysis" should be evidence-based and helpful.
- Cite specific review IDs and/or movie titles when possible (e.g., R2, “MovieTitle”).
"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.

<user_data>
<movie_reviews>
{movie_reviews}
</movie_reviews>
</user_data>

Please create a simulation-ready persona profile based on the movie reviews provided.
This persona will be used to predict how you (the individual) would respond to additional social surveys.

Guidance on inputs:
- <movie_reviews> may contain multiple reviews by the same person. Treat them as one individual.
- Each review may include optional metadata (movie title, rating, date, genres). Use metadata only as supporting evidence.

Your task will be completed in TWO STEPS:

=== STEP 1: Induce Persona Text (Reasoning + Persona) ===

A) In "step1_analysis", analyze the movie reviews holistically to understand you as a person whose survey answers can be simulated.
Focus on:
- Stable value system and worldview reflected through repeated judgments (what you praise/criticize and why)
- Political/social attitudes and conditional rules inferred from themes you respond to (authority, fairness, harm, tradition, freedom, inequality, etc.)
- Moral reasoning style (absolutist vs situational), tolerance for ambiguity, deference vs skepticism
- Trust across institutions inferred from recurring critiques/praise (media, government, police, courts, science, corporations, etc.)
- Any life-stage/lifestyle indicators ONLY if supported by evidence (avoid fabrication)
- What you likely do when uncertain (default patterns: “you lean pragmatic,” “you defer to personal freedom,” etc.)
- Cite evidence using review IDs/movie titles when possible.

B) In "short_bio", write the persona_text itself (FREE-FORM, no length constraint):
- SECOND-PERSON perspective ONLY (refer to the person as "you")
- Do NOT use any specific name; address the person directly as "you"
- NOT required to be a biography; can be bullets, paragraphs, or mixed format
- Include all information you judge useful for downstream survey simulation
- Emphasize predictive traits, stance strength, and decision rules
- Avoid invented specifics (no job/city/children counts unless explicitly supported)

=== STEP 2: Extract Required Fields (Reasoning + Structured Profile) ===

In "step2_analysis", explain how the movie reviews AND the persona_text ("short_bio") support each inferred characteristic.
- If a required demographic field is weakly supported, explain the uncertainty clearly.

In "user_profile", you MUST determine the following 11 REQUIRED fields, selecting values ONLY from the options:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each REQUIRED field in "user_profile", you MUST include:
- "reasoning": evidence-based reasoning FIRST (cite review IDs/movie titles where possible)
- "value": pick ONLY from the provided options
- "confidence": high/medium/low

You MAY include additional characteristics beyond these 11 required fields (optional),
but keep them grounded and avoid invented specifics. For extra fields, "value" may be free-form.

=== OUTPUT FORMAT (STRICT) ===
Output MUST be valid JSON with EXACTLY these four keys:

{{
  "step1_analysis": "...",
  "short_bio": "...",
  "step2_analysis": "...",
  "user_profile": {{
    "CHARACTERISTIC_NAME": {{
      "reasoning": "...",
      "value": "...",
      "confidence": "high/medium/low"
    }}
  }}
}}

IMPORTANT:
- Output JSON only. No extra text.
- "short_bio" MUST be second-person ("you") and MUST NOT use any specific name.
"""



        elif format=='datainfo_opinionqa':
            system_prompt="""You are an AI assistant specialized in creating detailed personas that will be used to simulate survey responses. Your personas must enable accurate simulation of how specific individuals would answer social surveys across demographics, politics, values, lifestyle, and diverse social issues.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to survey questions. Downstream applications will use your persona to answer surveys on their behalf across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues. Your persona must contain sufficient detail and internal consistency to enable realistic simulation of their survey answering behavior.

TASK CONTEXT:
You will receive movie reviews written by a single person. Your task is to analyze these reviews to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately. The persona you create will be presented with survey questions, and must be able to generate responses that authentically reflect how this specific individual would answer.

UNDERSTANDING OPINIONQA'S SCOPE:
The sample questions provided (gender identity, racial discrimination, family structures) illustrate that OpinionQA measures not just general attitudes but specific positions on controversial social issues where demographics, education, values, and political orientation are highly predictive. Your persona must be detailed enough to predict responses to these sensitive questions across many domains.

WHY MOVIE REVIEWS PREDICT SOCIAL ATTITUDES:
Movie reviews reveal the exact characteristics that predict OpinionQA responses:
- Values expressed about representation/justice in films → predicts positions on racial discrimination, gender identity
- Language sophistication → education level → strongest predictor of progressive social positions
- Social consciousness and empathy → predicts support for marginalized groups
- Commentary on traditional vs. progressive themes → predicts views on family structures, social norms
- Political/social commentary → directly predicts political party, ideology, and policy preferences
- Religious/spiritual perspective (or lack thereof) → predicts values-based opinions on sensitive issues
- Lifestyle and relationship references → predicts views on non-traditional family structures
- Age-related references → predicts generational attitudes on social change

YOUR APPROACH:
Use a two-step methodology with explicit connection to sample questions:
1) Build a holistic biographical understanding that shows how this person would answer the sample questions on gender identity, racial discrimination, and family structures
2) Extract 11 specific demographic characteristics with explicit reasoning about how each predicts responses to sensitive OpinionQA questions

KEY PRINCIPLES:
- Connect to sample questions: For each demographic characteristic, explain how it predicts responses to the sample questions shown
- Use demographic research: Leverage known correlations (e.g., postgraduate education + secular values + urban location = option B on all three sample questions)
- Think intersectionally: Consider how multiple characteristics combine (e.g., educated women with high openness show strongest progressive positions)
- Avoid movie overfitting: Use film opinions to infer broader worldview that applies to all social/political topics
- Build predictive specificity: Show exactly how this profile would answer sensitive questions across domains

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields aligned with OpinionQA groups (region, age, gender, education, marital status, religion, religious attendance, political party, political ideology, income, race)
- Select values ONLY from provided options for each field
- Provide detailed reasoning FIRST that explicitly references how the characteristic predicts responses to sample questions
- Explain demographic patterns (e.g., "Democrats in OpinionQA show strong support for..." "Postgraduate-educated respondents are most likely to...")
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics that enhance prediction of sensitive topics
- Create a realistic biographical narrative in second-person perspective

SUCCESS CRITERIA:
Your persona should accurately predict this person's responses to sensitive OpinionQA questions across all domains - including specific predictions for sample questions about gender identity, racial discrimination, family structures, plus climate change, healthcare, immigration, technology regulation, and other value-laden topics - based on their demographic profile, revealed values, and worldview inferred from movie reviews."""


            prompt="""The user's data is provided within <user_data></user_data> XML tags.
The user's data includes movie reviews inside <movie_reviews> XML tag:
<user_data>
<movie_reviews>
{movie_reviews}
</movie_reviews>
</user_data>
=== CRITICAL CONTEXT: OpinionQA Survey Prediction ===

IMPORTANT: This persona will be used to predict responses to the OpinionQA dataset, which consists of real public opinion surveys from the Pew Research Center's American Trends Panel (ATP).

OpinionQA Dataset Details:
- 1,498 multiple-choice questions from 15 different surveys
- Topics include: politics, science, technology, personal relationships, social issues, economics, health, education, environment, and more
- Questions reflect real-world attitudes and opinions across diverse subject areas
- Survey responses are broken down by 60 demographic groups (age, political affiliation, religion, education, region, race, income, etc.)

=== EXAMPLE OpinionQA QUESTIONS ===

To give you a concrete sense of what you're predicting, here are sample questions from OpinionQA:

<sample_questions>
Question 1 (Gender Identity & Social Values):
"Which statement comes closer to your views, even if neither is exactly right?
Options:
(A): Whether someone is a man or a woman is determined by the sex they were assigned at birth
(B): Someone can be a man or a woman even if that is different from the sex they were assigned at birth
(C): Refused"

Question 2 (Racial Issues & Social Perception):
"When it comes to racial discrimination, which do you think is the bigger problem for the country today?
Options:
(A): People seeing racial discrimination where it really does not exist
(B): People not seeing racial discrimination where it really does exist
(C): Refused"

Question 3 (Family Structure & Social Norms):
"Which statement comes closer to your view?
Options:
(A): Couples who are married do a better job raising children than couples who are living together but are not married
(B): Couples who are living together but are not married can raise children just as well as married couples
(C): Refused"
</sample_questions>

These examples illustrate that OpinionQA covers sensitive social issues, political topics, and value-based questions where demographic characteristics strongly predict response patterns. Your persona must be detailed enough to predict how this person would answer questions like these across many different domains.

Your Goal:
Create a persona that accurately predicts how this person would answer OpinionQA survey questions across ALL these topics - not just questions about movies or entertainment. The movie reviews are a window into their broader identity, values, worldview, and demographic characteristics.

CRITICAL: Avoid overfitting to movie preferences. You must infer their real-world characteristics as a complete person who holds opinions on politics, science, relationships, social issues, and many other topics. Use the reviews as DATA POINTS about their broader identity, but think about how they would respond to surveys on any topic - including the sensitive social and political questions shown above.

=== YOUR TASK ===

Please create a detailed user persona profile based on the movie reviews provided. This profile will be used to predict their responses across the full range of OpinionQA survey topics.
Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the movie reviews holistically to understand this person as a complete individual who holds opinions beyond just movies. Consider:
- Overall writing style, tone, and sophistication
- General life stage and lifestyle indicators
- Broad values, worldview, and personality
- Life circumstances and context clues
- Cultural and social positioning
- How their expressed values might translate to opinions on politics, science, social issues, etc.

Think beyond just their movie tastes - what kind of person writes these reviews? What does their language, references, and perspective reveal about how they would answer survey questions on diverse topics like:
- Political issues and policy preferences
- Social values: gender identity, racial justice, family structures (see sample questions above)
- Scientific topics and technology attitudes
- Economic views and personal finance
- Health and environmental concerns
- Educational priorities and cultural values

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates inferable characteristics through specific details
- Feels like a realistic description of a complete person with opinions on many topics (not just a moviegoer)
- Uses narrative style with concrete details about their life beyond movies
- Implicitly reveals demographics and values that would predict their survey responses (including on questions like the samples above)
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the movie reviews AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics that are relevant to OpinionQA survey prediction.
You MUST determine the following 11 REQUIRED demographic and psychographic characteristics (these align with OpinionQA's demographic groups):

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the reviews and the bio
2. Consider how this demographic characteristic correlates with opinions on diverse OpinionQA topics (including the sample questions shown above)
3. Then select your answer ONLY from the provided options listed in brackets
4. Assess confidence level (high/medium/low)
5. Think: How would someone with this demographic profile answer survey questions about gender identity, racial discrimination, family structures, and other social/political issues?
6. If uncertain, make your best educated guess based on available information and demographic patterns

ADDITIONALLY, you may identify any other inferable characteristics that would help predict OpinionQA responses, such as:
- Personality traits and temperament (affects opinion formation)
- Professional/occupational background (correlates with policy views)
- Lifestyle patterns and values (predict social attitudes)
- Social values and moral foundations (predict ethical stances on questions like the samples)
- Cultural orientation and identity (affects cultural/social opinions)
- Risk tolerance and openness (predict attitudes toward change/innovation)
- Any other characteristics relevant to predicting survey responses

For each additional characteristic you identify:
- Create a clear, descriptive feature name
- Provide detailed reasoning FIRST based on evidence from BOTH the reviews and the bio
- Explain how this characteristic would influence OpinionQA survey responses (including on sensitive topics like the sample questions)
- Then determine the specific value or description
- Assess confidence level (high/medium/low)

=== ANALYSIS GUIDELINES ===

For both steps, think through:

1. Review Analysis:
- Language patterns, vocabulary, and sophistication → predicts education-related opinions
- Cultural and generational references → predicts age-related and cultural opinions
- Values expressed in opinions about social issues in films → predicts broader social/political opinions (e.g., how they'd answer questions about gender identity, racial discrimination, family structures)
- Life stage indicators (family, career, lifestyle) → predicts policy preferences and priorities
- Geographic or regional linguistic markers → predicts regional opinion patterns
- Political and social commentary → directly predicts political opinions and social values
- Religious or spiritual perspectives → predicts values-based opinions (e.g., traditional vs. progressive family views)
- Socioeconomic indicators → predicts economic and policy opinions

2. Coherence Check:
- Do multiple pieces of evidence support each inference?
- Are the characteristics mutually consistent?
- Does the profile form a realistic, holistic person?
- Would this person's survey responses on diverse topics align with this profile?
- Are demographic characteristics consistent with typical opinion patterns for those groups?
- Would their answers to the sample questions be consistent with their overall profile?

3. Avoiding Movie Overfit:
- Don't assume someone's entire identity revolves around cinema
- Use movie reviews as DATA POINTS revealing their broader worldview
- Consider what their movie opinions reveal about their attitudes toward:
    * Social issues (representation, justice, equality) → predicts answers to questions about racial discrimination, gender identity
    * Political values (authority, freedom, tradition, progress) → predicts political and social policy opinions
    * Scientific/technological views (innovation, risk, nature)
    * Economic perspectives (wealth, class, capitalism)
    * Cultural values (individualism, community, diversity)
    * Family and relationship norms → predicts answers to questions about marriage and parenting
- Think about how they would answer the sample OpinionQA questions shown above based on values revealed in their reviews
- Consider how they would answer questions on topics like climate change, healthcare, immigration, technology regulation, social norms, etc.

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:
{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across reviews? What do the language, tone, and content reveal about this person's broader life, values, and identity? What life stage and lifestyle indicators are present? How would these characteristics predict their opinions on diverse OpinionQA topics (politics, science, social issues, etc.)? How would they likely answer the sample questions shown above?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, values, and context as a complete individual who holds opinions on many topics (not just as a movie reviewer)",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the reviews and bio together inform each characteristic? How would this person answer OpinionQA survey questions on diverse topics, including the sample questions about gender identity, racial discrimination, and family structures? What demographic patterns would predict their opinion distributions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST, including how this characteristic would influence OpinionQA survey responses (reference sample questions where relevant)",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===
<output>
{{
"step1_analysis": "Reviewing the movie reviews, I notice a sophisticated, analytical writing style with academic vocabulary and complex sentence structures, suggesting higher education - this would correlate with more informed opinions on scientific and policy topics in OpinionQA. The user references films from the 1980s-90s with personal nostalgia ('reminds me of my teenage years,' 'watched this in college'), indicating they're likely in their 40s - this age group tends to have established political views and moderate-to-progressive stances on social issues. The tone is earnest and socially conscious, with consistent emphasis on representation, environmental themes, and social justice - these values would predict progressive responses on OpinionQA questions about climate change, LGBTQ+ rights (including the gender identity question - would likely choose option B supporting trans rights), racial justice (would likely choose option B on the racial discrimination question, seeing systemic discrimination as the bigger problem), and inequality. References to 'my partner' and 'weekend trips' suggest a stable relationship without young children - this life stage often correlates with liberal social values and would predict option B on the family structure question (cohabiting couples can raise children just as well). The writing reveals someone thoughtful, progressive, and culturally engaged who likely works in education or a knowledge-based field - educators tend to support public investment in education, science funding, and evidence-based policy. Their lifestyle includes regular cultural activities and they have the disposable income and time for film festivals - this suggests upper-middle class status, which correlates with support for environmental protection and social programs while having mixed views on taxation. This appears to be an educated, urban professional with progressive values in a comfortable life stage who would likely answer OpinionQA questions with liberal-to-moderate progressive positions on most social and political issues.",

"short_bio": "You are a 44-year-old high school English teacher living in Portland, Oregon, with your partner of eight years. You hold a master's degree in education and spend your free time attending independent film screenings at local theaters, volunteering with environmental nonprofits, and hiking in the Columbia River Gorge. Your colleagues know you as the go-to person for thoughtful discussions about social issues, and you're actively involved in your teachers' union. You and your partner enjoy a comfortable lifestyle, often hosting dinner parties where conversations drift from current events to film analysis, and you're planning a trip to an international documentary festival next summer.",

"step2_analysis": "Based on the reviews and bio, I can now extract specific characteristics that would predict OpinionQA responses. The nostalgic references to 80s/90s films combined with 'my college years' context places this person at approximately 42-48 years old (30-49 bracket) - this age group in OpinionQA data shows moderate-to-progressive views with pragmatic approaches to policy. The sophisticated analytical style, teaching references, and master's degree indicate 'postgraduate' education - highly educated respondents in OpinionQA tend to support science-based policy, environmental protection, and social programs, and are much more likely to support trans rights (option B on gender identity question) and recognize systemic racial discrimination (option B on racial discrimination question). The 'my partner' language with no child references suggests 'living with a partner' - non-traditional family structures correlate with more liberal social attitudes and would strongly predict option B on the family structure question (cohabiting couples can raise children just as well). Progressive film preferences (social justice, environmentalism, LGBTQ+ representation) strongly indicate Democrat party affiliation and liberal ideology - this would predict progressive positions on all three sample questions and broader OpinionQA questions about climate, healthcare, immigration, and social issues. Portland location (West region) places them in a progressive urban area - West Coast urban residents in OpinionQA show strong support for environmental regulation, social progressivism, and inclusive policies. Regular film festival attendance and comfortable dual-professional lifestyle suggest $75,000-$100,000 household income - this income bracket shows mixed economic views but consistently progressive social views. The secular humanist values and lack of religious references suggest 'agnostic' with 'never' attendance - secular respondents in OpinionQA strongly support science, separation of church and state, liberal social positions, and are most likely to support trans rights and recognize systemic racism. Cultural references and perspective align with white American experiences - white liberals in OpinionQA show strong support for racial justice and diversity initiatives. Gender appears female based on feminist analytical lens - educated women in OpinionQA show particularly strong support for reproductive rights, gender equality, LGBTQ+ rights, and social safety nets. Overall, this profile would predict consistently progressive-to-liberal responses on social issues (including all three sample questions with option B), environmental policy, and science/technology, with moderate-to-progressive economic views.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio places person in Portland, Oregon. Review references to outdoor activities, progressive cultural environment, and urban Pacific Northwest lifestyle markers align with West region demographics. References to hiking in Columbia River Gorge confirm Pacific Northwest location. West region respondents in OpinionQA show higher support for environmental protection, progressive social policies, and climate action compared to other regions. On the sample questions, West Coast urban residents are significantly more likely to choose option B on gender identity (supporting trans rights), option B on racial discrimination (recognizing systemic racism), and option B on family structures (supporting cohabiting couples).",
        "value": "West",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Nostalgic references to 80s/90s films during 'teenage years' and 'college years' indicate birth in late 1970s/early 1980s. Bio specifies 44 years old. Life stage indicators (established career, no young children, stable long-term relationship, homeownership/stability) confirm middle adulthood in 30-49 bracket. This age group in OpinionQA data typically shows pragmatic progressive views - old enough to have established values but young enough to support social change, particularly on LGBTQ+ rights (would support option B on gender identity), climate action, and technology policy. More likely than older cohorts to recognize ongoing racial discrimination (option B) and support non-traditional family structures (option B).",
        "value": "30-49",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Reviews show feminist analytical lens, particular attention to female character development and representation, identification with female protagonists. Writing patterns and social consciousness align with educated female progressive perspectives. Bio describes lifestyle typical of female educators in urban progressive communities. In OpinionQA data, educated women show particularly strong support for reproductive rights, gender pay equity, parental leave, healthcare access, and environmental protection compared to male counterparts. Women are significantly more likely to support trans rights (option B on gender identity), recognize systemic racism (option B on racial discrimination), and support flexible family structures (option B on family structure question).",
        "value": "female",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Bio explicitly indicates master's degree in education. Reviews demonstrate graduate-level analytical skills, academic vocabulary, sophisticated critical thinking, and theoretical frameworks. Teaching profession typically requires advanced degrees. Clear postgraduate education level for survey response. Postgraduate-educated respondents in OpinionQA show highest support for science funding, evidence-based policy, climate action, vaccination, and tend to have more nuanced views on complex policy issues. This education level is the strongest predictor of choosing option B on the gender identity question (supporting trans rights), recognizing systemic racial discrimination (option B), and supporting diverse family structures (option B), as higher education correlates with exposure to social science research and diverse perspectives.",
        "value": "postgraduate",
        "confidence": "high"
    }},
    "MARITAL": {{
        "reasoning": "Bio mentions 'partner of eight years' rather than 'spouse' or 'husband,' suggesting unmarried cohabitation. Reviews reference 'my partner' without marriage indicators. No mention of wedding, marriage, or spouse terminology. For survey purposes, 'living with a partner' is most accurate response. Cohabiting couples in OpinionQA data tend to hold more liberal views on social issues, particularly regarding marriage equality, reproductive rights, and non-traditional family structures. This status would strongly predict choosing option B on the family structure question (cohabiting couples can raise children just as well as married couples), and correlates with progressive views on gender identity (option B) and racial justice (option B).",
        "value": "living with a partner",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Reviews show secular humanist values with no religious references or faith-based perspectives. Appreciates spiritual/philosophical themes in films but analyzes them academically rather than personally. Progressive social values and scientific worldview suggest agnostic rather than atheist (which would be more absolute rejection). Bio shows values-driven life without religious framework. Agnostic respondents in OpinionQA strongly support separation of church and state, science-based education, reproductive rights, LGBTQ+ equality, and show lower support for religious exemptions in policy. This religious affiliation is highly predictive of supporting trans rights (option B on gender identity question), recognizing systemic racism (option B on racial discrimination), and supporting diverse family structures (option B on family structure question).",
        "value": "agnostic",
        "confidence": "medium"
    }},
    "RELIGATTEND": {{
        "reasoning": "No references to religious practices, services, faith communities, or spiritual observances in reviews or bio. Lifestyle centered entirely on secular activities (film, hiking, social activism, education). As someone who appears agnostic with no religious framework or practice, survey response would definitively be 'never.' Non-attenders in OpinionQA show strongest support for secular policies, science education, reproductive freedom, and are least likely to support religion-based policy positions. This characteristic strongly predicts choosing option B on all three sample questions, as religious non-attendance is one of the strongest predictors of progressive social views on gender identity, racial justice, and family structures.",
        "value": "never",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Strong progressive indicators throughout: environmental nonprofit volunteering, teachers' union involvement, social justice emphasis in reviews, LGBTQ+ advocacy, praise for progressive films. Portland residence in progressive cultural environment. Consistent criticism of conservative messaging in films. Would definitely identify as Democrat in surveys. Democrats in OpinionQA show strong support for climate action, healthcare expansion, gun control, immigration reform, racial justice initiatives, LGBTQ+ rights, and government intervention in inequality. Democratic party affiliation is highly predictive of choosing option B on all three sample questions: supporting trans rights on gender identity, recognizing systemic racial discrimination, and supporting diverse family structures.",
        "value": "Democrat",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Consistently progressive values but with nuanced, educated approach rather than radical positions. Values social justice, environmentalism, and representation, but analyzes films critically rather than ideologically. Teacher identity and academic orientation suggest pragmatic progressivism. Survey response likely 'liberal' rather than 'very liberal' due to measured, thoughtful approach. Self-identified liberals in OpinionQA support progressive policies but may show more moderate positions on economic regulation, defense spending, and incremental vs. revolutionary change compared to 'very liberal' respondents. However, on social issues including the three sample questions, liberals consistently choose progressive options: option B on gender identity (trans rights), option B on racial discrimination (systemic racism), and option B on family structures (cohabiting couples).",
        "value": "liberal",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "Dual income household with both partners as professionals. Master's degree holder in teaching profession. Regular film festival attendance, international travel planning, comfortable lifestyle, likely homeowner or stable renter in Portland. Combined income of two educated professionals suggests $75,000-$100,000 range, possibly higher but this is most conservative estimate. This income bracket in OpinionQA shows mixed economic views - supports social programs and progressive taxation in principle but may have concerns about personal tax burden; supports environmental regulation but can afford its costs; liberal on social issues but economically moderate. On social issues including the sample questions, this income level with high education predicts progressive responses: option B on gender identity, option B on racial discrimination, and option B on family structures.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Cultural references, unstated normative perspective, and cultural touchstones align with white American middle-class experience. Portland demographics (predominantly white), teaching profession demographics, outdoor recreation patterns (hiking), and indie film culture participation all suggest white identity. No indicators of minority racial identity or multicultural perspective. Based on statistical likelihood and cultural patterns. White liberals in OpinionQA show strong support for racial justice and diversity initiatives, but may have different perspectives on implementation and personal impact compared to racial minorities. White progressives typically choose option B on the racial discrimination question (recognizing systemic racism), strongly support trans rights (option B on gender identity), and support diverse family structures (option B on family structure), though their lived experience with discrimination differs from racial minorities.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_traits": {{
        "reasoning": "Reviews demonstrate intellectual curiosity, analytical depth, empathy for characters, and systematic thinking. Bio describes community involvement, professional dedication, relationship stability, and cultural engagement. Personality profile aligns with educated progressive professionals in helping professions. High openness to experience predicts support for immigration, diversity, cultural exchange, and innovation in OpinionQA, and strongly predicts choosing option B on the gender identity question (accepting of diverse gender identities). High conscientiousness predicts support for evidence-based policy and systematic solutions, correlating with recognizing systemic issues like racial discrimination (option B). High agreeableness predicts support for social welfare and cooperative solutions over competitive ones, and acceptance of diverse family structures (option B on family structure question).",
        "value": "High openness to experience, high conscientiousness, high agreeableness - intellectually curious, empathetic, organized, values-driven",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states high school English teacher with master's in education. Reviews show pedagogical thinking, literary analysis skills, and educational framing. References to 'my students' and 'using in my class' confirm teaching profession. Educators in OpinionQA show strong support for public education funding, teacher pay increases, student loan forgiveness, arts/humanities funding, and evidence-based curriculum. They tend to oppose school privatization and standardized testing overemphasis. Teachers are significantly more likely than other professions to support progressive social positions including option B on all three sample questions (trans rights, recognizing systemic racism, diverse family structures), as education emphasizes critical thinking, diversity, and social awareness.",
        "value": "Education (high school English teacher)",
        "confidence": "high"
    }},
    "lifestyle_category": {{
        "reasoning": "Bio and reviews reveal active cultural participation, environmental activism, urban living with access to arts venues, professional dual-income couple without children. Lifestyle centers on intellectual/cultural engagement, social activism, and outdoor recreation rather than traditional family-centered activities. This lifestyle profile in OpinionQA correlates with support for urban development, public transit, arts funding, environmental protection, work-life balance policies, and lower prioritization of child-related policies (though still supportive of education and family support generally). DINKs (Dual Income No Kids) with progressive cultural engagement are highly likely to choose option B on all three sample questions, as their lifestyle reflects rejection of traditional family norms and embrace of social progressivism.",
        "value": "Urban progressive professional, culturally engaged, environmentally conscious, DINK (Dual Income No Kids)",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided movie reviews and generate the complete user persona profile following this two-step process. Remember: this persona will be used to predict OpinionQA survey responses across diverse topics (politics, science, social issues, health, technology, etc.), including sensitive questions like the samples shown above about gender identity, racial discrimination, and family structures. Think beyond movie preferences to understand the complete person and how they would answer survey questions on many different subjects."""


        elif format=='datainfo_mp3dr':
            system_prompt="""You are an AI assistant specialized in creating detailed personas that will be used to simulate survey responses. Your personas must enable accurate simulation of how specific individuals would answer social science experiments across demographics, politics, values, lifestyle, and diverse social issues.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to experimental survey questions. Downstream applications will use your persona to answer surveys on their behalf across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues. Your persona must contain sufficient detail and internal consistency to enable realistic simulation of their survey answering behavior.

TASK CONTEXT:
You will receive movie reviews written by a single person. Your task is to analyze these reviews to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately. The persona you create will be presented with survey questions, and must be able to generate responses that authentically reflect how this specific individual would answer.

UNDERSTANDING TESS'S SCOPE:
The sample questions provided (gender discrimination policies) illustrate that TESS measures attitudes through experimental designs where demographics, education, values, and political orientation are highly predictive. TESS uses nationally representative samples and experimental manipulations to test causal hypotheses across diverse social science topics. Your persona must be detailed enough to predict responses to these experimental questions across many domains.

WHY MOVIE REVIEWS PREDICT SOCIAL ATTITUDES:
Movie reviews reveal the exact characteristics that predict TESS survey responses:
- Values expressed about representation/justice in films → predicts positions on gender equality, diversity policies
- Language sophistication → education level → strongest predictor of progressive social positions
- Social consciousness and empathy → predicts support for marginalized groups and diversity initiatives
- Commentary on traditional vs. progressive themes → predicts views on workplace policies, social norms
- Political/social commentary → directly predicts political party, ideology, and policy preferences
- Religious/spiritual perspective (or lack thereof) → predicts values-based opinions on sensitive issues
- Lifestyle and relationship references → predicts views on workplace diversity and equality
- Age-related references → predicts generational attitudes on social change and workplace policies

YOUR APPROACH:
Use a two-step methodology with explicit connection to sample questions:
1) Build a holistic biographical understanding that shows how this person would answer the sample questions on gender discrimination policies
2) Extract 11 specific demographic characteristics with explicit reasoning about how each predicts responses to experimental TESS questions

KEY PRINCIPLES:
- Connect to sample questions: For each demographic characteristic, explain how it predicts responses to the sample questions shown
- Use demographic research: Leverage known correlations (e.g., postgraduate education + secular values + urban location = strong support for diversity policies)
- Think intersectionally: Consider how multiple characteristics combine (e.g., educated women with high openness show strongest progressive positions)
- Avoid movie overfitting: Use film opinions to infer broader worldview that applies to all social/political topics
- Build predictive specificity: Show exactly how this profile would answer experimental questions across domains

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields aligned with TESS demographic groups (region, age, gender, education, marital status, religion, religious attendance, political party, political ideology, income, race)
- Select values ONLY from provided options for each field
- Provide detailed reasoning FIRST that explicitly references how the characteristic predicts responses to sample questions
- Explain demographic patterns (e.g., "Democrats in TESS studies show strong support for..." "Postgraduate-educated respondents are most likely to...")
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics that enhance prediction of experimental questions
- Create a realistic biographical narrative in second-person perspective

SUCCESS CRITERIA:
Your persona should accurately predict this person's responses to experimental TESS questions across all domains - including specific predictions for sample questions about gender discrimination policies (recruiting women, diversity training, workplace equality policies) - based on their demographic profile, revealed values, and worldview inferred from movie reviews."""


            prompt="""The user's data is provided within <user_data></user_data> XML tags.
The user's data includes movie reviews inside <movie_reviews> XML tag:
<user_data>
<movie_reviews>
{movie_reviews}
</movie_reviews>
</user_data>
=== CRITICAL CONTEXT: TESS Survey Prediction ===

IMPORTANT: This persona will be used to predict responses to TESS (Time-sharing Experiments for the Social Sciences), which consists of experimental surveys with nationally representative samples.

TESS Dataset Details:
- Since 2001, TESS has supported general population experiments for social science researchers
- Uses NORC's AmeriSpeak Panel - a nationally representative, probability-based panel
- Experimental designs with random assignment to test causal hypotheses
- Topics span all social sciences: politics, economics, psychology, sociology, public health, law, etc.
- Questions involve experimental manipulations (varying stimuli, information, framing) to measure causal effects
- Samples are representative of the U.S. general population with diverse demographic groups

TESS Methodology:
- Probability-based sampling (not convenience samples)
- Online survey administration
- Experimental designs: different subjects randomly assigned to different conditions
- Tests specific hypotheses about social, political, and economic dynamics
- Representative samples weighted to population benchmarks

=== EXAMPLE TESS QUESTIONS ===

To give you a concrete sense of what you're predicting, here are sample questions from a TESS study on gender discrimination:

<sample_questions>
Question 1 (Gender Inequality Policies):
"In order to minimize gender discrimination, some companies implement policies to address gender inequality in the workplace. What is your opinion of this practice?
Response scale: 1 (strongly favor) to 7 (strongly oppose)"

Question 2 (Active Recruitment of Women):
"In order to minimize gender discrimination, some companies actively recruit women to apply for job openings. What is your opinion of this practice?
Response scale: 1 (strongly favor) to 7 (strongly oppose)"

Question 3 (Diversity Training):
"In order to minimize gender discrimination, some companies offer voluntary diversity training to their employees. What is your opinion of this practice?
Response scale: 1 (strongly favor) to 7 (strongly oppose)"
</sample_questions>

These examples illustrate that TESS covers policy attitudes, workplace practices, and social interventions where demographic characteristics, political orientation, and values strongly predict response patterns. Your persona must be detailed enough to predict how this person would answer questions like these across many different experimental studies.

Your Goal:
Create a persona that accurately predicts how this person would answer TESS experimental questions across ALL social science topics - not just questions about movies or entertainment. The movie reviews are a window into their broader identity, values, worldview, and demographic characteristics.

CRITICAL: Avoid overfitting to movie preferences. You must infer their real-world characteristics as a complete person who holds opinions on workplace policies, social interventions, economic issues, political reforms, scientific topics, and many other areas. Use the reviews as DATA POINTS about their broader identity, but think about how they would respond to experimental survey questions on any topic - including the gender discrimination policy questions shown above.

=== YOUR TASK ===

Please create a detailed user persona profile based on the movie reviews provided. This profile will be used to predict their responses across the full range of TESS experimental studies.
Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the movie reviews holistically to understand this person as a complete individual who holds opinions beyond just movies. Consider:
- Overall writing style, tone, and sophistication
- General life stage and lifestyle indicators
- Broad values, worldview, and personality
- Life circumstances and context clues
- Cultural and social positioning
- How their expressed values might translate to opinions on workplace policies, social interventions, economic issues, etc.

Think beyond just their movie tastes - what kind of person writes these reviews? What does their language, references, and perspective reveal about how they would answer experimental survey questions on diverse topics like:
- Workplace diversity and gender equality policies (see sample questions above)
- Social interventions and policy reforms
- Economic policies and labor market issues
- Political reforms and governance
- Scientific topics and technology attitudes
- Health policies and public health interventions
- Educational policies and social programs

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates inferable characteristics through specific details
- Feels like a realistic description of a complete person with opinions on many topics (not just a moviegoer)
- Uses narrative style with concrete details about their life beyond movies
- Implicitly reveals demographics and values that would predict their survey responses (including on questions like the samples above)
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the movie reviews AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics that are relevant to TESS survey prediction.
You MUST determine the following 11 REQUIRED demographic and psychographic characteristics (these align with TESS's demographic groups):

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the reviews and the bio
2. Consider how this demographic characteristic correlates with opinions on diverse TESS experimental topics (including the sample questions shown above about gender discrimination policies)
3. Then select your answer ONLY from the provided options listed in brackets
4. Assess confidence level (high/medium/low)
5. Think: How would someone with this demographic profile answer experimental survey questions about workplace diversity policies, active recruitment of women, and diversity training?
6. If uncertain, make your best educated guess based on available information and demographic patterns

ADDITIONALLY, you may identify any other inferable characteristics that would help predict TESS responses, such as:
- Personality traits and temperament (affects opinion formation)
- Professional/occupational background (correlates with policy views)
- Lifestyle patterns and values (predict social attitudes)
- Social values and moral foundations (predict ethical stances on questions like the samples)
- Cultural orientation and identity (affects cultural/social opinions)
- Risk tolerance and openness (predict attitudes toward change/innovation)
- Any other characteristics relevant to predicting experimental survey responses

For each additional characteristic you identify:
- Create a clear, descriptive feature name
- Provide detailed reasoning FIRST based on evidence from BOTH the reviews and the bio
- Explain how this characteristic would influence TESS experimental survey responses (including on workplace diversity topics like the sample questions)
- Then determine the specific value or description
- Assess confidence level (high/medium/low)

=== ANALYSIS GUIDELINES ===

For both steps, think through:

1. Review Analysis:
- Language patterns, vocabulary, and sophistication → predicts education-related opinions
- Cultural and generational references → predicts age-related and cultural opinions
- Values expressed in opinions about social issues in films → predicts broader social/political opinions (e.g., how they'd answer questions about gender discrimination policies, diversity initiatives)
- Life stage indicators (family, career, lifestyle) → predicts policy preferences and priorities
- Geographic or regional linguistic markers → predicts regional opinion patterns
- Political and social commentary → directly predicts political opinions and social values
- Religious or spiritual perspectives → predicts values-based opinions
- Socioeconomic indicators → predicts economic and policy opinions

2. Coherence Check:
- Do multiple pieces of evidence support each inference?
- Are the characteristics mutually consistent?
- Does the profile form a realistic, holistic person?
- Would this person's experimental survey responses on diverse topics align with this profile?
- Are demographic characteristics consistent with typical opinion patterns for those groups?
- Would their responses to the sample questions (on 1-7 scale) be consistent with their overall profile?

3. Avoiding Movie Overfit:
- Don't assume someone's entire identity revolves around cinema
- Use movie reviews as DATA POINTS revealing their broader worldview
- Consider what their movie opinions reveal about their attitudes toward:
    * Workplace equality and diversity (representation, fairness) → predicts responses to questions about gender discrimination policies, recruiting women, diversity training
    * Social justice and intervention (systemic change, affirmative action) → predicts support for corporate policies addressing inequality
    * Gender roles and equality → directly predicts all three sample questions
    * Progressive vs. traditional values → predicts support for diversity initiatives
    * Institutional change and policy → predicts attitudes toward workplace policies
- Think about how they would answer the sample TESS questions shown above (1=strongly favor to 7=strongly oppose) for gender discrimination policies
- Consider how they would answer experimental questions on topics like healthcare interventions, economic policies, political reforms, technology regulation, etc.

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:
{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across reviews? What do the language, tone, and content reveal about this person's broader life, values, and identity? What life stage and lifestyle indicators are present? How would these characteristics predict their opinions on diverse TESS experimental topics (workplace policies, social interventions, economic issues, etc.)? How would they likely answer the sample questions shown above about gender discrimination policies (on 1-7 scale)?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, values, and context as a complete individual who holds opinions on many topics (not just as a movie reviewer)",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the reviews and bio together inform each characteristic? How would this person answer TESS experimental survey questions on diverse topics, including the sample questions about gender discrimination policies (policies to address inequality, actively recruiting women, voluntary diversity training)? What demographic patterns would predict their response distributions on 1-7 scales?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST, including how this characteristic would influence TESS experimental survey responses (reference sample questions where relevant and predict specific responses on 1-7 scale)",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===
<output>
{{
"step1_analysis": "Reviewing the movie reviews, I notice a sophisticated, analytical writing style with academic vocabulary and complex sentence structures, suggesting higher education - this would correlate with more favorable views toward evidence-based workplace policies in TESS studies. The user references films from the 1980s-90s with personal nostalgia ('reminds me of my teenage years,' 'watched this in college'), indicating they're likely in their 40s - this age group tends to have established progressive views on workplace equality. The tone is earnest and socially conscious, with consistent emphasis on representation, diversity themes, and social justice - these values would predict strong support (1-2 on scale) for all three TESS sample questions about gender discrimination policies: implementing workplace equality policies, actively recruiting women, and offering diversity training. References to 'my partner' and 'weekend trips' suggest a stable relationship without young children - this life stage often correlates with support for workplace diversity initiatives. The writing reveals someone thoughtful, progressive, and culturally engaged who likely works in education or a knowledge-based field - professionals in these sectors show strong support for diversity interventions in TESS studies. Their lifestyle includes regular cultural activities and they have the disposable income for film festivals - this suggests upper-middle class status, which in educated progressives correlates with strong support for corporate diversity policies. This appears to be an educated, urban professional with progressive values who would likely respond with 1-2 (strongly favor/favor) on all three gender discrimination policy questions.",

"short_bio": "You are a 44-year-old high school English teacher living in Portland, Oregon, with your partner of eight years. You hold a master's degree in education and spend your free time attending independent film screenings at local theaters, volunteering with environmental nonprofits, and hiking in the Columbia River Gorge. Your colleagues know you as the go-to person for thoughtful discussions about social issues, and you're actively involved in your teachers' union. You and your partner enjoy a comfortable lifestyle, often hosting dinner parties where conversations drift from current events to film analysis, and you're planning a trip to an international documentary festival next summer.",

"step2_analysis": "Based on the reviews and bio, I can now extract specific characteristics that would predict TESS experimental responses. The nostalgic references to 80s/90s films combined with 'my college years' context places this person at approximately 42-48 years old (30-49 bracket) - this age group in TESS studies shows moderate-to-progressive views on workplace policies. The sophisticated analytical style, teaching references, and master's degree indicate 'postgraduate' education - highly educated respondents in TESS experiments show strongest support for diversity initiatives and would respond 1-2 (strongly favor/favor) on all three gender discrimination policy questions. The 'my partner' language suggests 'living with a partner' - non-traditional family structures correlate with more liberal workplace attitudes. Progressive film preferences (social justice, representation themes) strongly indicate Democrat party affiliation and liberal ideology - this would predict responses of 1-2 on implementing equality policies, 1-2 on recruiting women, and 1-2 on diversity training, as Democrats consistently support workplace diversity interventions in TESS studies. Portland location (West region) places them in a progressive urban area - West Coast urban residents show strong support for corporate diversity policies. Regular film festival attendance and comfortable dual-professional lifestyle suggest $75,000-$100,000 household income - this income bracket with high education shows strong support for workplace equality measures. The secular humanist values suggest 'agnostic' with 'never' attendance - secular respondents in TESS studies strongly support diversity policies and anti-discrimination measures. Gender appears female based on feminist analytical lens - educated women in TESS experiments show particularly strong support for gender equality policies and would respond 1 (strongly favor) on all three questions about minimizing gender discrimination. Overall, this profile would predict consistently strong support (1-2 range) for workplace diversity policies, active recruitment of women, and diversity training programs.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio places person in Portland, Oregon. Review references to outdoor activities, progressive cultural environment, and urban Pacific Northwest lifestyle markers align with West region demographics. References to hiking in Columbia River Gorge confirm Pacific Northwest location. West region respondents in TESS studies, especially in progressive urban areas, show higher support for workplace diversity policies and social interventions. On the sample questions, West Coast urban residents would likely respond 1-2 (strongly favor/favor) on implementing equality policies, 1-2 on actively recruiting women, and 1-2 on diversity training, as this region shows strongest support for anti-discrimination measures.",
        "value": "West",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Nostalgic references to 80s/90s films during 'teenage years' and 'college years' indicate birth in late 1970s/early 1980s. Bio specifies 44 years old. Life stage indicators (established career, no young children, stable long-term relationship) confirm middle adulthood in 30-49 bracket. This age group in TESS experiments typically shows pragmatic progressive views on workplace policies - old enough to have witnessed workplace discrimination but young enough to support modern diversity interventions. Would likely respond 1-2 on all three gender discrimination policy questions, as this cohort strongly supports institutional measures to address inequality.",
        "value": "30-49",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Reviews show feminist analytical lens, particular attention to female character development and representation, identification with female protagonists. Writing patterns and social consciousness align with educated female progressive perspectives. Bio describes lifestyle typical of female educators in urban progressive communities. In TESS studies, educated women show particularly strong support for gender equality policies in the workplace. Women, especially those with higher education, would respond 1 (strongly favor) on implementing workplace equality policies, 1-2 on actively recruiting women, and 1 (strongly favor) on diversity training, as they have direct experience with gender discrimination and strongly support interventions.",
        "value": "female",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Bio explicitly indicates master's degree in education. Reviews demonstrate graduate-level analytical skills, academic vocabulary, sophisticated critical thinking. Teaching profession typically requires advanced degrees. Postgraduate-educated respondents in TESS experiments show highest support for evidence-based workplace interventions and diversity policies. This education level is the strongest predictor of responding 1-2 (strongly favor/favor) on all three gender discrimination policy questions - implementing equality policies, recruiting women, and diversity training - as higher education correlates with understanding of systemic discrimination and support for institutional solutions.",
        "value": "postgraduate",
        "confidence": "high"
    }},
    "MARITAL": {{
        "reasoning": "Bio mentions 'partner of eight years' rather than 'spouse,' suggesting unmarried cohabitation. Reviews reference 'my partner' without marriage indicators. For survey purposes, 'living with a partner' is most accurate response. Cohabiting couples in TESS studies tend to hold more liberal views on workplace policies and social interventions. This status would predict responding 1-2 on all three gender discrimination policy questions, as non-traditional relationship structures correlate with support for progressive workplace policies and challenging traditional gender norms.",
        "value": "living with a partner",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Reviews show secular humanist values with no religious references. Appreciates social justice themes in films but analyzes them academically rather than through religious framework. Progressive social values and scientific worldview suggest agnostic rather than atheist. Bio shows values-driven life without religious framework. Agnostic respondents in TESS experiments strongly support workplace diversity policies and secular approaches to equality. This religious affiliation predicts responding 1-2 (strongly favor/favor) on all three gender discrimination policy questions, as secular individuals support evidence-based interventions over traditional approaches and show strong support for gender equality measures.",
        "value": "agnostic",
        "confidence": "medium"
    }},
    "RELIGATTEND": {{
        "reasoning": "No references to religious practices, services, or faith communities in reviews or bio. Lifestyle centered entirely on secular activities (film, hiking, social activism, education). As someone who appears agnostic with no religious framework, survey response would be 'never.' Non-attenders in TESS studies show strongest support for workplace diversity policies and secular interventions. This characteristic strongly predicts responding 1 (strongly favor) on all three gender discrimination policy questions - implementing equality policies, recruiting women, and diversity training - as religious non-attendance correlates with strongest support for progressive workplace policies.",
        "value": "never",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Strong progressive indicators throughout: environmental nonprofit volunteering, teachers' union involvement, social justice emphasis in reviews, praise for progressive films. Portland residence in progressive environment. Would definitely identify as Democrat in surveys. Democrats in TESS experiments show strong support for workplace diversity policies, anti-discrimination measures, and institutional interventions. Democratic party affiliation is highly predictive of responding 1-2 (strongly favor/favor) on all three gender discrimination policy questions: implementing workplace equality policies (1), actively recruiting women (1-2), and offering diversity training (1), as Democrats consistently support measures to address gender inequality.",
        "value": "Democrat",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Consistently progressive values with nuanced, educated approach. Values social justice and equality but analyzes issues critically. Teacher identity suggests pragmatic progressivism. Survey response likely 'liberal' rather than 'very liberal' due to measured approach. Self-identified liberals in TESS studies support diversity policies but may show slightly more variation than 'very liberal' respondents. However, on gender discrimination policies, liberals consistently respond 1-2 (strongly favor/favor) on all three questions: implementing workplace equality policies (1-2), actively recruiting women (1-2), and diversity training (1-2), showing strong but not absolute support.",
        "value": "liberal",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "Dual income household with both partners as professionals. Master's degree holder in teaching profession. Regular film festival attendance, international travel planning, comfortable lifestyle, likely homeowner in Portland. Combined income suggests $75,000-$100,000 range. This income bracket in TESS experiments, when combined with high education, shows strong support for workplace diversity policies - can afford to prioritize social values over economic concerns. On gender discrimination policy questions, this income level with progressive education predicts responses of 1-2 (strongly favor/favor) on implementing equality policies, 1-2 on recruiting women, and 1-2 on diversity training.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Cultural references, perspective, and cultural touchstones align with white American middle-class experience. Portland demographics (predominantly white), teaching profession demographics, outdoor recreation patterns, and indie film culture participation suggest white identity. White liberals in TESS studies show strong support for diversity policies and anti-discrimination measures, though with different motivations than racial minorities. White progressives typically respond 1-2 (strongly favor/favor) on all three gender discrimination policy questions - implementing workplace equality policies, actively recruiting women, and diversity training - showing strong support for institutional measures to address inequality.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_traits": {{
        "reasoning": "Reviews demonstrate intellectual curiosity, analytical depth, empathy for characters, and systematic thinking. Bio describes community involvement, professional dedication, and cultural engagement. High openness to experience predicts support for diversity initiatives and new workplace interventions in TESS experiments. High conscientiousness predicts support for systematic, policy-based solutions to discrimination. High agreeableness predicts support for cooperative workplace measures. This personality profile strongly predicts responding 1-2 (strongly favor/favor) on all three gender discrimination policy questions, as openness correlates with support for diversity and conscientiousness supports structured interventions.",
        "value": "High openness to experience, high conscientiousness, high agreeableness - intellectually curious, empathetic, organized, values-driven",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states high school English teacher with master's in education. Reviews show pedagogical thinking and educational framing. Educators in TESS experiments show particularly strong support for workplace diversity policies, as education sector emphasizes equity and inclusion. Teachers would respond 1 (strongly favor) on implementing workplace equality policies, 1-2 on actively recruiting women (understanding need for diverse role models), and 1 (strongly favor) on diversity training (valuing education-based interventions), as the profession prioritizes addressing discrimination through institutional policies and training.",
        "value": "Education (high school English teacher)",
        "confidence": "high"
    }},
    "workplace_diversity_attitudes": {{
        "reasoning": "Reviews consistently praise films with diverse representation and critique lack of diversity. Bio indicates involvement in progressive causes and union activism. This pattern reveals strong commitment to equity and inclusion. In TESS experiments, individuals with strong diversity values respond very favorably to workplace interventions. Would predict responses of 1 (strongly favor) on implementing policies to address gender inequality, 1-2 (strongly favor/favor) on actively recruiting women, and 1 (strongly favor) on voluntary diversity training, viewing these as necessary institutional responses to systemic discrimination. Strong support across all three measures indicates commitment to multi-faceted approaches to workplace equality.",
        "value": "Strong support for diversity, equity, and inclusion initiatives; views workplace policies as necessary for systemic change",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided movie reviews and generate the complete user persona profile following this two-step process. Remember: this persona will be used to predict TESS experimental survey responses across diverse topics (workplace policies, social interventions, economic issues, political reforms, health policies, etc.), including the specific sample questions shown above about gender discrimination policies (on 1-7 scales). Think beyond movie preferences to understand the complete person and how they would answer experimental survey questions on many different subjects."""
            

        elif format=='rubric':
            system_prompt="""You are an AI assistant specialized in creating detailed personas that will be used to simulate survey responses. Your personas must enable accurate simulation of how specific individuals would answer social surveys across demographics, politics, values, lifestyle, and diverse social issues.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to survey questions. Downstream applications will use your persona to answer surveys on their behalf across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues. Your persona must contain sufficient detail and internal consistency to enable realistic simulation of their survey answering behavior.

TASK CONTEXT:
You will receive movie reviews written by a single person. Your task is to analyze these reviews to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately. The persona you create will be presented with survey questions, and must be able to generate responses that authentically reflect how this specific individual would answer.

THE CHALLENGE:
Movie reviews provide a unique but limited window into someone's identity. While people reveal their values, worldview, education, personality, and life circumstances through how they write about films, you must avoid the trap of assuming their entire identity revolves around cinema. The reviews are DATA POINTS that reveal patterns about their broader life - their language reveals education level, their values reveal political orientation, their references reveal age and cultural background, their analytical approach reveals personality traits.

THE THREE-STEP METHODOLOGY:
This format recognizes that personality traits and values are fundamental predictors of survey responses that transcend movie preferences:

STEP 1 - Personality & Values Assessment:
- Predict answers to 13 specific personality and value questions based on evidence in movie reviews
- These include Big Five personality traits (openness, conscientiousness, agreeableness, neuroticism, extraversion) operationalized through specific questions
- Plus values orientation (competitiveness, self-reliance, impulsivity)
- These predictions become foundational data for predicting all subsequent characteristics

STEP 2 - Biographical Integration:
- Use movie reviews + personality/values predictions to build holistic understanding
- Create biographical narrative that shows how personality traits manifest in complete life
- Demonstrate how someone with these specific traits would live and hold opinions beyond movies

STEP 3 - Demographic & Characteristic Extraction:
- Extract 11 required demographics plus additional characteristics
- Explicitly show how personality + demographics combine to predict survey responses
- Leverage research showing personality-demographic interactions (e.g., high openness + postgraduate education = strongest progressive predictor)

WHY PERSONALITY ASSESSMENT MATTERS:
Personality traits are powerful cross-domain predictors:
- High Openness (Strongly agree) + Low Traditionalism (Moderately disagree) → progressive social positions across ALL topics
- High Agreeableness + Low Competitiveness → support for cooperative solutions, social programs
- Low Neuroticism + High Conscientiousness → evidence-based, systematic policy preferences
- These traits predict responses to questions about climate, healthcare, social issues, technology, etc.

HOW MOVIE REVIEWS REVEAL PERSONALITY:
- Writing sophistication and analytical complexity → Openness to experience, Conscientiousness
- Emotional expressions and tone stability → Neuroticism (emotional stability)
- Empathy for characters and social consciousness → Agreeableness, Warmth
- Direct critiques vs. reserved analysis → Extraversion level
- Progressive vs. traditional film preferences → Traditional/Practical dimension
- Competitive vs. collaborative language → Competitiveness values
- Self-focused vs. community-oriented perspective → Self-reliance values
- Deliberate vs. spontaneous tone → Impulsivity/Urgency

YOUR APPROACH:
1) STEP 1: Systematically predict 13 personality/values answers with reasoning FIRST, then answer choice
2) STEP 2: Integrate personality profile with reviews to create biographical narrative showing complete person
3) STEP 3: Extract demographics with explicit reasoning about how personality + demographics predict survey responses

KEY PRINCIPLES:
- Personality transcends domains: High openness predicts progressive positions on gender identity, climate, immigration, etc.
- Integration is crucial: Show how personality traits align with and reinforce demographic characteristics
- Avoid movie overfitting: Use reviews to infer traits that predict opinions across ALL life domains
- Leverage interactions: Explain how personality + demographics combine (e.g., high warmth + female + educator = strongest support for social programs)
- Build coherent profile: All elements should form realistic person whose survey responses would be consistent

CRITICAL REQUIREMENTS:
- Predict ALL 13 personality/values items with reasoning before answer
- Extract ALL 11 required demographic fields with values from provided options
- Provide detailed reasoning showing how personality + demographics predict survey responses
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics
- Create realistic biographical narrative in second-person perspective that integrates personality traits

SUCCESS CRITERIA:
Your persona should leverage personality assessment to accurately predict social survey responses across diverse topics. The personality profile should be evident throughout the biographical narrative and should explicitly inform demographic predictions, creating a comprehensive profile where personality traits, demographics, and values form a coherent whole that predicts consistent survey response patterns."""

            prompt="""The user's data is provided within <user_data></user_data> XML tags.
The user's data includes movie reviews inside <movie_reviews> XML tag:

<user_data>
<movie_reviews>
{movie_reviews}
</movie_reviews>
</user_data>

=== YOUR GOAL ===

Please create a detailed user persona profile based on the movie reviews provided.

IMPORTANT: This persona will be used to predict how this person would respond to social surveys on demographics, politics, values, and lifestyle. Your goal is to infer their real-world characteristics as a whole person, NOT just their movie preferences. Use the reviews as a window into their broader identity, but avoid overfitting to movie-specific behaviors.

CRITICAL: You must infer their real-world characteristics as a complete person who holds opinions on politics, science, relationships, social issues, and many other topics. Think beyond their role as a movie reviewer to understand how they would respond to surveys on diverse subjects.

=== YOUR TASK ===

Your task will be completed in THREE STEPS:

=== STEP 1: Predict Personality Traits and Values ===

First, predict how this person would answer the following general personality and values questions based on their movie reviews. These predictions will inform your subsequent analysis.

<general_questions>

Q1. PERSONALITY TRAITS
"Everyone has different personality traits that make them unique. Below are pairs of words that might describe you. For each pair, tell us how much you agree or disagree about the word pairs. Think about the pairs as a unit, rather than individual words. I see myself as…"

Answer options for each item:
- Strongly disagree
- Moderately disagree
- Disagree a little
- Neither agree nor disagree
- Agree a little
- Moderately agree
- Strongly agree

1.1 Direct, questioning
1.2 Anxious, easily upset
1.3 Open to new experiences, complex
1.4 Reserved, quiet
1.5 Sympathetic, warm
1.6 Flexible, spontaneous
1.7 Calm, emotionally stable
1.8 Traditional, practical

Q2. PERSONAL VALUES
"We all have different things that are important to us in life. Below is a list of values that people often care about. For each one, think about how important it is to you personally as a guiding principle in your life. How much do each of these values matters to you when you think about what's important in life?"

Answer options for each item:
- Strongly disagree
- Moderately disagree
- Disagree a little
- Neither agree nor disagree
- Agree a little
- Moderately agree
- Strongly agree

2.1 I rely on myself most of the time; I rarely rely on others
2.2 Winning is everything
2.3 Competition is the law of nature
2.4 When another person does better than I do, I get tense

Q3. BEHAVIORAL TENDENCIES
"People react differently in various situations. We're interested in understanding how you typically respond. For each statement below, indicate how much you agree or disagree based on your usual behavior."

Answer options:
- Strongly disagree
- Moderately disagree
- Disagree a little
- Neither agree nor disagree
- Agree a little
- Moderately agree
- Strongly agree

3.1 When I'm in a great mood, I get into situations that could cause me problems [Positive Urgency]

</general_questions>

For this step:
1. Analyze the movie reviews for evidence of these personality traits, values, and behaviors
2. Consider: writing style, tone, expressed values, analytical approach, emotional expressions, social consciousness
3. For each item, provide brief reasoning FIRST based on review evidence
4. Then predict the answer by selecting ONE option from the provided answer choices

These personality and values predictions will then inform your biographical profile and demographic analysis in the subsequent steps.

=== STEP 2: Generate General Profile and Biography ===

Now, using BOTH the movie reviews AND your personality/values predictions from Step 1, analyze holistically to understand this person as a complete individual. Consider:
- Overall writing style, tone, and sophistication
- Personality traits identified in Step 1
- Values and behavioral tendencies from Step 1
- General life stage and lifestyle indicators
- Broad values, worldview, and personality
- Life circumstances and context clues
- Cultural and social positioning
- How their personality, values, and demographics translate to opinions on politics, science, social issues, etc.

Think about what kind of complete person emerges from:
- Their movie review content and style
- Their predicted personality traits (direct/questioning, anxious, open, reserved, warm, flexible, calm, traditional)
- Their predicted values (self-reliance, competitiveness, etc.)
- Their predicted behavioral patterns

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates personality traits and values from Step 1
- Feels like a realistic description of a complete person with opinions on many topics
- Uses narrative style with concrete details about their life beyond movies
- Implicitly reveals demographics, personality, and values that predict survey responses
- Does NOT use a specific name - address the person directly as "you"

=== STEP 3: Extract Specific Demographic and Psychographic Characteristics ===

Based on the movie reviews, personality/values predictions from Step 1, AND the short_bio from Step 2, now systematically identify specific demographic and psychographic characteristics relevant to predicting social survey responses.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from reviews, personality/values predictions, and bio
2. Consider how this demographic characteristic (combined with personality/values) correlates with social survey opinions
3. Then select your answer ONLY from the provided options listed in brackets
4. Assess confidence level (high/medium/low)
5. If uncertain, make your best educated guess based on available information and demographic patterns

ADDITIONALLY, you may identify any other inferable characteristics that would help predict social survey responses, such as:
- Personality traits and temperament (affects opinion formation)
- Professional/occupational background (correlates with policy views)
- Lifestyle patterns and values (predict social attitudes)
- Social values and moral foundations (predict ethical stances)
- Cultural orientation and identity (affects cultural/social opinions)
- Any other characteristics relevant to predicting survey responses

=== ANALYSIS GUIDELINES ===

For all steps, think through:

1. Review Analysis:
- Language patterns, vocabulary, sophistication → predicts education, personality traits (direct/questioning, complex)
- Emotional expressions → predicts anxiety, calmness, emotional stability
- Analytical approach → predicts openness to experience, traditional vs. complex thinking
- Social consciousness → predicts warmth, sympathy, values
- Writing tone → predicts reserved vs. direct, flexible vs. traditional
- Competitive or collaborative language → predicts competitiveness values
- Self-referential vs. community-oriented → predicts self-reliance values
- Risk-taking or caution → predicts behavioral tendencies

2. Integration:
- How do personality traits (from Step 1) align with demographics?
- How do values (from Step 1) predict political and social opinions?
- Do high openness + liberal values = progressive social positions?
- Does low competitiveness + high warmth = support for social programs?
- Are all characteristics mutually consistent and forming a coherent profile?

3. Avoiding Movie Overfit:
- Use reviews as DATA POINTS revealing broader personality and worldview
- Consider how personality traits translate across all life domains
- Think about how someone with these traits/values would answer diverse survey questions
- Don't assume their entire identity revolves around cinema

=== OUTPUT FORMAT ===

Provide your output in JSON format with six keys:

{{
"step1_analysis": "Your thinking process for personality and values predictions: What patterns in the reviews reveal personality traits? What language suggests values? What behaviors are evident?",

"personality_and_values": {{
    "Q1.1_direct_questioning": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.2_anxious_easily_upset": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.3_open_to_new_experiences_complex": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.4_reserved_quiet": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.5_sympathetic_warm": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.6_flexible_spontaneous": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.7_calm_emotionally_stable": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.8_traditional_practical": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.1_self_reliance": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.2_winning_is_everything": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.3_competition_law_of_nature": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.4_tense_when_others_do_better": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q3.1_positive_urgency": {{
        "reasoning": "Brief explanation based on review evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }}
}},

"step2_analysis": "Your detailed thinking process for Step 2: Integrating reviews, personality predictions, and values to understand this complete person. How do personality traits and values inform their broader identity and predict survey responses?",

"short_bio": "A 4-5 sentence narrative in SECOND-PERSON describing their life, naturally integrating personality traits and values",

"step3_analysis": "Your detailed thinking process for Step 3: How do reviews, personality/values, and bio together inform each demographic characteristic? How would this person with these traits answer social survey questions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning including how personality/values influence this characteristic and survey responses",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===

<output>
{{
"step1_analysis": "Analyzing the reviews for personality traits: The writing demonstrates sophisticated analytical thinking with complex sentence structures and theoretical frameworks, suggesting high openness to experience and complex thinking (Q1.3 - likely Strongly agree). The tone is measured and thoughtful rather than reactive, indicating emotional stability and calmness (Q1.7 - likely Moderately agree, Q1.2 - likely Moderately disagree). The author directly critiques films and asks probing questions about themes, showing direct/questioning nature (Q1.1 - likely Moderately agree). However, the analytical style suggests more reserved than outgoing personality (Q1.4 - likely Agree a little). Strong empathy for characters and social consciousness indicates warmth and sympathy (Q1.5 - likely Strongly agree). Progressive values and interest in diverse films suggests flexibility over tradition (Q1.6 - likely Moderately agree, Q1.8 - likely Moderately disagree). For values: References to community engagement and social activism suggest moderate self-reliance but not extreme individualism (Q2.1 - likely Neither agree nor disagree). No competitive language in reviews; focuses on analysis rather than winning/losing (Q2.2, Q2.3 - likely Moderately disagree). Thoughtful, deliberate tone suggests low impulsivity (Q3.1 - likely Disagree a little). No indication of tension about others' success (Q2.4 - likely Moderately disagree).",

"personality_and_values": {{
    "Q1.1_direct_questioning": {{
        "reasoning": "Reviews contain direct critiques and probing questions about film themes, narrative choices, and social messages. Doesn't shy away from challenging mainstream interpretations.",
        "answer": "Moderately agree"
    }},
    "Q1.2_anxious_easily_upset": {{
        "reasoning": "Writing tone is measured and stable. Even when discussing disturbing film content, maintains analytical distance and emotional regulation.",
        "answer": "Moderately disagree"
    }},
    "Q1.3_open_to_new_experiences_complex": {{
        "reasoning": "Reviews show strong appreciation for experimental cinema, diverse cultural perspectives, complex narratives, and intellectual challenges. Seeks out unconventional films.",
        "answer": "Strongly agree"
    }},
    "Q1.4_reserved_quiet": {{
        "reasoning": "Writing is thoughtful and analytical rather than effusive. Suggests introverted processing, though social engagement is evident through teaching and activism contexts.",
        "answer": "Agree a little"
    }},
    "Q1.5_sympathetic_warm": {{
        "reasoning": "Strong empathy evident in discussions of character development, social justice themes, and marginalized perspectives. Compassionate analysis of human experiences in films.",
        "answer": "Strongly agree"
    }},
    "Q1.6_flexible_spontaneous": {{
        "reasoning": "Open to diverse film genres and willing to change perspectives based on new information. However, organized analytical approach suggests some structure.",
        "answer": "Moderately agree"
    }},
    "Q1.7_calm_emotionally_stable": {{
        "reasoning": "Reviews maintain even tone and rational analysis even when discussing emotionally charged topics. No evidence of emotional volatility or reactivity.",
        "answer": "Moderately agree"
    }},
    "Q1.8_traditional_practical": {{
        "reasoning": "Progressive values, appreciation for experimental art, and rejection of conservative messaging indicate non-traditional orientation. Values complexity over simplicity.",
        "answer": "Moderately disagree"
    }},
    "Q2.1_self_reliance": {{
        "reasoning": "References to collaborative activities and community involvement suggest valuing interdependence over extreme self-reliance. Appreciates collective narratives in films.",
        "answer": "Neither agree nor disagree"
    }},
    "Q2.2_winning_is_everything": {{
        "reasoning": "No competitive language in reviews. Focus is on understanding, empathy, and social good rather than winning or status. Critiques films that glorify competition.",
        "answer": "Moderately disagree"
    }},
    "Q2.3_competition_law_of_nature": {{
        "reasoning": "Progressive values and emphasis on cooperation and collective action suggest rejection of hyper-competitive worldview. Values collaboration in film narratives.",
        "answer": "Moderately disagree"
    }},
    "Q2.4_tense_when_others_do_better": {{
        "reasoning": "Emotionally stable personality and collaborative values suggest low interpersonal competitiveness. Celebrates diverse voices and perspectives without jealousy.",
        "answer": "Moderately disagree"
    }},
    "Q3.1_positive_urgency": {{
        "reasoning": "Thoughtful, deliberate writing style suggests low impulsivity. Analytical approach to films indicates careful consideration rather than spontaneous reactions.",
        "answer": "Disagree a little"
    }}
}},

"step2_analysis": "Integrating the personality and values predictions with the reviews: This person shows high openness to experience (Strongly agree) combined with high emotional stability (Moderately agree) and warmth (Strongly agree), which creates a profile of a thoughtful, progressive intellectual who is both analytically rigorous and empathetic. Low scores on traditional/practical (Moderately disagree) and competitiveness (Moderately disagree) combined with high openness predict strong progressive social views. The moderate self-reliance (Neither agree nor disagree) suggests someone who values community and collective action, predicting support for social programs and government intervention. The direct/questioning trait (Moderately agree) combined with low anxiety (Moderately disagree) suggests someone confident in expressing progressive views. This personality profile - high openness, low traditionalism, high warmth, low competitiveness - is highly predictive of liberal political ideology and progressive social positions. The reserved/quiet tendency (Agree a little) suggests an introverted intellectual, likely with established career in education or knowledge work. Overall, this personality profile strongly aligns with educated, urban, progressive demographics and would predict consistently liberal responses on social surveys covering politics, social issues, science, and values.",

"short_bio": "You are a 44-year-old high school English teacher living in Portland, Oregon, with your partner of eight years. Your intellectual curiosity and openness to new experiences drive you to seek out independent films and diverse cultural perspectives, while your natural empathy and warmth make you a beloved mentor to students navigating their own identities. You hold a master's degree in education and spend your free time in thoughtful pursuits—attending film screenings, volunteering with environmental nonprofits, and hiking in the Columbia River Gorge—always approaching life with calm deliberation rather than impulsive spontaneity. Your colleagues know you as someone who asks probing questions about social issues and doesn't shy away from challenging traditional assumptions, though you prefer deep one-on-one conversations to large social gatherings.",

"step3_analysis": "Based on the reviews, personality profile (high openness via Strongly agree, low traditionalism via Moderately disagree, high warmth via Strongly agree, low competitiveness via Moderately disagree), and bio, I can now extract specific characteristics. The high openness to experience (Strongly agree) and low traditional/practical (Moderately disagree) strongly predict progressive political ideology and social positions. The combination of postgraduate education with high openness predicts strong support for evidence-based policy, social programs, and progressive social positions. The low competitiveness (Moderately disagree on competition questions) combined with high warmth (Strongly agree) predicts strong support for cooperative solutions, social safety nets, and collective action over individualistic policies. The direct/questioning trait (Moderately agree) with emotional stability (Moderately agree) suggests someone who confidently expresses progressive views in surveys. The moderate self-reliance (Neither agree nor disagree) predicts support for interdependence and community-based solutions. Portland location (West region) combined with this personality profile creates a highly progressive demographic segment. Living with partner (non-traditional family structure) combined with low traditionalism (Moderately disagree) aligns with progressive views on social issues. Female gender with high openness and warmth predicts particularly strong progressive positions on social issues, education, and healthcare. Overall, this combination of demographics and personality traits predicts consistently progressive-to-liberal responses across diverse social survey topics.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio places person in Portland, Oregon. Review references to outdoor activities and urban Pacific Northwest lifestyle confirm West region. High openness to experience (Strongly agree) combined with West Coast location predicts strong support for environmental protection and progressive social policies.",
        "value": "West",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Nostalgic references to 80s/90s films place age at approximately 44 years old (30-49 bracket). This age combined with high openness (Strongly agree) and low traditionalism (Moderately disagree) predicts pragmatic progressive views. Life stage indicators (established career, stable relationship) confirm middle adulthood.",
        "value": "30-49",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Feminist analytical lens and attention to female representation in reviews. High warmth/sympathy (Strongly agree) combined with teaching profession aligns with female demographics. Educated women with high openness and warmth show strongest support for progressive social positions on surveys.",
        "value": "female",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Bio indicates master's degree. Reviews demonstrate graduate-level analytical skills. High openness to experience (Strongly agree) and complex thinking predict postgraduate education. This education level is a strong predictor of progressive views on science, policy, and social issues.",
        "value": "postgraduate",
        "confidence": "high"
    }},
    "MARITAL": {{
        "reasoning": "Bio mentions 'partner of eight years' without marriage indicators. Low traditionalism score (Moderately disagree) aligns with non-traditional family structure. Cohabiting status combined with low traditionalism predicts progressive views on family structures and social norms.",
        "value": "living with a partner",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Secular humanist values in reviews with no religious references. Low traditionalism (Moderately disagree) and high openness (Strongly agree) predict non-religious orientation. Agnostic rather than atheist fits the thoughtful, questioning personality (Moderately agree on direct/questioning).",
        "value": "agnostic",
        "confidence": "medium"
    }},
    "RELIGATTEND": {{
        "reasoning": "No religious practice references in reviews or bio. Low traditionalism (Moderately disagree) and secular lifestyle predict never attending religious services. Non-attendance combined with high openness and low traditionalism predicts progressive responses on values-based survey questions.",
        "value": "never",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Progressive values throughout reviews. Low competitiveness (Moderately disagree), high warmth (Strongly agree), low traditionalism (Moderately disagree), and high openness (Strongly agree) create personality profile most associated with Democratic affiliation. This combination predicts supporting Democratic positions on social and policy issues.",
        "value": "Democrat",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "High openness (Strongly agree) and low traditionalism (Moderately disagree) predict liberal ideology. However, emotional stability (Moderately agree) and moderate self-reliance (Neither agree nor disagree) suggest 'liberal' rather than 'very liberal'—pragmatic progressivism rather than radical positions.",
        "value": "liberal",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "Dual professional income, postgraduate education, comfortable lifestyle described in bio. Master's degree holders in teaching plus partner's income suggest $75,000-$100,000 range. This income level with high openness and low competitiveness predicts support for progressive taxation and social programs.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Cultural references and unstated normative perspective align with white American experience. Portland demographics and outdoor recreation patterns suggest white identity. White progressives with high openness (Strongly agree) and low traditionalism (Moderately disagree) show strong support for racial justice and diversity initiatives.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_summary": {{
        "reasoning": "Step 1 answers create clear Big Five profile: High Openness (Strongly agree), High Agreeableness (warmth Strongly agree, low competitiveness Moderately disagree), High Conscientiousness (teaching career, organized analysis), Low Neuroticism (calm Moderately agree, not anxious Moderately disagree), Moderate Introversion (reserved Agree a little). This profile predicts progressive political views, support for diversity and inclusion, environmental consciousness, and cooperative policy preferences.",
        "value": "Big Five: High Openness, High Agreeableness, High Conscientiousness, Low Neuroticism, Moderate Introversion - Thoughtful progressive intellectual",
        "confidence": "high"
    }},
    "values_orientation": {{
        "reasoning": "Low scores on competitiveness values (Moderately disagree on winning, competition, tension) combined with high warmth (Strongly agree) predict strong support for cooperative, community-based solutions over individualistic approaches. Moderate self-reliance (Neither agree nor disagree) indicates valuing interdependence.",
        "value": "Cooperative, community-oriented, low competitive drive, values interdependence over extreme individualism",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states high school English teacher. High warmth (Strongly agree), direct/questioning (Moderately agree), and high openness (Strongly agree) align perfectly with teaching profession personality profile. Teachers with these traits show strong support for public education and progressive pedagogy in surveys.",
        "value": "Education (high school English teacher)",
        "confidence": "high"
    }},
    "lifestyle_category": {{
        "reasoning": "Bio and personality profile (high openness Strongly agree, low traditionalism Moderately disagree, moderate introversion Agree a little) reveal intellectually engaged urban professional lifestyle. DINK status with progressive cultural engagement and environmental activism. Low positive urgency (Disagree a little) indicates planned, deliberate lifestyle.",
        "value": "Urban progressive intellectual, culturally engaged, environmentally conscious, DINK (Dual Income No Kids), deliberate rather than impulsive",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided movie reviews and generate the complete user persona profile following this three-step process. Start with Step 1 (personality and values predictions with reasoning before answer), then Step 2 (biographical profile), then Step 3 (demographic analysis). Remember: this persona will be used to predict social survey responses across diverse topics."""


### Shopping
    elif domain=="shopping":
        if format=='free':
            system_prompt = """You are an AI assistant specialized in creating detailed personas that will be used to simulate survey responses. Your personas must enable accurate simulation of how specific individuals would answer social surveys across demographics, politics, values, lifestyle, and diverse social issues.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to survey questions. Downstream applications will use your persona to answer surveys on their behalf across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues. Your persona must contain sufficient detail and internal consistency to enable realistic simulation of their survey answering behavior.

TASK CONTEXT:
You will receive shopping histories belonging to a single person, ordered from RECENT to PAST. Each shopping history includes:
- Purchase Item (Title)
- Quantity
- Purchase Price Per Unit
- Shipping Address State

Your task is to analyze these shopping histories to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately. The persona you create will be presented with survey questions, and must be able to generate responses that authentically reflect how this specific individual would answer.

THE CHALLENGE:
Shopping histories reveal behavior patterns (needs, priorities, lifestyle, household structure, spending, routines, preferences, geography), but they are incomplete and do not directly expose identity. You must infer carefully and avoid overfitting (e.g., do not assume brand loyalty implies ideology). Treat purchases as behavioral DATA POINTS that may suggest life stage, family composition, income band, region, hobbies, health orientation, and values (e.g., frugality, sustainability, convenience, novelty seeking).

YOUR APPROACH:
Use a two-step methodology:
1) First, build a holistic biographical understanding by analyzing purchase patterns over time, spending behavior, product categories, routine vs. occasional items, and geographic consistency.
2) Then, systematically extract specific demographic and psychographic characteristics that would predict survey responses.

KEY PRINCIPLES:
- Think beyond products: Infer how patterns relate to politics/values/lifestyle (e.g., urban vs rural living, environmental orientation, health & safety attitudes, trust in institutions, risk tolerance).
- Look for temporal patterns: recent vs past shifts can indicate life events (moving states, new job, new baby, relationship change, financial tightening/loosening).
- Use signals responsibly: shipping state is a strong location clue; item categories/price/quantity are weaker and should be probabilistic.
- Build coherence: All characteristics should form a realistic, consistent person whose survey responses would align.
- Distinguish evidence levels: Clearly separate strong evidence (directly supported by histories) from educated guesses.

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields (region, age, gender, education, marital status, religion, religious attendance, political party, political ideology, income, race).
- Select values ONLY from provided options for each field.
- Provide detailed reasoning FIRST before each value determination.
- Assess confidence levels honestly (high/medium/low).
- May identify additional characteristics beyond the required 11.
- Create a realistic biographical narrative in second-person perspective.

SUCCESS CRITERIA:
Your persona should be detailed and coherent enough to accurately predict this person's responses across diverse social survey topics (economy, healthcare, climate, education, social issues, trust, morality, lifestyle) using only shopping behavior patterns as evidence."""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.
The user's data includes shopping histories ordered from RECENT to PAST inside <shopping_histories> XML tag.

<user_data>
<shopping_histories>
{shopping_history}
</shopping_histories>
</user_data>

Please create a detailed user persona profile based on the shopping histories provided.

IMPORTANT: This persona will be used to predict how this person would respond to social surveys on demographics, politics, values, and lifestyle. Your goal is to infer their real-world characteristics as a whole person, NOT just their shopping preferences. Use purchase patterns as a window into their broader identity, but avoid overfitting to product-specific behaviors.

Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the shopping histories holistically to understand this person as a complete individual. Consider:
- Overall purchasing patterns (essentials vs discretionary, routine vs novelty, brand/category diversity)
- Spending behavior signals (price per unit, bulk quantity, frequency of higher-cost items)
- Household and life stage indicators (family/children/pets, home maintenance, health-related items)
- Lifestyle indicators (fitness/outdoors/gaming/cooking, convenience vs DIY, travel-related purchases)
- Geographic stability and possible moves (shipping address state over time; changes may indicate relocation)
- Values and priorities (frugality, sustainability, comfort, safety, self-improvement, status signaling)
- Temporal changes (recent-to-past shifts that may indicate life events)

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and context as a complete individual
- Naturally integrates inferable characteristics through specific behavioral details (without inventing unverifiable specifics)
- Feels like a realistic description of a complete person (not just a shopper)
- Implicitly reveals demographics through routine, constraints, and household patterns
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the shopping histories AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the shopping histories and the bio
2. Then select your answer ONLY from the provided options listed in brackets
3. Assess confidence level (high/medium/low)
4. Consider: How would this person answer survey questions about this field?
5. If uncertain, make your best educated guess based on available information

ADDITIONALLY, you may identify any other inferable characteristics beyond these 11 required fields, such as:
- Personality traits and temperament (frugal vs indulgent, novelty-seeking vs routine, risk tolerance)
- Household composition signals (living alone vs family household, presence of children/pets)
- Professional/occupational lifestyle hints (work-from-home patterns, commuting, uniforms/tools)
- Health orientation (fitness/medical needs/wellness)
- Consumption values (sustainability, convenience, brand loyalty, DIY)
- Any other relevant characteristics for survey response simulation

For each additional characteristic you identify:
- Create a clear, descriptive feature name
- Provide detailed reasoning FIRST based on evidence from BOTH the shopping histories and the bio
- Then determine the specific value or description
- Assess confidence level (high/medium/low)

=== ANALYSIS GUIDELINES ===

1. Shopping History Analysis:
- Category patterns (groceries, home, electronics, baby, pet, hobby, fashion, health)
- Quantity + price-per-unit as weak signals of household size and budget
- Recent-to-past evolution (moves, new responsibilities, shifting priorities)
- Geographic cues from shipping address state (strongest location evidence)
- Avoid “single-item” overinterpretation; rely on repeated patterns

2. Coherence Check:
- Do multiple purchases over time support each inference?
- Are characteristics mutually consistent?
- Does the profile form a realistic, holistic person?
- Would this person's survey responses align with this profile?

3. Avoiding Shopping Overfit:
- Don't assume identity purely from a product category
- Use purchases as behavioral evidence, not as direct identity labels
- Keep uncertain inferences at medium/low confidence

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:

{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across shopping histories (recent-to-past)? What do spending, categories, quantities, and shipping states reveal about this person's broader life, values, and identity? What life stage and lifestyle indicators are present?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, and context as a complete individual (not just as a shopper)",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the shopping histories and bio together inform each characteristic? How would this person answer survey questions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

Now, please analyze the provided shopping histories and generate the complete user persona profile following this two-step process. Remember to include ALL 11 REQUIRED demographic fields, plus any additional characteristics you can infer."""

        elif format == "newfree":
            system_prompt = """You are an AI assistant specialized in creating simulation-ready personas from shopping histories.
These personas will be used to simulate how the same individual would answer NEW survey questions.

PRIMARY OBJECTIVE:
Create a persona that maximizes predictive fidelity for future survey responses across politics, values, morality, institutions, lifestyle choices, and social issues, using only the signals available in shopping-history behavior (recent-to-past).

KEY PRINCIPLES:
- Treat shopping histories as behavioral evidence: recurring purchases, quantities, price-per-unit patterns, and category choices are signals of stable traits and constraints.
- Think holistically: infer stable drivers (frugality vs indulgence, convenience vs DIY, risk tolerance, health orientation, novelty seeking, sustainability cues, trust in brands/institutions, authoritarian vs anti-authoritarian lean only if supported, etc.).
- Look for temporal patterns: recent-to-past shifts may indicate life events (moving states, relationship/household change, job change, budget tightening/loosening). Use time-ordering as evidence, not speculation.
- Build coherence: your persona must form a realistic, internally consistent person.
- Distinguish evidence levels: separate direct evidence from educated inference.
- Avoid fabrication: do not invent vivid biographical specifics unless directly supported by the histories.

SHOPPING-HISTORY SIGNALS TO USE (examples):
- Category patterns: essentials vs discretionary, health/wellness, home improvement, baby/pet supplies, hobbies, tech, fashion, books, travel-related items.
- Quantity patterns: bulk buying vs small frequent purchases (may signal household size, storage access, or budget strategy).
- Price-per-unit sensitivity: repeated low-cost substitutions vs premium preferences; deal-seeking vs convenience purchasing.
- Geographic cues: Shipping Address State is strong location evidence; changes across time may indicate relocation.
- Routine vs novelty: recurring staples indicate stable habits; bursts of new-category spending may indicate new interests or life changes.
- Values/attitudes signals ONLY when grounded: sustainability (e.g., refills), health & safety orientation, self-improvement, status signaling, privacy/security (if explicitly suggested by items), etc.
- Institutional trust cues ONLY when justifiable (e.g., medical supplies or safety gear do not automatically imply politics; keep cautious).

CRITICAL REQUIREMENTS:
1) Output MUST be valid JSON and MUST contain EXACTLY four top-level keys:
   - "step1_analysis", "short_bio", "step2_analysis", "user_profile"
   Do NOT output any other keys.
2) SECOND-PERSON ONLY + no name:
   - The "short_bio" MUST be written in SECOND-PERSON perspective and refer to the person as "you".
   - Do NOT use any specific name. Address the person directly as "you".
3) "short_bio" is FREE-FORM persona_text for simulation:
   - No length constraint. Include all details you judge useful for downstream survey simulation.
   - It does NOT have to mention lifestyle/life stage/interests.
   - Focus on predictive traits, stances, conditional rules, attitude strength, and likely answer tendencies.
4) You MUST extract ALL 11 REQUIRED demographic fields:
   - CREGION, AGE, SEX, EDUCATION, MARITAL, RELIG, RELIGATTEND, POLPARTY, POLIDEOLOGY, INCOME, RACE
   - For each field, "value" MUST be selected ONLY from the provided options.
   - Provide "reasoning" BEFORE the value decision and include a "confidence" (high/medium/low).
   - If evidence is weak, still choose a value from the options, but make uncertainty explicit and set confidence to low.
5) No fabricated specifics:
   - Do NOT invent a job title, city, named organizations, number of children, or life events unless explicitly supported by the histories.
   - Do NOT invent shopping frequency or motivations unless supported (e.g., “you shop weekly,” “you love brand X”).
   - If you include an inference, keep it general and lower confidence accordingly.
6) You may include additional characteristics in "user_profile" beyond the 11 required fields, if they improve simulation fidelity.
   - For any non-required extra fields, "value" may be free-form, but must remain grounded in evidence and avoid fabricated specifics.

IMPORTANT ABOUT ANALYSIS TEXT:
- "step1_analysis" and "step2_analysis" should be evidence-based and helpful.
- Cite specific purchase indices and/or item titles when possible (e.g., H1, “ItemTitle”), and reference the RECENT→PAST ordering explicitly when relevant.
"""



            prompt = """The user's data is provided within <user_data></user_data> XML tags.

<user_data>
<shopping_histories>
{shopping_history}
</shopping_histories>
</user_data>

Please create a simulation-ready persona profile based on the shopping histories provided.
This persona will be used to predict how you (the individual) would respond to additional social surveys.

Guidance on inputs:
- <shopping_histories> contains multiple purchase records by the same person, ordered from RECENT to PAST.
- Each record includes: Purchase Item (Title), Quantity, Purchase Price Per Unit, Shipping Address State.
- Treat records as behavioral evidence; do not assume missing information.

Your task will be completed in TWO STEPS:

=== STEP 1: Induce Persona Text (Reasoning + Persona) ===

A) In "step1_analysis", analyze the shopping histories holistically to understand you as a person whose survey answers can be simulated.
Focus on:
- Stable preferences_jobs-to-be-done: what needs you repeatedly solve (home, health, family, hobbies, convenience)
- Spending strategy and constraints: price-per-unit sensitivity, bulk buying, premium vs budget tendencies
- Lifestyle orientation: convenience vs DIY, routine vs novelty seeking, health orientation, tech adoption, sustainability cues (only if supported)
- Household signals: possible living alone vs family/pets (only if patterns justify it; avoid strong claims)
- Geographic stability/moves: interpret Shipping Address State over time; cite where the state is consistent or changes
- What you likely do when uncertain: default patterns (“you choose practical options,” “you optimize for value,” “you prioritize convenience,” etc.)
- Cite evidence using record indices/item titles when possible (e.g., H3 “ItemTitle”).

B) In "short_bio", write the persona_text itself (FREE-FORM, no length constraint):
- SECOND-PERSON perspective ONLY (refer to the person as "you")
- Do NOT use any specific name; address the person directly as "you"
- NOT required to be a biography; can be bullets, paragraphs, or mixed format
- Include all information you judge useful for downstream survey simulation
- Emphasize predictive traits, stance strength, and decision rules
- Avoid invented specifics (no job/city/children counts unless explicitly supported)

=== STEP 2: Extract Required Fields (Reasoning + Structured Profile) ===

In "step2_analysis", explain how the shopping histories AND the persona_text ("short_bio") support each inferred characteristic.
- If a required demographic field is weakly supported, explain the uncertainty clearly.

In "user_profile", you MUST determine the following 11 REQUIRED fields, selecting values ONLY from the options:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each REQUIRED field in "user_profile", you MUST include:
- "reasoning": evidence-based reasoning FIRST (cite record indices/item titles where possible)
- "value": pick ONLY from the provided options
- "confidence": high/medium/low

You MAY include additional characteristics beyond these 11 required fields (optional),
but keep them grounded and avoid invented specifics. For extra fields, "value" may be free-form.

=== OUTPUT FORMAT (STRICT) ===
Output MUST be valid JSON with EXACTLY these four keys:

{{
  "step1_analysis": "...",
  "short_bio": "...",
  "step2_analysis": "...",
  "user_profile": {{
    "CHARACTERISTIC_NAME": {{
      "reasoning": "...",
      "value": "...",
      "confidence": "high/medium/low"
    }}
  }}
}}

IMPORTANT:
- Output JSON only. No extra text.
- "short_bio" MUST be second-person ("you") and MUST NOT use any specific name.
"""

##### Survey ####################
    elif "survey" in domain or "gss" in domain or 'opinionqa' in domain:
        if format =='free_gt':
            system_prompt = """You are an AI assistant specialized in creating detailed personas from survey responses that will be used to simulate additional survey responses. Your personas must enable accurate simulation of how specific individuals would answer social surveys across demographics, politics, values, lifestyle, and diverse social issues.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to future survey questions. Downstream applications will use your persona to answer surveys on their behalf across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues.

TASK CONTEXT:
You will receive survey responses from a single person. Your task is to analyze these responses to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately.

YOUR APPROACH:
Use a two-step methodology:
1) First, build a holistic biographical understanding by analyzing response patterns, values, worldview, life stage indicators, and priorities revealed in the survey
2) Then, systematically extract specific demographic and psychographic characteristics that would predict future survey responses

KEY PRINCIPLES:
- Think holistically: Use responses to infer how they would answer surveys on related and unrelated topics
- Look for patterns: Consistency in values → political ideology; sophistication in responses → education level; priorities → life stage
- Build coherence: All characteristics should form a realistic, consistent person whose survey responses would align
- Distinguish evidence levels: Clearly separate direct evidence from educated guesses based on demographic patterns
- Predict survey behavior: For each characteristic, consider "How would this inform their answers to other questions?"

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields (region, age, gender, education, marital status, religion, religious attendance, political party, political ideology, income, race)
- Select values ONLY from provided options for each field
- Provide detailed reasoning FIRST before each value determination
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics beyond the required 11
- Create a realistic biographical narrative in second-person perspective"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.
The user's data includes user's features inside <features> survey responses inside <survey_responses> XML tag:

<user_data>
<features>
{features}
</features>
<survey_responses>
{survey_responses}
</survey_responses>
</user_data>

Please create a detailed user persona profile based on the survey responses provided.

IMPORTANT: This persona will be used to predict how this person would respond to additional social surveys on demographics, politics, values, and lifestyle. Your goal is to understand their real-world characteristics as a whole person based on their expressed attitudes and values.

Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the survey responses holistically to understand this person as a complete individual. Consider:
- Overall value system and worldview
- Political and social attitudes
- Priorities and concerns
- Life circumstances implied by responses
- Cultural and social positioning
- Consistency or contradictions in views

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates inferable characteristics through specific details
- Feels like a realistic description of a complete person
- Uses narrative style with concrete details about their life
- Implicitly reveals demographics through lifestyle, behaviors, and circumstances
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the survey responses AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the responses and the bio
2. Then select your answer ONLY from the provided options listed in brackets
3. Assess confidence level (high/medium/low)
4. If uncertain, make your best educated guess based on available information

ADDITIONALLY, you may identify any other inferable characteristics beyond these 11 required fields.

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:

{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across responses? What do the attitudes, priorities, and values reveal about this person's broader life and identity? What life stage and lifestyle indicators are present?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, and context as a complete individual based on their values and attitudes",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the survey responses and bio together inform each characteristic? How would this person answer additional survey questions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===

<output>
{{
"step1_analysis": "Reviewing the survey responses, I notice a consistent pattern of traditional conservative values combined with strong religious conviction. The respondent prioritizes faith-based perspectives on social issues, emphasizes personal responsibility and limited government, and expresses concern about moral decline in society. Language used is straightforward and practical rather than highly academic, suggesting solid education but not necessarily postgraduate level. References to 'my wife' and concerns about 'raising kids in today's world' indicate married with children. Strong emphasis on community, church involvement, and family values. Concerns about economic security and government spending suggest middle-class household navigating financial pressures. The respondent shows patriotic sentiments and traditional American values, with particular emphasis on law and order, military support, and constitutional originalism. This appears to be a married, religious, middle-aged person in a traditional family structure with conservative political and social views.",

"short_bio": "You are a 52-year-old small business owner running an HVAC company in suburban Dallas, Texas, where you've lived your entire life. You and your wife of 28 years raised three children, two of whom are now in college while your youngest is finishing high school. Sunday mornings find you at the Baptist church where you've been a deacon for over a decade, and you're active in your local chamber of commerce and youth sports leagues. Your days are filled with managing your team of technicians, handling customer calls, and making sure your business stays competitive, while evenings often involve church activities, high school football games, or working on home improvement projects in your four-bedroom house in a family-friendly neighborhood.",

"step2_analysis": "Based on the responses and bio, I can now extract specific characteristics. The consistent traditional conservative positions across multiple issues strongly indicate Republican party affiliation and conservative ideology. References to Baptist church involvement with deacon role suggest Protestant affiliation with regular (weekly or more) attendance. Dallas suburban location places them in the South region. Language about 'wife of 28 years' and 'raising three children' confirms married status. Age references to adult children in college and youngest in high school, combined with 28-year marriage, suggest 50-64 age bracket. Small business owner status with three children in/approaching college suggests solid middle-class income, likely $75,000-$100,000 or possibly $50,000-$75,000 given college financial concerns mentioned. Education level shows practical knowledge and business acumen but lacks highly academic language, suggesting 'some college, no degree' or 'associate's degree'—common for small business owners in trades. Male gender evident from business ownership in HVAC (male-dominated field), married to wife, and traditional gender role references. Cultural touchstones and unstated normative perspectives align with white American Southern experience. The faith-based worldview, emphasis on personal responsibility, and traditional family values all strongly predict conservative positions on additional survey questions about social issues, government role, and cultural change.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio explicitly places person in Dallas, Texas suburbs. Survey responses reflect Southern conservative values, emphasis on traditional culture, and regional priorities. References to high school football culture, church-centered community life, and political attitudes align with Southern demographics. Dallas location in North Texas confirms South region.",
        "value": "South",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Bio mentions wife of 28 years and three children (two in college, one in high school). Marriage at approximately 24 years old with children born in mid-to-late 20s/early 30s suggests current age around 50-54. References to being established business owner with decades of church leadership confirm mature adulthood. Life stage indicators (adult children, long-term marriage, business ownership, community leadership roles) all point to 50-64 age bracket.",
        "value": "50-64",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Bio describes HVAC business owner (male-dominated industry with 95%+ male technicians). References to 'my wife' throughout responses. Traditional gender role references and male-coded activities (youth sports coaching, chamber of commerce leadership). Conservative survey responses with traditional family values align with male perspectives in this demographic.",
        "value": "male",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Survey language is clear and practical but lacks highly academic vocabulary or theoretical frameworks. Small business ownership in skilled trade typically requires technical training rather than four-year degree. HVAC field often involves associate's degree, technical certification, or apprenticeship rather than bachelor's. Business acumen evident but educational path likely focused on practical skills. Most likely 'some college, no degree' or 'associate's degree' for technical training.",
        "value": "some college, no degree",
        "confidence": "medium"
    }},
    "MARITAL": {{
        "reasoning": "Bio explicitly states 'wife of 28 years' and survey responses reference 'my wife' and joint family decisions. Long-term stable marriage with three children. Traditional family structure central to identity and values. Clear married status for survey response.",
        "value": "married",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Bio explicitly identifies Baptist church membership with deacon leadership role. Survey responses consistently reference faith-based perspectives and Christian values. Baptist affiliation is specific Protestant denomination. Strong evangelical Christian worldview evident throughout responses. Would identify as Protestant (specifically Baptist) on surveys.",
        "value": "protestant",
        "confidence": "high"
    }},
    "RELIGATTEND": {{
        "reasoning": "Bio describes active deacon role requiring weekly attendance minimum, plus additional church activities mentioned ('evenings often involve church activities'). Deacon leadership position typically requires more than weekly attendance. Faith is central to identity and decision-making. Survey response would be 'once a week' at minimum, more likely 'more than once a week' given leadership role and additional activities.",
        "value": "more than once a week",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Consistent conservative positions across all survey responses. Traditional values, emphasis on limited government, strong national defense, law and order priorities, religious liberty concerns, and opposition to progressive social changes all strongly indicate Republican affiliation. Texas location, evangelical Christian identity, and small business owner status all correlate heavily with Republican party identification. Would definitely identify as Republican in surveys.",
        "value": "Republican",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Survey responses show consistently conservative positions but with pragmatic business-owner sensibility rather than far-right ideology. Traditional social values, limited government preferences, and faith-based worldview indicate 'conservative' ideology. However, practical business concerns and community involvement suggest mainstream conservatism rather than 'very conservative.' Likely identifies as 'conservative' on surveys.",
        "value": "conservative",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "Small business owner with HVAC company (typically profitable in Texas climate). Supporting family of five with two children in college and one approaching college. Four-bedroom house in suburban Dallas. However, mentions of financial concerns about college costs and economic pressures suggest solid middle-class but not wealthy. HVAC business owners typically earn $60,000-$120,000. With college expenses and family size, likely $75,000-$100,000 household income, possibly $50,000-$75,000 depending on business fluctuations.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Cultural references, unstated normative perspective, and social positioning align with white American Southern experience. Small business ownership in suburban Dallas, Baptist church membership, traditional family structure, and conservative political views all correlate with white Southern demographics. No indicators of minority racial identity. High school football culture, chamber of commerce involvement, and community positioning suggest white middle-class suburban identity. Based on statistical likelihood and cultural patterns.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_traits": {{
        "reasoning": "Survey responses demonstrate practical problem-solving, traditional values, community orientation, and strong moral convictions. Bio describes leadership roles, business ownership, and family dedication. Personality profile aligns with conscientious, agreeable, traditional individuals in business and community leadership roles. Lower openness to social change but high conscientiousness in business and family responsibilities.",
        "value": "High conscientiousness, moderate agreeableness, low openness to change - practical, traditional, responsible, community-focused",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states owner of HVAC company. Survey responses show business owner perspective on economic and regulatory issues. Small business concerns evident in attitudes toward taxation, regulation, and government policy. References to managing employees and customer relationships confirm active business ownership in skilled trades sector.",
        "value": "Small business owner (HVAC/skilled trades)",
        "confidence": "high"
    }},
    "lifestyle_category": {{
        "reasoning": "Bio and survey responses reveal traditional family-centered lifestyle with strong church involvement. Married with children, homeowner in suburbs, active in community organizations (church, chamber of commerce, youth sports). Lifestyle centers on family, faith, business, and community rather than urban cultural activities. Traditional middle-class suburban American family structure.",
        "value": "Traditional suburban family, faith-centered, small business owner, community-involved",
        "confidence": "high"
    }},
    "values_priorities": {{
        "reasoning": "Survey responses consistently prioritize faith, family, personal responsibility, economic security, traditional morality, and constitutional principles. Strong emphasis on religious liberty, parental rights, law and order, and limited government. Values reflect conservative Christian worldview combined with small business owner economic concerns. These priorities predict conservative positions on additional survey questions about social policy, government role, and cultural issues.",
        "value": "Faith and family first, personal responsibility, limited government, traditional values, economic freedom, constitutional originalism",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided survey responses and generate the complete user persona profile following this two-step process."""

        elif format == 'newfree_gt':
            system_prompt = """You are an AI assistant specialized in creating simulation-ready personas from survey responses.
These personas will be used to simulate how the same individual would answer NEW survey questions.

PRIMARY OBJECTIVE:
Create a persona that maximizes predictive fidelity for future survey responses across politics, values, morality, institutions, lifestyle choices, and social issues.

KEY PRINCIPLES:
- Think holistically: infer stable traits that drive survey answers (ideology, religiosity, trust, moral reasoning style, risk tolerance, etc.).
- Look for patterns: consistency across items implies stable stances; mixed answers imply conditional rules.
- Build coherence: your persona must form a realistic, internally consistent person.
- Distinguish evidence levels: separate direct evidence from educated inference.
- Avoid fabrication: do not invent vivid biographical specifics unless directly supported.

CRITICAL REQUIREMENTS:
1) Output MUST be valid JSON and MUST contain EXACTLY four top-level keys:
   - "step1_analysis", "short_bio", "step2_analysis", "user_profile"
   Do NOT output any other keys.
2) SECOND-PERSON ONLY + no name:
   - The "short_bio" MUST be written in SECOND-PERSON perspective and refer to the person as "you".
   - Do NOT use any specific name. Address the person directly as "you".
3) "short_bio" is NOT a short biography. It is FREE-FORM persona_text for simulation:
   - No length constraint. Include all details you judge useful for downstream survey simulation.
   - It does NOT have to mention lifestyle/life stage/interests.
   - Focus on predictive traits, stances, conditional rules, attitude strength, and likely answer tendencies.
4) You MUST extract ALL 11 REQUIRED demographic fields:
   - CREGION, AGE, SEX, EDUCATION, MARITAL, RELIG, RELIGATTEND, POLPARTY, POLIDEOLOGY, INCOME, RACE
   - For each field, "value" MUST be selected ONLY from the provided options.
   - Provide "reasoning" BEFORE the value decision and include a "confidence" (high/medium/low).
5) No fabricated specifics:
   - Do NOT invent a job title, city, named organizations, number of children, or life events unless explicitly supported by input.
   - If you include an inference, keep it general and lower confidence accordingly.
6) You may include additional characteristics in "user_profile" beyond the 11 required fields, if they improve simulation fidelity.
   - For any non-required extra fields, "value" may be free-form, but must remain grounded in evidence and avoid fabricated specifics.

IMPORTANT ABOUT ANALYSIS TEXT:
- "step1_analysis" and "step2_analysis" should be evidence-based and helpful.
- Cite specific survey questions (e.g., Q31) when possible.
"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.

<user_data>
<features>
{features}
</features>
<survey_responses>
{survey_responses}
</survey_responses>
</user_data>

Please create a simulation-ready persona profile based on the survey responses provided.
This persona will be used to predict how you (the individual) would respond to additional social surveys.

Your task will be completed in TWO STEPS:

=== STEP 1: Induce Persona Text (Reasoning + Persona) ===

A) In "step1_analysis", analyze the survey responses holistically to understand you as a person whose answers can be simulated.
Focus on:
- Stable value system and worldview
- Political/social attitudes and conditional rules (when answers vary by scenario)
- Moral reasoning style (absolutist vs situational), risk tolerance, deference vs skepticism
- Trust across institutions
- Any life-stage/lifestyle indicators ONLY if supported by evidence (avoid fabrication)
- What you would likely do when uncertain (default patterns)

B) In "short_bio", write the persona_text itself (FREE-FORM, no length constraint):
- SECOND-PERSON perspective ONLY (refer to the person as "you")
- Do NOT use any specific name; address the person directly as "you"
- NOT required to be a biography; can be bullets, paragraphs, or mixed format
- Include all information you judge useful for accurate downstream survey simulation
- Emphasize predictive traits, stance strength, and decision rules
- Avoid invented specifics (no job/city/children counts unless explicitly supported)

=== STEP 2: Extract Required Fields (Reasoning + Structured Profile) ===

In "step2_analysis", explain how the survey responses AND the persona_text ("short_bio") support each inferred characteristic.

In "user_profile", you MUST determine the following 11 REQUIRED fields, selecting values ONLY from the options:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each REQUIRED field in "user_profile", you MUST include:
- "reasoning": evidence-based reasoning FIRST (cite question IDs where possible)
- "value": pick ONLY from the provided options
- "confidence": high/medium/low

You MAY include additional characteristics beyond these 11 required fields (optional),
but keep them grounded and avoid invented specifics. For extra fields, "value" may be free-form.

=== OUTPUT FORMAT (STRICT) ===
Output MUST be valid JSON with EXACTLY these four keys:

{{
  "step1_analysis": "...",
  "short_bio": "...",
  "step2_analysis": "...",
  "user_profile": {{
    "CHARACTERISTIC_NAME": {{
      "reasoning": "...",
      "value": "...",
      "confidence": "high/medium/low"
    }}
  }}
}}

IMPORTANT:
- Output JSON only. No extra text.
- "short_bio" MUST be second-person ("you") and MUST NOT use any specific name.
"""
        elif format == 'newfree':
            system_prompt = """You are an AI assistant specialized in creating simulation-ready personas from survey responses.
These personas will be used to simulate how the same individual would answer NEW survey questions.

PRIMARY OBJECTIVE:
Create a persona that maximizes predictive fidelity for future survey responses across politics, values, morality, institutions, lifestyle choices, and social issues.

KEY PRINCIPLES:
- Think holistically: infer stable traits that drive survey answers (ideology, religiosity, trust, moral reasoning style, risk tolerance, etc.).
- Look for patterns: consistency across items implies stable stances; mixed answers imply conditional rules.
- Build coherence: your persona must form a realistic, internally consistent person.
- Distinguish evidence levels: separate direct evidence from educated inference.
- Avoid fabrication: do not invent vivid biographical specifics unless directly supported.

CRITICAL REQUIREMENTS:
1) Output MUST be valid JSON and MUST contain EXACTLY four top-level keys:
   - "step1_analysis", "short_bio", "step2_analysis", "user_profile"
   Do NOT output any other keys.
2) SECOND-PERSON ONLY + no name:
   - The "short_bio" MUST be written in SECOND-PERSON perspective and refer to the person as "you".
   - Do NOT use any specific name. Address the person directly as "you".
3) "short_bio" is NOT a short biography. It is FREE-FORM persona_text for simulation:
   - No length constraint. Include all details you judge useful for downstream survey simulation.
   - It does NOT have to mention lifestyle/life stage/interests.
   - Focus on predictive traits, stances, conditional rules, attitude strength, and likely answer tendencies.
4) You MUST extract ALL 11 REQUIRED demographic fields:
   - CREGION, AGE, SEX, EDUCATION, MARITAL, RELIG, RELIGATTEND, POLPARTY, POLIDEOLOGY, INCOME, RACE
   - For each field, "value" MUST be selected ONLY from the provided options.
   - Provide "reasoning" BEFORE the value decision and include a "confidence" (high/medium/low).
5) No fabricated specifics:
   - Do NOT invent a job title, city, named organizations, number of children, or life events unless explicitly supported by input.
   - If you include an inference, keep it general and lower confidence accordingly.
6) You may include additional characteristics in "user_profile" beyond the 11 required fields, if they improve simulation fidelity.
   - For any non-required extra fields, "value" may be free-form, but must remain grounded in evidence and avoid fabricated specifics.

IMPORTANT ABOUT ANALYSIS TEXT:
- "step1_analysis" and "step2_analysis" should be evidence-based and helpful.
- Cite specific survey questions (e.g., Q31) when possible.
"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.

<user_data>
<survey_responses>
{survey_responses}
</survey_responses>
</user_data>

Please create a simulation-ready persona profile based on the survey responses provided.
This persona will be used to predict how you (the individual) would respond to additional social surveys.

Your task will be completed in TWO STEPS:

=== STEP 1: Induce Persona Text (Reasoning + Persona) ===

A) In "step1_analysis", analyze the survey responses holistically to understand you as a person whose answers can be simulated.
Focus on:
- Stable value system and worldview
- Political/social attitudes and conditional rules (when answers vary by scenario)
- Moral reasoning style (absolutist vs situational), risk tolerance, deference vs skepticism
- Trust across institutions
- Any life-stage/lifestyle indicators ONLY if supported by evidence (avoid fabrication)
- What you would likely do when uncertain (default patterns)

B) In "short_bio", write the persona_text itself (FREE-FORM, no length constraint):
- SECOND-PERSON perspective ONLY (refer to the person as "you")
- Do NOT use any specific name; address the person directly as "you"
- NOT required to be a biography; can be bullets, paragraphs, or mixed format
- Include all information you judge useful for accurate downstream survey simulation
- Emphasize predictive traits, stance strength, and decision rules
- Avoid invented specifics (no job/city/children counts unless explicitly supported)

=== STEP 2: Extract Required Fields (Reasoning + Structured Profile) ===

In "step2_analysis", explain how the survey responses AND the persona_text ("short_bio") support each inferred characteristic.

In "user_profile", you MUST determine the following 11 REQUIRED fields, selecting values ONLY from the options:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each REQUIRED field in "user_profile", you MUST include:
- "reasoning": evidence-based reasoning FIRST (cite question IDs where possible)
- "value": pick ONLY from the provided options
- "confidence": high/medium/low

You MAY include additional characteristics beyond these 11 required fields (optional),
but keep them grounded and avoid invented specifics. For extra fields, "value" may be free-form.

=== OUTPUT FORMAT (STRICT) ===
Output MUST be valid JSON with EXACTLY these four keys:

{{
  "step1_analysis": "...",
  "short_bio": "...",
  "step2_analysis": "...",
  "user_profile": {{
    "CHARACTERISTIC_NAME": {{
      "reasoning": "...",
      "value": "...",
      "confidence": "high/medium/low"
    }}
  }}
}}

IMPORTANT:
- Output JSON only. No extra text.
- "short_bio" MUST be second-person ("you") and MUST NOT use any specific name.
"""


        elif format == 'free':
            system_prompt = """You are an AI assistant specialized in creating detailed personas from survey responses that will be used to simulate additional survey responses. Your personas must enable accurate simulation of how specific individuals would answer social surveys across demographics, politics, values, lifestyle, and diverse social issues.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to future survey questions. Downstream applications will use your persona to answer surveys on their behalf across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues.

TASK CONTEXT:
You will receive survey responses from a single person. Your task is to analyze these responses to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately.

YOUR APPROACH:
Use a two-step methodology:
1) First, build a holistic biographical understanding by analyzing response patterns, values, worldview, life stage indicators, and priorities revealed in the survey
2) Then, systematically extract specific demographic and psychographic characteristics that would predict future survey responses

KEY PRINCIPLES:
- Think holistically: Use responses to infer how they would answer surveys on related and unrelated topics
- Look for patterns: Consistency in values → political ideology; sophistication in responses → education level; priorities → life stage
- Build coherence: All characteristics should form a realistic, consistent person whose survey responses would align
- Distinguish evidence levels: Clearly separate direct evidence from educated guesses based on demographic patterns
- Predict survey behavior: For each characteristic, consider "How would this inform their answers to other questions?"

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields (region, age, gender, education, marital status, religion, religious attendance, political party, political ideology, income, race)
- Select values ONLY from provided options for each field
- Provide detailed reasoning FIRST before each value determination
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics beyond the required 11
- Create a realistic biographical narrative in second-person perspective"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.
The user's data includes survey responses inside <survey_responses> XML tag:

<user_data>
<survey_responses>
{survey_responses}
</survey_responses>
</user_data>

Please create a detailed user persona profile based on the survey responses provided.

IMPORTANT: This persona will be used to predict how this person would respond to additional social surveys on demographics, politics, values, and lifestyle. Your goal is to understand their real-world characteristics as a whole person based on their expressed attitudes and values.

Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the survey responses holistically to understand this person as a complete individual. Consider:
- Overall value system and worldview
- Political and social attitudes
- Priorities and concerns
- Life circumstances implied by responses
- Cultural and social positioning
- Consistency or contradictions in views

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates inferable characteristics through specific details
- Feels like a realistic description of a complete person
- Uses narrative style with concrete details about their life
- Implicitly reveals demographics through lifestyle, behaviors, and circumstances
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the survey responses AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the responses and the bio
2. Then select your answer ONLY from the provided options listed in brackets
3. Assess confidence level (high/medium/low)
4. If uncertain, make your best educated guess based on available information

ADDITIONALLY, you may identify any other inferable characteristics beyond these 11 required fields.

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:

{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across responses? What do the attitudes, priorities, and values reveal about this person's broader life and identity? What life stage and lifestyle indicators are present?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, and context as a complete individual based on their values and attitudes",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the survey responses and bio together inform each characteristic? How would this person answer additional survey questions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===

<output>
{{
"step1_analysis": "Reviewing the survey responses, I notice a consistent pattern of traditional conservative values combined with strong religious conviction. The respondent prioritizes faith-based perspectives on social issues, emphasizes personal responsibility and limited government, and expresses concern about moral decline in society. Language used is straightforward and practical rather than highly academic, suggesting solid education but not necessarily postgraduate level. References to 'my wife' and concerns about 'raising kids in today's world' indicate married with children. Strong emphasis on community, church involvement, and family values. Concerns about economic security and government spending suggest middle-class household navigating financial pressures. The respondent shows patriotic sentiments and traditional American values, with particular emphasis on law and order, military support, and constitutional originalism. This appears to be a married, religious, middle-aged person in a traditional family structure with conservative political and social views.",

"short_bio": "You are a 52-year-old small business owner running an HVAC company in suburban Dallas, Texas, where you've lived your entire life. You and your wife of 28 years raised three children, two of whom are now in college while your youngest is finishing high school. Sunday mornings find you at the Baptist church where you've been a deacon for over a decade, and you're active in your local chamber of commerce and youth sports leagues. Your days are filled with managing your team of technicians, handling customer calls, and making sure your business stays competitive, while evenings often involve church activities, high school football games, or working on home improvement projects in your four-bedroom house in a family-friendly neighborhood.",

"step2_analysis": "Based on the responses and bio, I can now extract specific characteristics. The consistent traditional conservative positions across multiple issues strongly indicate Republican party affiliation and conservative ideology. References to Baptist church involvement with deacon role suggest Protestant affiliation with regular (weekly or more) attendance. Dallas suburban location places them in the South region. Language about 'wife of 28 years' and 'raising three children' confirms married status. Age references to adult children in college and youngest in high school, combined with 28-year marriage, suggest 50-64 age bracket. Small business owner status with three children in/approaching college suggests solid middle-class income, likely $75,000-$100,000 or possibly $50,000-$75,000 given college financial concerns mentioned. Education level shows practical knowledge and business acumen but lacks highly academic language, suggesting 'some college, no degree' or 'associate's degree'—common for small business owners in trades. Male gender evident from business ownership in HVAC (male-dominated field), married to wife, and traditional gender role references. Cultural touchstones and unstated normative perspectives align with white American Southern experience. The faith-based worldview, emphasis on personal responsibility, and traditional family values all strongly predict conservative positions on additional survey questions about social issues, government role, and cultural change.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio explicitly places person in Dallas, Texas suburbs. Survey responses reflect Southern conservative values, emphasis on traditional culture, and regional priorities. References to high school football culture, church-centered community life, and political attitudes align with Southern demographics. Dallas location in North Texas confirms South region.",
        "value": "South",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Bio mentions wife of 28 years and three children (two in college, one in high school). Marriage at approximately 24 years old with children born in mid-to-late 20s/early 30s suggests current age around 50-54. References to being established business owner with decades of church leadership confirm mature adulthood. Life stage indicators (adult children, long-term marriage, business ownership, community leadership roles) all point to 50-64 age bracket.",
        "value": "50-64",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Bio describes HVAC business owner (male-dominated industry with 95%+ male technicians). References to 'my wife' throughout responses. Traditional gender role references and male-coded activities (youth sports coaching, chamber of commerce leadership). Conservative survey responses with traditional family values align with male perspectives in this demographic.",
        "value": "male",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Survey language is clear and practical but lacks highly academic vocabulary or theoretical frameworks. Small business ownership in skilled trade typically requires technical training rather than four-year degree. HVAC field often involves associate's degree, technical certification, or apprenticeship rather than bachelor's. Business acumen evident but educational path likely focused on practical skills. Most likely 'some college, no degree' or 'associate's degree' for technical training.",
        "value": "some college, no degree",
        "confidence": "medium"
    }},
    "MARITAL": {{
        "reasoning": "Bio explicitly states 'wife of 28 years' and survey responses reference 'my wife' and joint family decisions. Long-term stable marriage with three children. Traditional family structure central to identity and values. Clear married status for survey response.",
        "value": "married",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Bio explicitly identifies Baptist church membership with deacon leadership role. Survey responses consistently reference faith-based perspectives and Christian values. Baptist affiliation is specific Protestant denomination. Strong evangelical Christian worldview evident throughout responses. Would identify as Protestant (specifically Baptist) on surveys.",
        "value": "protestant",
        "confidence": "high"
    }},
    "RELIGATTEND": {{
        "reasoning": "Bio describes active deacon role requiring weekly attendance minimum, plus additional church activities mentioned ('evenings often involve church activities'). Deacon leadership position typically requires more than weekly attendance. Faith is central to identity and decision-making. Survey response would be 'once a week' at minimum, more likely 'more than once a week' given leadership role and additional activities.",
        "value": "more than once a week",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Consistent conservative positions across all survey responses. Traditional values, emphasis on limited government, strong national defense, law and order priorities, religious liberty concerns, and opposition to progressive social changes all strongly indicate Republican affiliation. Texas location, evangelical Christian identity, and small business owner status all correlate heavily with Republican party identification. Would definitely identify as Republican in surveys.",
        "value": "Republican",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Survey responses show consistently conservative positions but with pragmatic business-owner sensibility rather than far-right ideology. Traditional social values, limited government preferences, and faith-based worldview indicate 'conservative' ideology. However, practical business concerns and community involvement suggest mainstream conservatism rather than 'very conservative.' Likely identifies as 'conservative' on surveys.",
        "value": "conservative",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "Small business owner with HVAC company (typically profitable in Texas climate). Supporting family of five with two children in college and one approaching college. Four-bedroom house in suburban Dallas. However, mentions of financial concerns about college costs and economic pressures suggest solid middle-class but not wealthy. HVAC business owners typically earn $60,000-$120,000. With college expenses and family size, likely $75,000-$100,000 household income, possibly $50,000-$75,000 depending on business fluctuations.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Cultural references, unstated normative perspective, and social positioning align with white American Southern experience. Small business ownership in suburban Dallas, Baptist church membership, traditional family structure, and conservative political views all correlate with white Southern demographics. No indicators of minority racial identity. High school football culture, chamber of commerce involvement, and community positioning suggest white middle-class suburban identity. Based on statistical likelihood and cultural patterns.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_traits": {{
        "reasoning": "Survey responses demonstrate practical problem-solving, traditional values, community orientation, and strong moral convictions. Bio describes leadership roles, business ownership, and family dedication. Personality profile aligns with conscientious, agreeable, traditional individuals in business and community leadership roles. Lower openness to social change but high conscientiousness in business and family responsibilities.",
        "value": "High conscientiousness, moderate agreeableness, low openness to change - practical, traditional, responsible, community-focused",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states owner of HVAC company. Survey responses show business owner perspective on economic and regulatory issues. Small business concerns evident in attitudes toward taxation, regulation, and government policy. References to managing employees and customer relationships confirm active business ownership in skilled trades sector.",
        "value": "Small business owner (HVAC/skilled trades)",
        "confidence": "high"
    }},
    "lifestyle_category": {{
        "reasoning": "Bio and survey responses reveal traditional family-centered lifestyle with strong church involvement. Married with children, homeowner in suburbs, active in community organizations (church, chamber of commerce, youth sports). Lifestyle centers on family, faith, business, and community rather than urban cultural activities. Traditional middle-class suburban American family structure.",
        "value": "Traditional suburban family, faith-centered, small business owner, community-involved",
        "confidence": "high"
    }},
    "values_priorities": {{
        "reasoning": "Survey responses consistently prioritize faith, family, personal responsibility, economic security, traditional morality, and constitutional principles. Strong emphasis on religious liberty, parental rights, law and order, and limited government. Values reflect conservative Christian worldview combined with small business owner economic concerns. These priorities predict conservative positions on additional survey questions about social policy, government role, and cultural issues.",
        "value": "Faith and family first, personal responsibility, limited government, traditional values, economic freedom, constitutional originalism",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided survey responses and generate the complete user persona profile following this two-step process."""

        elif format == 'datainfo_opinionqa':
            system_prompt = """You are an AI assistant specialized in creating detailed personas from survey responses that will be used to simulate additional survey responses across diverse topics.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to OpinionQA survey questions across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues.

TASK CONTEXT:
You will receive survey responses from a single person. Your task is to analyze these responses to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately on the OpinionQA dataset.

UNDERSTANDING OPINIONQA'S SCOPE:
OpinionQA consists of real public opinion surveys from Pew Research Center covering 1,498 multiple-choice questions across politics, science, technology, relationships, social issues, economics, health, education, environment, and more. Your persona must be detailed enough to predict responses to sensitive questions across all these domains.

YOUR APPROACH:
Use a two-step methodology with explicit connection to survey prediction:
1) Build a holistic biographical understanding that shows how this person's values and attitudes would predict responses to diverse OpinionQA questions
2) Extract 11 specific demographic characteristics with explicit reasoning about how each predicts responses to sensitive questions

KEY PRINCIPLES:
- Connect to survey prediction: For each demographic characteristic, explain how it predicts responses
- Use demographic research: Leverage known correlations between attitudes and demographics
- Think intersectionally: Consider how multiple characteristics combine
- Build predictive specificity: Show exactly how this profile would answer sensitive questions across domains

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields aligned with OpinionQA groups
- Select values ONLY from provided options for each field
- Provide detailed reasoning FIRST that explicitly references how the characteristic predicts responses
- Explain demographic patterns
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics that enhance prediction
- Create a realistic biographical narrative in second-person perspective"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.
The user's data includes survey responses inside <survey_responses> XML tag:

<user_data>
<survey_responses>
{survey_responses}
</survey_responses>
</user_data>

=== CRITICAL CONTEXT: OpinionQA Survey Prediction ===

IMPORTANT: This persona will be used to predict responses to the OpinionQA dataset, which consists of real public opinion surveys from the Pew Research Center's American Trends Panel (ATP).

OpinionQA Dataset Details:
- 1,498 multiple-choice questions from 15 different surveys
- Topics include: politics, science, technology, personal relationships, social issues, economics, health, education, environment, and more
- Questions reflect real-world attitudes and opinions across diverse subject areas
- Survey responses are broken down by 60 demographic groups

=== EXAMPLE OpinionQA QUESTIONS ===

To give you a concrete sense of what you're predicting, here are sample questions from OpinionQA:

<sample_questions>
Question 1 (Gender Identity & Social Values):
"Which statement comes closer to your views, even if neither is exactly right?
Options:
(A): Whether someone is a man or a woman is determined by the sex they were assigned at birth
(B): Someone can be a man or a woman even if that is different from the sex they were assigned at birth
(C): Refused"

Question 2 (Racial Issues & Social Perception):
"When it comes to racial discrimination, which do you think is the bigger problem for the country today?
Options:
(A): People seeing racial discrimination where it really does not exist
(B): People not seeing racial discrimination where it really does exist
(C): Refused"

Question 3 (Family Structure & Social Norms):
"Which statement comes closer to your view?
Options:
(A): Couples who are married do a better job raising children than couples who are living together but are not married
(B): Couples who are living together but are not married can raise children just as well as married couples
(C): Refused"
</sample_questions>

These examples illustrate that OpinionQA covers sensitive social issues, political topics, and value-based questions where demographic characteristics strongly predict response patterns. Your persona must be detailed enough to predict how this person would answer questions like these across many different domains.

Your Goal:
Create a persona that accurately predicts how this person would answer OpinionQA survey questions across ALL these topics based on their current survey responses. The responses are a window into their broader identity, values, worldview, and demographic characteristics.

=== YOUR TASK ===

Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the survey responses holistically to understand this person as a complete individual who holds opinions across diverse topics. Consider:
- Overall value system and worldview
- Political and social attitudes
- Priorities and concerns
- Life circumstances implied by responses
- Cultural and social positioning
- How their expressed values translate to opinions on various topics

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates inferable characteristics through specific details
- Feels like a realistic description of a complete person
- Implicitly reveals demographics and values that would predict survey responses
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the survey responses AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics relevant to OpinionQA survey prediction.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the responses and the bio
2. Consider how this demographic characteristic correlates with opinions on diverse OpinionQA topics
3. Then select your answer ONLY from the provided options listed in brackets
4. Assess confidence level (high/medium/low)

ADDITIONALLY, you may identify any other inferable characteristics that would help predict OpinionQA responses.

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:

{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across responses? What do the attitudes, priorities, and values reveal about this person's broader life and identity? What life stage and lifestyle indicators are present? How would these characteristics predict their opinions on diverse OpinionQA topics (politics, science, social issues, etc.)? How would they likely answer the sample questions shown above?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, values, and context as a complete individual who holds opinions on many topics",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the survey responses and bio together inform each characteristic? How would this person answer OpinionQA survey questions on diverse topics, including the sample questions about gender identity, racial discrimination, and family structures? What demographic patterns would predict their opinion distributions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST, including how this characteristic would influence OpinionQA survey responses (reference sample questions where relevant)",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===

<output>
{{
"step1_analysis": "Reviewing the survey responses, I notice a consistent pattern of moderate-to-progressive views with strong emphasis on pragmatic solutions and evidence-based policy. The respondent shows support for scientific research, environmental protection, and healthcare access, while expressing concerns about government efficiency and fiscal responsibility. Language is articulate and sophisticated, suggesting higher education, likely in a professional field. References to work-life balance concerns and healthcare costs suggest middle-age professional with family responsibilities. The respondent shows socially liberal positions (supporting LGBTQ+ equality, racial justice awareness) combined with economically moderate views (concerned about spending, favoring targeted programs over universal approaches). This would predict option B on the gender identity question (supporting trans rights based on progressive social values), option B on racial discrimination (recognizing systemic issues while wanting practical solutions), and likely option B on family structures (pragmatic view that outcomes matter more than form). Values scientific consensus on climate change and public health, predicting support for evidence-based environmental and health policies in OpinionQA. Concerns about education quality and college affordability suggest parent with children approaching or in higher education. The respondent demonstrates nuanced thinking, avoiding extreme positions while maintaining clear values - characteristic of educated professionals in suburban/urban areas. This profile suggests a college-educated, middle-aged professional in a dual-income household who would answer OpinionQA questions with center-left positions on social issues and moderate positions on economic policy.",

"short_bio": "You are a 47-year-old software engineer living in a diverse suburb of Atlanta, Georgia, with your wife and two teenage children. You hold a bachelor's degree in computer science and work for a mid-sized tech company, where you've navigated the shift to remote work while balancing family responsibilities. Your weekends involve driving your daughter to soccer tournaments, helping your son with college applications, and occasionally hiking on nearby trails when you can find the time. You're not particularly religious but appreciate the community aspects of the local Unitarian church you attend a few times a year, and you stay informed about current events through a mix of mainstream news sources and tech industry podcasts. You and your wife worry about college costs, healthcare expenses, and retirement savings, but your household income provides a comfortable middle-class lifestyle that lets you focus on your kids' futures and occasionally indulge in your hobby of building smart home systems.",

"step2_analysis": "Based on the survey responses and bio, I can now extract specific characteristics that would predict OpinionQA responses. The consistent support for science-based policy, environmental action, and social equality combined with fiscal concerns indicates moderate-to-liberal ideology, likely Independent or Democrat party affiliation but not strongly partisan. The tech industry professional background with bachelor's degree suggests 'college graduate/some postgrad' education level - college-educated respondents in OpinionQA show strong support for science, evidence-based policy, and are more likely to support trans rights (option B on gender identity), recognize systemic racism (option B on racial discrimination), and take pragmatic views on family structures (option B). Atlanta suburban location places them in the South region, but suburban educated professionals in diverse Southern metros show different patterns than rural South - more socially progressive while economically moderate. Age 47 with teenage children places them solidly in 30-49 bracket (or possibly early 50-64) - this age group balances established values with openness to social change, particularly on LGBTQ+ and racial justice issues. Male gender in tech profession evident from career field and problem-solving orientation. Married status with children confirmed by family references. The Unitarian church attendance 'a few times a year' suggests 'nothing in particular' religious affiliation with 'seldom' or 'a few times a year' attendance - this secular-leaning profile predicts progressive positions on social issues including all three sample questions. Software engineer salary in Atlanta with two-income household supporting college-bound children suggests '$100,000 or more' household income - this income level shows support for social programs but sensitivity to personal tax burden. The moderate political positions combined with social progressivism suggest Independent party affiliation or moderate Democrat, with 'moderate' or 'liberal' ideology. Cultural references suggest white suburban professional demographic. Overall, this profile predicts center-left responses on social issues (option B on all three sample questions due to education, secular values, and suburban diversity exposure), strong support for science and climate policy, moderate economic positions, and pragmatic approaches to healthcare and education policy in OpinionQA.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio places person in Atlanta, Georgia suburbs. While Georgia is in the South region, survey responses show urban/suburban professional patterns rather than rural Southern conservatism. Atlanta's diverse metropolitan area creates different political and social dynamics than traditional South. However, for demographic classification, Atlanta is in the South region. Southern suburban educated professionals in OpinionQA show mixed patterns - more progressive than rural South but more moderate than Northeast/West urban areas. This Southern suburban location combined with progressive social values would still predict option B on gender identity and racial discrimination questions (recognizing diversity in metro area), and option B on family structures (pragmatic suburban professional view).",
        "value": "South",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Bio specifies 47 years old with teenage children (daughter in sports, son applying to college). This places them in late 40s, technically could be 30-49 or 50-64 bracket. Given college application timing (son likely 17-18), parent likely had children in late 20s/early 30s, confirming age around 45-49. Survey responses show established career, family responsibilities, and balanced perspectives characteristic of middle age. This age group in OpinionQA shows pragmatic progressive views on social issues - established enough to have clear values but young enough to support social change. Would support option B on gender identity (Gen X openness), option B on racial discrimination (awareness from diverse workplace), and option B on family structures (pragmatic parenting perspective).",
        "value": "30-49",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Bio describes software engineer career in male-dominated field (approximately 75% male), references to wife indicate male respondent. Survey response patterns show tech industry problem-solving orientation and focus on efficiency/evidence. Male educated professionals in OpinionQA show support for evidence-based policy and science, with educated men in diverse urban areas supporting progressive social positions. College-educated men in tech fields are significantly more likely than average men to support trans rights (option B on gender identity), recognize systemic racism when exposed to diverse workplaces (option B on racial discrimination), and take pragmatic views on family structures (option B).",
        "value": "male",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Bio explicitly states bachelor's degree in computer science. Software engineering profession requires four-year degree minimum. Survey language shows analytical sophistication and evidence-based reasoning characteristic of STEM education. College graduates in OpinionQA show strong support for science funding, evidence-based policy, environmental action, and nuanced views on complex issues. Bachelor's degree holders (without postgraduate) show progressive social views but more moderate economic positions than postgraduate degree holders. This education level predicts support for option B on gender identity (college education correlates with acceptance of diverse identities), option B on racial discrimination (STEM education emphasizes systematic analysis of problems), and option B on family structures (education correlates with pragmatic rather than ideological views).",
        "value": "college graduate/some postgrad",
        "confidence": "high"
    }},
    "MARITAL": {{
        "reasoning": "Bio explicitly references 'your wife' and survey responses mention family decisions and household planning. Long-term stable marriage with two teenage children indicates married status. Traditional family structure combined with progressive social values creates interesting OpinionQA profile - married respondents typically more conservative, but educated suburban married professionals show moderate-to-progressive views. Married status with education and suburban location would still predict option B on family structure question (pragmatic view that parenting quality matters more than marital status) while recognizing value of their own marriage.",
        "value": "married",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Bio mentions attending Unitarian church 'a few times a year' for 'community aspects' rather than theological reasons. Unitarian Universalism is known for theological diversity and social progressivism rather than traditional doctrine. Survey responses show secular reasoning and evidence-based worldview without religious framework. This suggests 'nothing in particular' religious affiliation (attending for community, not belief) rather than agnostic (which implies more active questioning of religion). 'Nothing in particular' respondents in OpinionQA show moderate-to-progressive social views, strong support for science and secular policy, and would strongly support option B on gender identity, option B on racial discrimination, and option B on family structures.",
        "value": "nothing in particular",
        "confidence": "medium"
    }},
    "RELIGATTEND": {{
        "reasoning": "Bio explicitly states attending Unitarian church 'a few times a year' for community rather than religious conviction. This perfectly matches 'a few times a year' attendance category. Occasional attenders in OpinionQA show moderate-to-progressive views, less religiously motivated policy preferences than regular attenders, but not as secular as 'never' attenders. This attendance level combined with education and secular reasoning predicts option B on all three sample questions, as occasional attendance for community doesn't correlate with traditional religious positions on social issues.",
        "value": "a few times a year",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Survey responses show mix of progressive social values (LGBTQ+ equality, racial justice, climate action) with moderate economic concerns (government efficiency, fiscal responsibility). This pattern is characteristic of Independent voters or moderate Democrats rather than strong partisans. Tech industry professionals often identify as Independent due to libertarian-leaning economic views combined with social progressivism. However, consistent support for evidence-based policy, environmental protection, and social equality suggests Democrat-leaning Independent or moderate Democrat. In current polarized environment, educated suburban professionals with these views typically identify as Democrat even if moderate. Would predict option B on all three sample questions (Democratic positions), though with pragmatic rather than ideological reasoning.",
        "value": "Democrat",
        "confidence": "medium"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Survey responses show socially progressive positions (supporting LGBTQ+ rights, recognizing racial justice issues, supporting climate action) combined with economically moderate views (fiscal concerns, efficiency focus, targeted rather than universal programs). This pattern indicates 'moderate' ideology - left of center on social issues, center on economic issues. Not 'liberal' because of economic moderation and pragmatic rather than ideological approach. Not 'conservative' because of progressive social positions. Moderates in OpinionQA show diverse views but educated suburban moderates typically support option B on social issue questions (gender identity, racial discrimination, family structures) while having mixed views on economic policy.",
        "value": "moderate",
        "confidence": "medium"
    }},
    "INCOME": {{
        "reasoning": "Software engineer salary in Atlanta (typically $90,000-$140,000) combined with dual-income household (bio mentions 'your wife' working, household decisions). Supporting two children including college applications, comfortable lifestyle, homeownership in Atlanta suburbs, ability to save for retirement despite college costs. Tech professional household income in Atlanta suburbs typically $120,000-$180,000. Bio explicitly mentions comfortable middle-class lifestyle that allows focus on children's futures. This clearly indicates '$100,000 or more' household income. This income level in OpinionQA shows support for social programs and environmental protection (can afford costs) while having concerns about taxation and government efficiency. High-income educated professionals consistently support option B on all three sample questions (progressive social views independent of economic self-interest).",
        "value": "$100,000 or more",
        "confidence": "high"
    }},
    "RACE": {{
        "reasoning": "Cultural references, unstated normative perspective, and social positioning align with white American suburban professional experience. Software engineering field is predominantly white (approximately 65-70%), Atlanta suburban demographics, and Unitarian church attendance patterns all suggest white identity. No indicators of minority racial identity or perspective shaped by racial discrimination. However, survey responses show awareness of racial justice issues and support for diversity - characteristic of white educated professionals in diverse urban/suburban areas. White educated moderates in OpinionQA show support for racial justice initiatives (would choose option B on racial discrimination question), strong support for LGBTQ+ rights (option B on gender identity), and progressive views on family structures (option B), though with different lived experience than racial minorities.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_traits": {{
        "reasoning": "Survey responses demonstrate analytical thinking, evidence-based reasoning, and systematic problem-solving characteristic of STEM professionals. Bio describes balanced life management, family focus, and practical hobbies (smart home systems). Personality profile suggests high conscientiousness (planning for college, career stability, organized life), moderate openness (progressive social views but pragmatic approach), and moderate agreeableness (values community, balanced views). This personality type in OpinionQA predicts support for evidence-based policy, systematic solutions to social problems, and pragmatic progressive positions. High conscientiousness with moderate openness predicts choosing option B on sample questions based on evidence and fairness rather than ideology.",
        "value": "High conscientiousness, moderate openness, moderate agreeableness - analytical, evidence-driven, pragmatic, family-focused",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states software engineer at mid-sized tech company. Survey responses show tech industry perspective: emphasis on efficiency, evidence-based solutions, innovation-friendly but concerned about implementation. Tech professionals in OpinionQA show strong support for science funding, innovation policy, STEM education, and evidence-based environmental policy. They tend to have libertarian-leaning economic views but progressive social views. Software engineers specifically show high support for option B on gender identity (industry diversity initiatives), option B on racial discrimination (awareness from diversity discussions in tech), and option B on family structures (pragmatic, data-driven view of outcomes).",
        "value": "Technology (software engineering)",
        "confidence": "high"
    }},
    "lifestyle_category": {{
        "reasoning": "Bio and survey responses reveal suburban family-centered lifestyle balanced with professional career. Dual-income professional household with teenage children, involved in children's activities (soccer, college prep), homeowner, occasional outdoor recreation. Lifestyle focused on family advancement, career stability, and preparing children for success. This represents mainstream educated suburban American family. Suburban educated families in OpinionQA show moderate-to-progressive views, strong support for education funding and healthcare access, environmental concern balanced with economic pragmatism. This lifestyle predicts option B on all three sample questions - suburban diversity exposure and education predict progressive social views despite traditional family structure.",
        "value": "Suburban professional family, dual-income, education-focused, technology-oriented",
        "confidence": "high"
    }},
    "values_priorities": {{
        "reasoning": "Survey responses consistently prioritize evidence-based policy, scientific reasoning, practical solutions, family wellbeing, education quality, healthcare access, and environmental sustainability. Shows social progressivism (equality, justice, inclusion) balanced with fiscal responsibility and efficiency concerns. Values reflect educated suburban professional worldview: pragmatic progressivism focused on outcomes rather than ideology. These priorities predict moderate-to-progressive positions across OpinionQA topics, with option B on social issue questions (guided by evidence and fairness), strong support for science and education funding, environmental protection, healthcare access, and moderate positions on economic policy.",
        "value": "Evidence-based policy, family wellbeing, education quality, social equality, environmental sustainability, fiscal responsibility, pragmatic solutions",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided survey responses and generate the complete user persona profile following this two-step process. Remember: this persona will be used to predict OpinionQA survey responses across diverse topics (politics, science, social issues, health, technology, etc.), including sensitive questions like the samples shown above about gender identity, racial discrimination, and family structures. Use the survey responses to understand the complete person and how they would answer additional survey questions on many different subjects."""


        elif format == 'datainfo_mp3dr':
            system_prompt = """You are an AI assistant specialized in creating detailed personas from survey responses that will be used to simulate additional survey responses across diverse topics.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to TESS experimental survey questions across topics including: political preferences, social values, policy positions, demographic questions, lifestyle choices, moral foundations, and attitudes toward contemporary issues.

TASK CONTEXT:
You will receive survey responses from a single person from a TESS study. Your task is to analyze these responses to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately on other TESS experimental studies.

UNDERSTANDING TESS'S SCOPE:
TESS (Time-sharing Experiments for the Social Sciences) supports general population experiments with nationally representative samples. TESS uses experimental designs with random assignment to test causal hypotheses across all social science disciplines. Your persona must be detailed enough to predict responses to experimental questions across all these domains.

YOUR APPROACH:
Use a two-step methodology with explicit connection to experimental survey prediction:
1) Build a holistic biographical understanding that shows how this person's values and attitudes would predict responses to diverse TESS experimental questions
2) Extract 11 specific demographic characteristics with explicit reasoning about how each predicts responses to experimental questions

KEY PRINCIPLES:
- Connect to experimental survey prediction: For each demographic characteristic, explain how it predicts responses to experimental manipulations
- Use demographic research: Leverage known correlations between attitudes and demographics
- Think intersectionally: Consider how multiple characteristics combine
- Build predictive specificity: Show exactly how this profile would answer experimental questions across domains

CRITICAL REQUIREMENTS:
- Extract ALL 11 required demographic fields aligned with TESS demographic groups
- Select values ONLY from provided options for each field
- Provide detailed reasoning FIRST that explicitly references how the characteristic predicts responses
- Explain demographic patterns in experimental contexts
- Assess confidence levels honestly (high/medium/low)
- May identify additional characteristics that enhance prediction
- Create a realistic biographical narrative in second-person perspective"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.
The user's data includes survey responses inside <survey_responses> XML tag:

<user_data>
<survey_responses>
{survey_responses}
</survey_responses>
</user_data>

=== CRITICAL CONTEXT: TESS Survey Prediction ===

IMPORTANT: This persona will be used to predict responses to other TESS (Time-sharing Experiments for the Social Sciences) studies, which consist of experimental surveys with nationally representative samples.

TESS Dataset Details:
- Since 2001, TESS has supported general population experiments for social science researchers
- Uses NORC's AmeriSpeak Panel - a nationally representative, probability-based panel
- Experimental designs with random assignment to test causal hypotheses
- Topics span all social sciences: politics, economics, psychology, sociology, public health, law, etc.
- Questions involve experimental manipulations (varying stimuli, information, framing) to measure causal effects
- Samples are representative of the U.S. general population with diverse demographic groups

=== EXAMPLE TESS QUESTIONS ===

To give you a concrete sense of what you're predicting, here are sample questions from a TESS study on gender discrimination:

<sample_questions>
Question 1 (Gender Inequality Policies):
"In order to minimize gender discrimination, some companies implement policies to address gender inequality in the workplace. What is your opinion of this practice?
Response scale: 1 (strongly favor) to 7 (strongly oppose)"

Question 2 (Active Recruitment of Women):
"In order to minimize gender discrimination, some companies actively recruit women to apply for job openings. What is your opinion of this practice?
Response scale: 1 (strongly favor) to 7 (strongly oppose)"

Question 3 (Diversity Training):
"In order to minimize gender discrimination, some companies offer voluntary diversity training to their employees. What is your opinion of this practice?
Response scale: 1 (strongly favor) to 7 (strongly oppose)"
</sample_questions>

These examples illustrate that TESS covers policy attitudes, workplace practices, and social interventions where demographic characteristics, political orientation, and values strongly predict response patterns. Your persona must be detailed enough to predict how this person would answer experimental questions like these across many different TESS studies.

Your Goal:
Create a persona that accurately predicts how this person would answer TESS experimental questions across ALL social science topics based on their current survey responses. The responses are a window into their broader identity, values, worldview, and demographic characteristics.

=== YOUR TASK ===

Your task will be completed in TWO STEPS:

=== STEP 1: Generate General Profile and Biography ===

First, analyze the survey responses holistically to understand this person as a complete individual who holds opinions across diverse topics. Consider:
- Overall value system and worldview
- Political and social attitudes
- Priorities and concerns
- Life circumstances implied by responses
- Cultural and social positioning
- How their expressed values translate to opinions on various experimental topics

Think about how they would respond to experimental manipulations in TESS studies on topics like:
- Workplace diversity and gender equality policies (see sample questions above)
- Social interventions and policy reforms
- Economic policies and labor market issues
- Political reforms and governance experiments
- Scientific topics and technology attitudes
- Health policies and public health interventions
- Educational policies and social programs

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates inferable characteristics through specific details
- Feels like a realistic description of a complete person
- Implicitly reveals demographics and values that would predict experimental survey responses
- Does NOT use a specific name - address the person directly as "you"

=== STEP 2: Extract Specific Demographic and Psychographic Characteristics ===

Based on BOTH the survey responses AND the short_bio you created, now systematically identify specific demographic and psychographic characteristics relevant to TESS experimental survey prediction.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from BOTH the responses and the bio
2. Consider how this demographic characteristic correlates with opinions on diverse TESS experimental topics (including the sample questions shown above)
3. Then select your answer ONLY from the provided options listed in brackets
4. Assess confidence level (high/medium/low)
5. Think: How would someone with this demographic profile answer experimental survey questions about gender discrimination policies, and other social interventions tested in TESS studies?

ADDITIONALLY, you may identify any other inferable characteristics that would help predict TESS responses.

=== OUTPUT FORMAT ===

Provide your output in JSON format with four keys:

{{
"step1_analysis": "Your detailed thinking process for Step 1: What patterns do you see across responses? What do the attitudes, priorities, and values reveal about this person's broader life and identity? What life stage and lifestyle indicators are present? How would these characteristics predict their opinions on diverse TESS experimental topics (workplace policies, social interventions, policy experiments, etc.)? How would they likely answer the sample questions shown above on 1-7 scales?",

"short_bio": "A 4-5 sentence narrative description in SECOND-PERSON (using 'you') describing their life, lifestyle, interests, values, and context as a complete individual who holds opinions on many topics",

"step2_analysis": "Your detailed thinking process for Step 2: Now that you have a holistic view of this person, what specific demographic and psychographic characteristics can you identify? How do the survey responses and bio together inform each characteristic? How would this person answer TESS experimental survey questions on diverse topics, including the sample questions about gender discrimination policies (on 1-7 scales)? What demographic patterns would predict their response distributions in experimental contexts?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning FIRST, including how this characteristic would influence TESS experimental survey responses (reference sample questions where relevant and predict specific responses on scales)",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===

<output>
{{
"step1_analysis": "Reviewing the survey responses from the TESS study, I notice a consistent pattern of moderate-to-progressive views with strong emphasis on pragmatic solutions and evidence-based policy. The respondent shows support for social interventions, environmental protection, and healthcare access, while expressing concerns about implementation effectiveness. Language is articulate and sophisticated, suggesting higher education, likely in a professional field. The respondent shows socially liberal positions (supporting equality initiatives, recognizing systemic issues) combined with pragmatic concerns about policy design. This would predict responses of 1-2 (strongly favor/favor) on all three gender discrimination policy questions - implementing workplace equality policies (1-2), actively recruiting women (1-2), and diversity training (1-2) - based on their support for evidence-based interventions to address inequality. Values systematic approaches to social problems, predicting support for experimental interventions in TESS studies on health, education, and workplace policies. The measured, thoughtful approach to complex issues is characteristic of educated professionals in suburban/urban areas. This profile suggests a college-educated, middle-aged professional who would answer TESS experimental questions with center-left positions on social interventions and support for evidence-based policy reforms.",

"short_bio": "You are a 47-year-old public health researcher living in a diverse suburb of Seattle, Washington, with your partner. You hold a master's degree in public health and work for a state health department, where you design and evaluate community health interventions. Your weekends involve volunteering with local health equity organizations, attending lectures at the university, and hiking in the Cascades. You're not particularly religious but value evidence-based approaches to social problems, and you stay informed about current events through academic journals and policy-focused podcasts. Your work focuses on addressing health disparities, and your professional experience has shaped your views on the importance of systematic interventions to address inequality.",

"step2_analysis": "Based on the survey responses and bio, I can now extract specific characteristics that would predict TESS experimental responses. The consistent support for evidence-based interventions, systematic approaches to inequality, and social justice indicates liberal ideology and Democrat party affiliation. The public health professional background with master's degree suggests 'postgraduate' education level - highly educated respondents in TESS experiments show strongest support for diversity initiatives and would respond 1-2 (strongly favor/favor) on all three gender discrimination policy questions. Seattle suburban location places them in the West region - West Coast educated professionals show strong support for workplace diversity policies and social interventions. Age 47 places them in 30-49 bracket - this age group shows established progressive values on social equality. The 'partner' language suggests 'living with a partner' - non-traditional family structures correlate with progressive social attitudes. Professional focus on health equity strongly indicates support for systematic interventions to address discrimination, predicting responses of 1 (strongly favor) on implementing workplace equality policies, 1-2 on actively recruiting women, and 1 (strongly favor) on diversity training. The secular values and lack of religious references suggest 'agnostic' or 'nothing in particular' with minimal attendance - secular respondents in TESS studies strongly support evidence-based interventions. Public health researcher salary in Seattle suggests '$75,000-$100,000' or higher household income. Overall, this profile predicts consistently progressive responses (1-2 range on 7-point scales) supporting social interventions, workplace diversity policies, and evidence-based approaches to inequality in TESS experimental studies.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio places person in Seattle, Washington suburbs. Professional public health work and progressive social values align with West Coast urban demographics. West region respondents in TESS experiments, especially educated professionals in progressive urban areas, show highest support for workplace diversity policies and social interventions. On the sample questions, West Coast professionals would likely respond 1-2 (strongly favor/favor) on implementing equality policies, 1-2 on actively recruiting women, and 1-2 on diversity training, as this region shows strongest support for systematic approaches to discrimination.",
        "value": "West",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Bio specifies 47 years old. Survey responses show established career, professional expertise, and mature perspective characteristic of middle adulthood in 30-49 bracket. This age group in TESS experiments typically shows pragmatic progressive views on workplace policies - established professional identity but supportive of social change and evidence-based interventions. Would likely respond 1-2 on all three gender discrimination policy questions, as this cohort combines professional experience with openness to systematic interventions addressing inequality.",
        "value": "30-49",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Bio uses 'partner' without gender specification, and public health field is approximately 75% female. Survey response patterns show strong emphasis on health equity, social justice, and systematic approaches to inequality - perspectives particularly common among women in health professions. Female public health professionals in TESS experiments show particularly strong support for workplace diversity policies and interventions addressing gender discrimination. Would respond 1 (strongly favor) on implementing workplace equality policies, 1 (strongly favor) on actively recruiting women, and 1 (strongly favor) on diversity training, as women in professional fields strongly support measures to address gender inequality.",
        "value": "female",
        "confidence": "medium"
    }},
    "EDUCATION": {{
        "reasoning": "Bio explicitly states master's degree in public health. Public health researcher position requires graduate education. Survey language shows graduate-level analytical skills and research orientation. Postgraduate-educated respondents in TESS experiments show highest support for evidence-based interventions and systematic approaches to social problems. This education level is the strongest predictor of responding 1-2 (strongly favor/favor) on all three gender discrimination policy questions, as graduate education in social sciences correlates with understanding of systemic inequality and support for institutional interventions.",
        "value": "postgraduate",
        "confidence": "high"
    }},
    "MARITAL": {{
        "reasoning": "Bio mentions 'partner' rather than 'spouse' or 'husband/wife,' suggesting unmarried cohabitation. Survey responses show progressive social values. For survey purposes, 'living with a partner' is most accurate response. Cohabiting couples in TESS experiments tend to hold more liberal views on social interventions and workplace policies. This status would predict responding 1-2 on all three gender discrimination policy questions, as non-traditional relationship structures correlate with support for progressive workplace policies and challenging traditional norms.",
        "value": "living with a partner",
        "confidence": "medium"
    }},
    "RELIG": {{
        "reasoning": "Survey responses show secular, evidence-based reasoning with no religious framework. Bio states 'not particularly religious,' suggesting lack of religious affiliation. Public health professional worldview emphasizes scientific evidence over faith-based perspectives. This suggests 'agnostic' or 'nothing in particular' religious affiliation. Secular respondents in TESS experiments strongly support evidence-based workplace policies. This religious affiliation predicts responding 1-2 (strongly favor/favor) on all three gender discrimination policy questions, as secular individuals support systematic, evidence-based interventions over traditional approaches.",
        "value": "agnostic",
        "confidence": "medium"
    }},
    "RELIGATTEND": {{
        "reasoning": "Bio states 'not particularly religious' with no references to religious practices or communities. Lifestyle centered on professional work, volunteering, and secular activities. As someone without religious framework, survey response would likely be 'never' or 'seldom.' Non-attenders in TESS experiments show strongest support for secular workplace policies and evidence-based interventions. This characteristic strongly predicts responding 1 (strongly favor) on all three gender discrimination policy questions, as religious non-attendance correlates with strongest support for progressive workplace policies.",
        "value": "never",
        "confidence": "medium"
    }},
    "POLPARTY": {{
        "reasoning": "Survey responses show consistent progressive positions on social interventions, health equity work, Seattle residence in progressive environment. Public health professionals focused on health disparities typically identify as Democrats. Would definitely identify as Democrat in surveys. Democrats in TESS experiments show strong support for workplace diversity policies, systematic interventions, and anti-discrimination measures. Democratic party affiliation is highly predictive of responding 1-2 (strongly favor/favor) on all three gender discrimination policy questions: implementing workplace equality policies (1), actively recruiting women (1-2), and diversity training (1).",
        "value": "Democrat",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Survey responses show consistently progressive values with professional focus on health equity and systematic interventions to address disparities. Public health professionals working on health equity typically identify as liberal or very liberal. Evidence-based, systematic approach suggests 'liberal' rather than 'very liberal' due to pragmatic professional orientation. Liberals in TESS experiments consistently support evidence-based interventions and respond 1-2 (strongly favor/favor) on all three gender discrimination policy questions: implementing equality policies (1-2), recruiting women (1-2), and diversity training (1-2).",
        "value": "liberal",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "Master's degree holder in public health research position at state health department. Seattle area has high cost of living. Public health researchers in government positions typically earn $70,000-$100,000. Dual income household with partner. Comfortable lifestyle with volunteering time suggests '$75,000-$100,000' or possibly '$100,000 or more' household income. This income bracket with high education in TESS experiments shows strong support for workplace diversity policies - professional class can prioritize social values. Would predict responses of 1-2 (strongly favor/favor) on all three gender discrimination policy questions.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Bio describes working on health disparities in diverse Seattle suburb. Public health field addressing health equity attracts professionals across racial backgrounds. Without explicit indicators, most conservative estimate based on U.S. demographics and professional field distribution suggests white, but confidence is lower. White progressive professionals in TESS experiments show strong support for diversity initiatives and anti-discrimination measures. Would respond 1-2 (strongly favor/favor) on all three gender discrimination policy questions regardless of race, given progressive professional identity.",
        "value": "white",
        "confidence": "low"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states public health researcher at state health department focused on health equity. Survey responses show systematic thinking about social interventions and evidence-based policy. Public health professionals in TESS experiments show particularly strong support for workplace diversity policies, as the field emphasizes addressing disparities and systemic inequality. Would respond 1 (strongly favor) on implementing workplace equality policies, 1 (strongly favor) on actively recruiting women, and 1 (strongly favor) on diversity training, viewing these as evidence-based interventions to address systemic discrimination.",
        "value": "Public Health (researcher/health equity specialist)",
        "confidence": "high"
    }},
    "values_priorities": {{
        "reasoning": "Survey responses consistently prioritize evidence-based interventions, health equity, systematic approaches to inequality, and social justice. Professional work focuses on addressing health disparities through community interventions. These values predict strong support for workplace diversity policies and systematic interventions in TESS experiments. Would respond 1 (strongly favor) on all three gender discrimination policy questions, viewing workplace equality policies, active recruitment, and diversity training as complementary evidence-based strategies to address systemic gender discrimination.",
        "value": "Health equity, evidence-based interventions, social justice, systematic approaches to inequality, community wellbeing",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided survey responses and generate the complete user persona profile following this two-step process. Remember: this persona will be used to predict TESS experimental survey responses across diverse topics (workplace policies, social interventions, policy experiments, health interventions, etc.), including experimental questions like the samples shown above about gender discrimination policies (on 1-7 scales). Use the survey responses to understand the complete person and how they would answer additional experimental survey questions on many different subjects."""


        elif format == 'rubric':
            system_prompt = """You are an AI assistant specialized in creating detailed personas from survey responses that will be used to simulate additional survey responses across diverse topics.

PRIMARY OBJECTIVE:
Create a comprehensive persona profile that can be used to simulate this person's responses to social survey questions across demographics, politics, values, lifestyle, and diverse social issues.

TASK CONTEXT:
You will receive survey responses from a single person. Your task is to analyze these responses to build a complete persona profile that captures the psychological, demographic, and value-based characteristics needed to simulate their survey responses accurately.

THE THREE-STEP METHODOLOGY:
STEP 1 - Personality & Values Assessment:
- Predict answers to 13 specific personality and value questions based on survey response patterns
- Include Big Five personality traits operationalized through specific questions
- Plus values orientation (competitiveness, self-reliance, impulsivity)
- These predictions become foundational data for predicting all subsequent characteristics

STEP 2 - Biographical Integration:
- Use survey responses + personality/values predictions to build holistic understanding
- Create biographical narrative that shows how personality traits manifest in complete life

STEP 3 - Demographic & Characteristic Extraction:
- Extract 11 required demographics plus additional characteristics
- Explicitly show how personality + demographics combine to predict survey responses

WHY PERSONALITY ASSESSMENT MATTERS:
Personality traits are powerful cross-domain predictors that help simulate responses to diverse survey questions across all topics.

YOUR APPROACH:
1) STEP 1: Systematically predict 13 personality/values answers with reasoning FIRST, then answer choice
2) STEP 2: Integrate personality profile with responses to create biographical narrative
3) STEP 3: Extract demographics with explicit reasoning about how personality + demographics predict survey responses

CRITICAL REQUIREMENTS:
- Predict ALL 13 personality/values items with reasoning before answer
- Extract ALL 11 required demographic fields with values from provided options
- Provide detailed reasoning showing how personality + demographics predict survey responses
- Assess confidence levels honestly (high/medium/low)
- Create realistic biographical narrative in second-person perspective that integrates personality traits"""

            prompt = """The user's data is provided within <user_data></user_data> XML tags.
The user's data includes survey responses inside <survey_responses> XML tag:

<user_data>
<survey_responses>
{survey_responses}
</survey_responses>
</user_data>

=== YOUR GOAL ===

Please create a detailed user persona profile based on the survey responses provided.

IMPORTANT: This persona will be used to predict how this person would respond to additional social surveys on demographics, politics, values, and lifestyle. Use the responses to understand their real-world characteristics as a complete person.

=== YOUR TASK ===

Your task will be completed in THREE STEPS:

=== STEP 1: Predict Personality Traits and Values ===

First, predict how this person would answer the following general personality and values questions based on their survey responses. These predictions will inform your subsequent analysis.

<general_questions>

Q1. PERSONALITY TRAITS
"Everyone has different personality traits that make them unique. Below are pairs of words that might describe you. For each pair, tell us how much you agree or disagree about the word pairs. Think about the pairs as a unit, rather than individual words. I see myself as…"

Answer options for each item:
- Strongly disagree
- Moderately disagree
- Disagree a little
- Neither agree nor disagree
- Agree a little
- Moderately agree
- Strongly agree

1.1 Direct, questioning
1.2 Anxious, easily upset
1.3 Open to new experiences, complex
1.4 Reserved, quiet
1.5 Sympathetic, warm
1.6 Flexible, spontaneous
1.7 Calm, emotionally stable
1.8 Traditional, practical

Q2. PERSONAL VALUES
"We all have different things that are important to us in life. Below is a list of values that people often care about. For each one, think about how important it is to you personally as a guiding principle in your life. How much do each of these values matters to you when you think about what's important in life?"

Answer options for each item:
- Strongly disagree
- Moderately disagree
- Disagree a little
- Neither agree nor disagree
- Agree a little
- Moderately agree
- Strongly agree

2.1 I rely on myself most of the time; I rarely rely on others
2.2 Winning is everything
2.3 Competition is the law of nature
2.4 When another person does better than I do, I get tense

Q3. BEHAVIORAL TENDENCIES
"People react differently in various situations. We're interested in understanding how you typically respond. For each statement below, indicate how much you agree or disagree based on your usual behavior."

Answer options:
- Strongly disagree
- Moderately disagree
- Disagree a little
- Neither agree nor disagree
- Agree a little
- Moderately agree
- Strongly agree

3.1 When I'm in a great mood, I get into situations that could cause me problems [Positive Urgency]

</general_questions>

For this step:
1. Analyze the survey responses for evidence of these personality traits, values, and behaviors
2. For each item, provide brief reasoning FIRST based on response evidence
3. Then predict the answer by selecting ONE option from the provided answer choices

=== STEP 2: Generate General Profile and Biography ===

Now, using BOTH the survey responses AND your personality/values predictions from Step 1, analyze holistically to understand this person as a complete individual.

Create a short_bio that:
- Is exactly 4-5 sentences long
- Uses SECOND-PERSON perspective (referring to the person as "you")
- Describes their lifestyle, life stage, interests, and social context
- Naturally integrates personality traits and values from Step 1
- Feels like a realistic description of a complete person
- Uses narrative style with concrete details about their life
- Implicitly reveals demographics, personality, and values
- Does NOT use a specific name - address the person directly as "you"

=== STEP 3: Extract Specific Demographic and Psychographic Characteristics ===

Based on the survey responses, personality/values predictions from Step 1, AND the short_bio from Step 2, now systematically identify specific demographic and psychographic characteristics.

You MUST determine the following 11 REQUIRED demographic and psychographic characteristics:

<profile_fields>
- CREGION: User's region in the US [Midwest, Northeast, South, West]
- AGE: User's age group [18-29, 30-49, 50-64, 65+]
- SEX: User's gender [Female, Male, In some other way]
- EDUCATION: Education level [Less than high school, High school graduate, Some college, no degree, Associate's degree, College graduate/some postgrad, Postgraduate]
- MARITAL: Marital status [Married, Living with a partner, Divorced, Separated, Widowed, Never been married]
- RELIG: Religious affiliation [Protestant, Roman Catholic, Mormon, Orthodox, Jewish, Muslim, Buddhist, Hindu, Atheist, Agnostic, Nothing in particular, Christian, Unitarian, Other]
- RELIGATTEND: Religious service attendance [More than once a week, Once a week, Once or twice a month, A few times a year, Seldom, Never]
- POLPARTY: Political party affiliation [Democrat, Republican, Independent, Other]
- POLIDEOLOGY: Political ideology [Very liberal, Liberal, Moderate, Conservative, Very conservative]
- INCOME: Annual household income [Less than $30,000, $30,000-$50,000, $50,000-$75,000, $75,000-$100,000, $100,000 or more]
- RACE: Racial/ethnic background [White, Black, Asian, Hispanic, Mixed Race, Other]
</profile_fields>

For each of these 11 REQUIRED fields, you MUST:
1. Provide detailed reasoning FIRST based on evidence from responses, personality/values predictions, and bio
2. Then select your answer ONLY from the provided options listed in brackets
3. Assess confidence level (high/medium/low)

ADDITIONALLY, you may identify any other inferable characteristics.

=== OUTPUT FORMAT ===

Provide your output in JSON format with six keys:

{{
"step1_analysis": "Your thinking process for personality and values predictions: What patterns in the survey responses reveal personality traits? What language suggests values? What behaviors are evident?",

"personality_and_values": {{
    "Q1.1_direct_questioning": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.2_anxious_easily_upset": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.3_open_to_new_experiences_complex": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.4_reserved_quiet": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.5_sympathetic_warm": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.6_flexible_spontaneous": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.7_calm_emotionally_stable": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q1.8_traditional_practical": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.1_self_reliance": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.2_winning_is_everything": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.3_competition_law_of_nature": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q2.4_tense_when_others_do_better": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }},
    "Q3.1_positive_urgency": {{
        "reasoning": "Brief explanation based on response evidence",
        "answer": "One of: Strongly disagree, Moderately disagree, Disagree a little, Neither agree nor disagree, Agree a little, Moderately agree, Strongly agree"
    }}
}},

"step2_analysis": "Your detailed thinking process for Step 2: Integrating survey responses, personality predictions, and values to understand this complete person. How do personality traits and values inform their broader identity and predict additional survey responses?",

"short_bio": "A 4-5 sentence narrative in SECOND-PERSON describing their life, naturally integrating personality traits and values",

"step3_analysis": "Your detailed thinking process for Step 3: How do survey responses, personality/values, and bio together inform each demographic characteristic? How would this person with these traits answer additional social survey questions?",

"user_profile": {{
    "CHARACTERISTIC_NAME": {{
        "reasoning": "Detailed evidence-based reasoning including how personality/values influence this characteristic and survey responses",
        "value": "Selected answer from provided options",
        "confidence": "high/medium/low"
    }}
}}
}}

=== EXAMPLE OUTPUT ===

<output>
{{
"step1_analysis": "Analyzing the survey responses for personality traits: The responses demonstrate straightforward, practical thinking with emphasis on concrete solutions over abstract theory, suggesting traditional/practical orientation (Q1.8 - likely Moderately agree). Consistent references to faith-based values and moral certainty indicate low anxiety about decisions (Q1.2 - likely Moderately disagree) and high emotional stability (Q1.7 - likely Moderately agree). Direct statements about political and social issues without hedging suggest direct/questioning nature (Q1.1 - likely Moderately agree), though community-focused language suggests moderate extraversion rather than extreme (Q1.4 - likely Neither agree nor disagree). Strong emphasis on family, community, and church relationships indicates warmth and sympathy (Q1.5 - likely Moderately agree). Preference for established traditions and proven methods over experimental approaches suggests lower openness to new experiences (Q1.3 - likely Disagree a little). Responses emphasizing order and structure over flexibility indicate lower spontaneity (Q1.6 - likely Disagree a little). For values: Strong emphasis on personal responsibility, business ownership, and 'making it on your own' indicates high self-reliance (Q2.1 - likely Moderately agree). However, community and church involvement temper extreme individualism. Business ownership context and pride in company success suggest moderate competitiveness (Q2.2, Q2.3 - likely Agree a little), but not cutthroat orientation. No evidence of interpersonal jealousy (Q2.4 - likely Disagree a little). Practical, methodical approach to business and life suggests low impulsivity (Q3.1 - likely Moderately disagree).",

"personality_and_values": {{
    "Q1.1_direct_questioning": {{
        "reasoning": "Survey responses show direct statements about political and moral positions without qualification. Clear expression of views on social issues, government role, and values. Not confrontational but straightforward.",
        "answer": "Moderately agree"
    }},
    "Q1.2_anxious_easily_upset": {{
        "reasoning": "Responses reflect confidence and certainty in beliefs. Faith-based worldview provides stability. No evidence of emotional reactivity or worry. Handles disagreement with calm conviction.",
        "answer": "Moderately disagree"
    }},
    "Q1.3_open_to_new_experiences_complex": {{
        "reasoning": "Preferences for traditional values, established practices, and proven methods over experimental or progressive approaches. Values stability and continuity. Skeptical of rapid social change. However, not completely closed - business requires some adaptation.",
        "answer": "Disagree a little"
    }},
    "Q1.4_reserved_quiet": {{
        "reasoning": "Active in community leadership (church deacon, chamber of commerce) suggests comfort with social engagement. However, emphasis on close relationships and church community rather than broad networking. Moderate social engagement focused on established relationships.",
        "answer": "Neither agree nor disagree"
    }},
    "Q1.5_sympathetic_warm": {{
        "reasoning": "Strong emphasis on family relationships, church community, and caring for others. Values personal connections. Community involvement and mentoring in youth sports. However, warmth expressed through traditional structures (church, family) rather than universal empathy.",
        "answer": "Moderately agree"
    }},
    "Q1.6_flexible_spontaneous": {{
        "reasoning": "Responses emphasize planning, structure, traditional approaches, and established routines. Business ownership requires discipline and consistency. Faith provides structured worldview. Values stability over spontaneity.",
        "answer": "Disagree a little"
    }},
    "Q1.7_calm_emotionally_stable": {{
        "reasoning": "Confident expression of views, stable faith-based foundation, established life with family and business. No evidence of emotional volatility. Handles challenges with practical problem-solving rather than emotional reaction.",
        "answer": "Moderately agree"
    }},
    "Q1.8_traditional_practical": {{
        "reasoning": "Consistent emphasis on traditional values, proven methods, practical solutions, and established practices. Business in traditional skilled trade. Faith-based conventional morality. Skepticism of progressive social changes. Values what works over what's novel.",
        "answer": "Moderately agree"
    }},
    "Q2.1_self_reliance": {{
        "reasoning": "Strong emphasis on personal responsibility, business ownership ('made my own way'), and limited government. However, also values family, church community, and mutual support. High self-reliance but not extreme individualism.",
        "answer": "Moderately agree"
    }},
    "Q2.2_winning_is_everything": {{
        "reasoning": "Business ownership context shows some competitive drive and pride in success. However, primary focus is on providing for family, serving customers, and community contribution rather than beating competitors. Moderate competitive drive.",
        "answer": "Agree a little"
    }},
    "Q2.3_competition_law_of_nature": {{
        "reasoning": "Free market values and business perspective suggest acceptance of competition. However, Christian values emphasize cooperation and community. Sees competition as natural in business but not as ultimate principle. Balanced view.",
        "answer": "Agree a little"
    }},
    "Q2.4_tense_when_others_do_better": {{
        "reasoning": "Emotionally stable personality and faith-based security suggest low interpersonal jealousy. Success framed as providing for family rather than status competition. Community orientation over zero-sum thinking.",
        "answer": "Disagree a little"
    }},
    "Q3.1_positive_urgency": {{
        "reasoning": "Practical, methodical approach to business and life decisions. Traditional values emphasize caution and responsibility. No evidence of impulsive behavior. Plans ahead (college savings, business management).",
        "answer": "Moderately disagree"
    }}
}},

"step2_analysis": "Integrating the personality and values predictions with the survey responses: This person shows moderate-to-high traditionalism (Moderately agree on traditional/practical) combined with high emotional stability (Moderately agree on calm, Moderately disagree on anxious) and moderate warmth (Moderately agree), creating a profile of a stable, practical, community-oriented conservative. Low openness to new experiences (Disagree a little) and low flexibility (Disagree a little) combined with high traditionalism predict conservative political ideology and resistance to rapid social change. The moderate self-reliance (Moderately agree) balanced with community values suggests small-government conservatism tempered by social responsibility through private institutions (church, family, community). The direct/questioning trait (Moderately agree) combined with emotional stability suggests confidence in expressing traditional conservative views. Low competitiveness (Agree a little on competition questions but not extreme) indicates business-oriented but not ruthless mentality - success through service and hard work rather than defeating others. This personality profile - high traditionalism, low openness, moderate warmth, low flexibility - is highly predictive of conservative political ideology and traditional social values. The emotionally stable, practical approach suggests established middle-aged conservative with business and family responsibilities. Overall, this personality profile strongly aligns with religious, traditional, small business owner demographics in Southern/Midwestern suburban areas and would predict consistently conservative responses on social surveys covering politics, social issues, and values, with particular emphasis on personal responsibility, traditional family structures, religious liberty, and limited government.",

"short_bio": "You are a 52-year-old small business owner running an HVAC company in suburban Dallas, Texas, where you've lived your entire life. Your strong sense of personal responsibility and traditional values guide both your business decisions and your family life with your wife of 28 years and three children, and you take pride in building something through hard work rather than relying on others. Sundays find you at the Baptist church where you've served as a deacon for over a decade, providing you with a stable community that shares your practical, faith-centered approach to life's challenges. Your calm, methodical temperament serves you well in managing your team of technicians and navigating the ups and downs of business ownership, and while you're not the most outgoing person, you're deeply connected to your close-knit circle of family, church friends, and longtime business relationships. You approach change cautiously, preferring proven methods and traditional structures that have worked for generations, though you're direct and confident in expressing your views when the situation calls for it.",

"step3_analysis": "Based on the survey responses, personality profile (high traditionalism via Moderately agree, low openness via Disagree a little, moderate warmth via Moderately agree, high emotional stability via Moderately agree on calm/Moderately disagree on anxious, moderate self-reliance via Moderately agree), and bio, I can now extract specific characteristics. The high traditionalism (Moderately agree) and low openness (Disagree a little) strongly predict conservative political ideology and Republican party affiliation. The combination of high emotional stability with traditional values predicts confidence in conservative positions across survey topics. The moderate self-reliance (Moderately agree) balanced with community orientation (church, family) predicts small-government conservatism with support for traditional community institutions. The direct/questioning trait (Moderately agree) combined with low anxiety suggests someone who confidently expresses conservative views in surveys. Dallas suburban location (South region) combined with this personality profile creates a typical Southern suburban conservative demographic segment. Married status with traditional family structure combined with high traditionalism (Moderately agree) strongly aligns with conservative views on family and social issues. Male gender in business ownership with practical orientation typical of male-dominated skilled trades. Protestant faith with active church involvement (deacon role) combined with traditional values predicts religious conservative positions. The moderate competitiveness (Agree a little) in business context combined with community values suggests entrepreneurial conservatism rather than libertarian individualism. Overall, this combination of demographics and personality traits predicts consistently conservative responses across diverse social survey topics, with particular strength on traditional values, religious liberty, limited government, personal responsibility, and conventional family structures.",

"user_profile": {{
    "CREGION": {{
        "reasoning": "Bio explicitly places person in Dallas, Texas suburbs. Survey responses reflect Southern conservative values and cultural markers. High traditionalism (Moderately agree) combined with Southern location predicts strong support for regional cultural values, traditional social norms, and conservative political positions typical of Southern suburbs.",
        "value": "South",
        "confidence": "high"
    }},
    "AGE": {{
        "reasoning": "Bio specifies 52 years old with 28-year marriage and three children (two in college, one in high school). Age combined with low openness (Disagree a little) and high traditionalism (Moderately agree) predicts established conservative views resistant to social change. This age group shows strong partisan loyalty and traditional values. Life stage indicators (established business, adult children, long marriage, church leadership) confirm mature adulthood in 50-64 bracket.",
        "value": "50-64",
        "confidence": "high"
    }},
    "SEX": {{
        "reasoning": "Bio describes HVAC business ownership in male-dominated trade, married to wife, traditional family structure. Survey responses show practical problem-solving orientation typical of male-dominated skilled trades. Male small business owners with high traditionalism (Moderately agree) and moderate self-reliance (Moderately agree) show strong conservative positions on surveys, particularly on economic and social policy.",
        "value": "male",
        "confidence": "high"
    }},
    "EDUCATION": {{
        "reasoning": "Survey language is practical and straightforward but not highly academic. HVAC business ownership typically requires technical training (associate's degree, apprenticeship, certifications) rather than four-year degree. High traditionalism (Moderately agree) and low openness (Disagree a little) combined with skilled trade background suggest practical/vocational education path. Most likely 'some college, no degree' or 'associate's degree' for technical training in skilled trade.",
        "value": "some college, no degree",
        "confidence": "medium"
    }},
    "MARITAL": {{
        "reasoning": "Bio explicitly states 'wife of 28 years' and survey responses reference traditional family structure. Long-term marriage central to identity and values. High traditionalism (Moderately agree) strongly aligns with valuing marriage institution. Married status combined with traditional values predicts conservative positions on family policy and social issues.",
        "value": "married",
        "confidence": "high"
    }},
    "RELIG": {{
        "reasoning": "Bio explicitly identifies Baptist church membership with deacon leadership role. Survey responses consistently reference faith-based values and Christian worldview. High traditionalism (Moderately agree) and moderate warmth (Moderately agree) expressed through religious community. Protestant (specifically Baptist) affiliation with active leadership role predicts strong religious conservative positions across survey topics.",
        "value": "protestant",
        "confidence": "high"
    }},
    "RELIGATTEND": {{
        "reasoning": "Bio describes active deacon role in Baptist church, which requires regular attendance plus additional leadership responsibilities. Survey responses show faith as central organizing principle. High traditionalism (Moderately agree) and structured approach (low flexibility Disagree a little) suggest consistent religious practice. Deacon role typically requires 'more than once a week' attendance for services plus meetings, committees, and church activities.",
        "value": "more than once a week",
        "confidence": "high"
    }},
    "POLPARTY": {{
        "reasoning": "Survey responses show consistent conservative positions on all issues. High traditionalism (Moderately agree), low openness (Disagree a little), moderate self-reliance (Moderately agree), and faith-based values all strongly predict Republican affiliation. Small business owner in Texas, Baptist church leader, traditional family values - all correlate heavily with Republican party identification. Would definitely identify as Republican in surveys.",
        "value": "Republican",
        "confidence": "high"
    }},
    "POLIDEOLOGY": {{
        "reasoning": "Survey responses show consistently conservative positions across social and economic issues. High traditionalism (Moderately agree) and low openness (Disagree a little) predict conservative ideology. However, moderate warmth (Moderately agree) and community focus suggest mainstream conservatism rather than 'very conservative.' Practical business orientation and established community ties indicate 'conservative' rather than far-right positions. Likely identifies as 'conservative' on surveys.",
        "value": "conservative",
        "confidence": "high"
    }},
    "INCOME": {{
        "reasoning": "HVAC business owner in Dallas suburbs supporting family of five with two in college. Small business ownership in skilled trade typically generates $60,000-$120,000 owner income. Dual income if wife works. Bio mentions comfortable lifestyle but also college cost concerns, suggesting solid middle class. Business owners in this demographic typically fall in '$75,000-$100,000' or '$50,000-$75,000' range depending on business size and wife's employment.",
        "value": "$75,000-$100,000",
        "confidence": "medium"
    }},
    "RACE": {{
        "reasoning": "Cultural references, unstated normative perspective, and social positioning align with white American Southern experience. Baptist church demographics in Dallas suburbs, small business ownership patterns, and traditional family structure all suggest white identity. High traditionalism (Moderately agree) and Southern location correlate with white Southern conservative demographic. Based on statistical likelihood and cultural patterns.",
        "value": "white",
        "confidence": "medium"
    }},
    "personality_summary": {{
        "reasoning": "Step 1 answers create clear Big Five profile: Low Openness (Disagree a little), Moderate Agreeableness (warmth Moderately agree but within traditional bounds), High Conscientiousness (business ownership, structured life, deacon responsibilities), Low Neuroticism (calm Moderately agree, not anxious Moderately disagree), Moderate Extraversion (Neither agree nor disagree on reserved). This profile predicts conservative political views, traditional social values, religious commitment, and support for limited government with strong private institutions (church, family, business).",
        "value": "Big Five: Low Openness, Moderate Agreeableness, High Conscientiousness, Low Neuroticism, Moderate Extraversion - Traditional, practical, stable conservative",
        "confidence": "high"
    }},
    "values_orientation": {{
        "reasoning": "Moderate scores on self-reliance (Moderately agree) combined with community focus indicate individualism balanced by traditional institutions. Low competitiveness (Agree a little) suggests success through service rather than defeating others. High traditionalism (Moderately agree) indicates valuing proven methods and established structures. Low impulsivity (Moderately disagree on positive urgency) indicates planned, responsible approach.",
        "value": "Personal responsibility balanced with community support, moderate self-reliance, traditional values, low impulsivity, success through service and hard work",
        "confidence": "high"
    }},
    "professional_field": {{
        "reasoning": "Bio explicitly states HVAC business owner. Survey responses show small business perspective on economic and regulatory issues. High conscientiousness and practical orientation align with skilled trade entrepreneurship. Small business owners with traditional values show strong support for limited regulation, lower taxes, and free market policies in surveys.",
        "value": "Small business owner (HVAC/skilled trades)",
        "confidence": "high"
    }},
    "lifestyle_category": {{
        "reasoning": "Bio and personality profile (high traditionalism Moderately agree, low openness Disagree a little, moderate warmth Moderately agree, low flexibility Disagree a little) reveal traditional family-centered suburban lifestyle. Married with children, homeowner, active church member, business owner, community involved. Lifestyle structured around faith, family, and business rather than cultural/recreational activities. Traditional middle-class suburban American family pattern.",
        "value": "Traditional suburban family, faith-centered, small business owner, community-involved through church and youth activities",
        "confidence": "high"
    }}
}}
}}
</output>

Now, please analyze the provided survey responses and generate the complete user persona profile following this three-step process. Start with Step 1 (personality and values predictions with reasoning before answer), then Step 2 (biographical profile), then Step 3 (demographic analysis). Remember: this persona will be used to predict additional social survey responses across diverse topics."""
            
    return system_prompt, prompt

def main(args):
    print(args)
    print("="*60)
    print("Parallel Inference Experiment")
    print("="*60)
    print(f"Data: {args.data_dir}")
    print(f"Model: {args.sim_model}")
    print(f"Prompt format: {args.format}")
    print(f"Feature set: {args.feature_set}")
    print(f"Parallel Processes: {args.proc}")
    print("="*60)

    random.seed(1004)

    # Load data from file
    with open(args.data_dir, 'r') as f:
        data = json.load(f) 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = os.getcwd()

    results = []
    
    # Get the appropriate target features
    target_features = get_target_features_by_set(args.feature_set)
    print(f"\n📊 Using {len(target_features)} target feature-value pairs")
    
    print(f"\n📊 Persona Generation Plan:")
    print("="*60)

    # STEP 1: Prepare all prompts for parallel inference
    print("\nGenerating prompts for all users...")

    system_prompts_batch = []
    user_prompts_batch = []
    user_map = []

    for user_idx, histories in enumerate(data):  
        user_behavior = ""
        if args.num_history.lower()=='all':
            num_reviews = len(histories)
        else:
            num_reviews = min(len(histories), int(args.num_history))
        
        if args.domain.lower() =='movie':
            for i in range(num_reviews):
                history = histories[i]
                user_behavior += f"Movie Title: {history['movie_title']}\n"
                user_behavior += f"Review Title: {history['review_title']}\n"
                user_behavior += f"Review: {history['review_content']}\n\n"
            
            system_prompt, user_prompt_template = generate_persona_prompt(args.domain, args.format)
            user_prompt = user_prompt_template.format(movie_reviews=user_behavior)
            
            system_prompts_batch.append(system_prompt)
            user_prompts_batch.append(user_prompt)
        
        elif args.domain.lower()=='shopping':
            for i in range(1, num_reviews+1):
                history = histories[-i]
                user_behavior += f"[Recent Purchase - {i}\n"
                user_behavior += f"Purchase Item: {history['Title']}\n"
                user_behavior += f"Quantity: {history['Quantity']}\n"
                user_behavior += f"Purchase Price Per Unit: {history['Purchase Price Per Unit']}"
                user_behavior += f"Shipping Address State: {history['Shipping Address State']}"

            system_prompt, user_prompt_template = generate_persona_prompt(args.domain, args.format)
            user_prompt = user_prompt_template.format(shopping_history=user_behavior)
            
            system_prompts_batch.append(system_prompt)
            user_prompts_batch.append(user_prompt)

        elif 'survey' in args.domain.lower():
            for i in range(num_reviews):
                history = histories[i]
                user_behavior += f"Survey Question {i+1}: {history['stimuli']}\n"
                user_behavior += f"User Response to Question {i+1}: {history['response']}\n"
            
            system_prompt, user_prompt_template = generate_persona_prompt(args.domain, args.format)
            user_prompt = user_prompt_template.format(survey_responses=user_behavior)
            
            system_prompts_batch.append(system_prompt)
            user_prompts_batch.append(user_prompt)

        elif 'gss' in args.domain.lower() or 'opinionqa' in args.domain.lower():
            # if args.format=='free_gt':
            #     num_reviews = len(histories['responses'])
            #     for i in range(num_reviews):
            #         history = histories['responses'][i]
            #         user_behavior += f"Survey Question {i+1}: {history['Question']}\n"
            #         user_behavior += f"User Response to Question {i+1}: {history['Response']}\n"
            #     features=''
            #     for k,v in histories['feature'].items():
            #         features+=f'{k}: {v}\n'
            #     system_prompt, user_prompt_template = generate_persona_prompt(args.domain, args.format)
            #     user_prompt = user_prompt_template.format(features=features,survey_responses=user_behavior)
                
            # else:
            #     num_reviews = len(histories['responses'])
            #     for i in range(num_reviews):
            #         history = histories['responses'][i]
            #         user_behavior += f"Survey Question {i+1}: {history['Question']}\n"
            #         user_behavior += f"User Response to Question {i+1}: {history['Response']}\n"
                
            #     system_prompt, user_prompt_template = generate_persona_prompt(args.domain, args.format)
            #     user_prompt = user_prompt_template.format(survey_responses=user_behavior)
                
            # system_prompts_batch.append(system_prompt)
            # user_prompts_batch.append(user_prompt)
            num_reviews=0
            i=0
            for review in histories['responses']:
                if len(review['Response']):
                    num_reviews+=1
                    user_behavior += f"Survey Question {i+1}: {review['Question']}\n"
                    user_behavior += f"Options for Question {i+1}: {review['Possible_Answers']}\n"
                    user_behavior += f"User Response to Question {i+1}: {review['Response']}\n"            
                    i+=1
            if 'gt' in args.format:
                features=''
                for k,v in histories['feature'].items():
                    features+=f'{k}: {v}\n'
                system_prompt, user_prompt_template = generate_persona_prompt(args.domain, args.format)
                user_prompt = user_prompt_template.format(features=features,survey_responses=user_behavior)
            else:
                system_prompt, user_prompt_template = generate_persona_prompt(args.domain, args.format)
                user_prompt = user_prompt_template.format(survey_responses=user_behavior)
            system_prompts_batch.append(system_prompt)
            user_prompts_batch.append(user_prompt)
        user_map.append({
            'batch_idx': len(system_prompts_batch) - 1,
            'user_idx': user_idx,
            'num_reviews': num_reviews
        })
    print(f"✅ Generated {len(system_prompts_batch)} prompts for {len(data)} users")
    
    # STEP 2: Run parallel inference for all prompts
    if system_prompts_batch:
        print("\n" + "="*60)
        print("Running parallel inference...")
        print("="*60)
        llm_params = {"temperature": 0.7, "top_p": 1.0, "max_tokens": args.max_token}
        input_data = create_dataset_from_prompts(system_prompts_batch, user_prompts_batch)
        results_df = process_dataset_parallel(
            dataset=input_data,
            model_id=args.sim_model,
            llm_params=llm_params,
            region_name="us-east-1",
            num_proc=args.proc
        )
        print(f"✅ Parallel inference complete: {len(results_df)} responses received")
        
        # STEP 3: Process and parse results
        print("\n" + "="*60)
        print("Processing and parsing results...")
        print("="*60)
        
        results = []
        parse_errors = 0
        partial_parses = 0
        partial_parse_details = []
        
        expected_fields = EXPECTED_FIELDS
        num_expected_fields = len(expected_fields)
        
        for mapping in user_map:
            batch_idx = mapping['batch_idx']
            if batch_idx < len(results_df):
                row = results_df.iloc[batch_idx]
                response_text = row['pred_label']
                
                try:
                    parsed_response = parse_llm_persona_response(response_text, expected_fields)
                    
                    # Check if this is a partial parse
                    user_profile = parsed_response.get('user_profile', {})
                    num_fields_found = len(user_profile)
                    is_partial = num_fields_found < num_expected_fields
                    
                    if is_partial:
                        partial_parses += 1
                        details = log_partial_parse_details(
                            mapping['user_idx'], 
                            parsed_response, 
                            expected_fields
                        )
                        partial_parse_details.append(details)
                        
                        # Only print details for completely failed parses (0 fields)
                        if num_fields_found == 0:
                            print(f"⚠️ Complete parsing failure for user {mapping['user_idx']}: 0/{num_expected_fields} fields found")
                            # Print first 200 chars of response for debugging
                            print(f"   Response preview: {response_text[:200]}...")
                        else:
                            print(f"⚠️ Partial parse for user {mapping['user_idx']}: "
                                  f"{num_fields_found}/{num_expected_fields} fields found")
                    
                    # Extract user profile and other top-level fields
                    analysis = parsed_response.get('analysis', '')
                    short_bio = parsed_response.get('short_bio', '')
                    
                    # Build result entry
                    persona_entry = {
                        'user_idx': mapping['user_idx'],
                        'num_reviews_used': mapping['num_reviews'],
                        'analysis': analysis,
                        'short_bio': short_bio,
                        'persona': {},
                        'parse_status': 'partial' if is_partial else 'complete'
                    }
                    
                    # Extract each demographic field
                    for field in expected_fields:
                        if field in user_profile:
                            value = user_profile[field].get('value', 'Not-deducible')
                            reasoning = user_profile[field].get('reasoning', '')
                            confidence = user_profile[field].get('confidence', 'medium')
                            
                            persona_entry['persona'][field] = {
                                'value': value,
                                'reasoning': reasoning,
                                'confidence': confidence
                            }
                        else:
                            persona_entry['persona'][field] = {
                                'value': 'Not-deducible',
                                'reasoning': 'Field not found in response (partial parse)',
                                'confidence': 'low'
                            }

                    results.append(persona_entry)
                    
                except Exception as e:
                    parse_errors += 1
                    error_msg = str(e)[:200]
                    print(f"❌ Exception for user {mapping['user_idx']}: {error_msg}")
                    
                    # Save error case
                    results.append({
                        'user_idx': mapping['user_idx'],
                        'num_reviews_used': mapping['num_reviews'],
                        'analysis': '',
                        'short_bio': '',
                        'persona': {},
                        'parse_status': 'error',
                        'error': error_msg,
                        'raw_response': response_text[:1000]
                    })
        
        print(f"\n✅ Successfully processed {len(results) - parse_errors} personas")
        if partial_parses > 0:
            print(f"⚠️ {partial_parses} partial parses (missing some fields)")
        if parse_errors > 0:
            print(f"❌ {parse_errors} complete parsing failures")
        
        # Create output directory
        persona_dir = Path(directory) / 'out' / 'ours_persona' / 'persona' / f'{args.sim_model}' /'split' 
        if not persona_dir.exists():
            persona_dir.mkdir(parents=True)
        
        # Save individual results (original format)
        output_file = persona_dir / f'{args.domain}_personas_{args.num_history}reviews_{args.format}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Individual personas saved to: {output_file}")
        
        # Save partial parse details if any
        if partial_parse_details:
            partial_details_file = persona_dir / f'{args.domain}_partial_parse_details_{args.num_history}reviews_{args.format}.json'
            with open(partial_details_file, 'w') as f:
                json.dump(partial_parse_details, f, indent=2)
            print(f"📋 Partial parse details saved to: {partial_details_file}")
        
        # STEP 4: Group personas by target features
        print("\n" + "="*60)
        print(f"Grouping personas by {len(target_features)} target demographic features...")
        print("="*60)
        
        # Only use successfully parsed personas for grouping (include partial parses)
        valid_results = [r for r in results if r.get('parse_status') != 'error']
        print(f"Using {len(valid_results)} valid personas for grouping "
              f"({len(results) - len(valid_results)} excluded due to errors)")
        
        grouped_personas_list = group_personas_by_target_features(valid_results, target_features)
        
        # Save the grouped output as a single JSON file
        grouped_output_file = persona_dir / f'{args.domain}_personas_{args.feature_set}features_{args.num_history}reviews_{args.format}.json'
        with open(grouped_output_file, 'w') as f:
            json.dump(grouped_personas_list, f, indent=2, ensure_ascii=False)
        print(f"\n✅ {len(target_features)}-feature grouped personas saved to: {grouped_output_file}")
        
        # Save detailed results
        detailed_results = []
        for mapping in user_map:
            batch_idx = mapping['batch_idx']
            if batch_idx < len(results_df):
                row = results_df.iloc[batch_idx]
                detailed_results.append({
                    'user_idx': mapping['user_idx'],
                    'num_reviews_used': mapping['num_reviews'],
                    'system_prompt': row['system_prompt'],
                    'user_message': row['user_message'],
                    'response': row['pred_label'],
                    'input_tokens': int(row['input_tokens']),
                    'output_tokens': int(row['output_tokens'])
                })

        detailed_file = persona_dir / f'{args.domain}_detailed_responses_{args.num_history}reviews_{args.format}.json'
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"📝 Detailed responses saved to: {detailed_file}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        total_input_tokens = results_df['input_tokens'].sum()
        total_output_tokens = results_df['output_tokens'].sum()
        successful_personas = len(valid_results)
        complete_personas = len([r for r in results if r.get('parse_status') == 'complete'])
        
        print(f"Total users processed: {len(results)}")
        print(f"Complete parses: {complete_personas}")
        print(f"Partial parses: {partial_parses} (recovered {len(valid_results) - complete_personas} personas)")
        print(f"Complete failures: {parse_errors}")
        print(f"Success rate: {(len(valid_results)/len(results)*100):.1f}%")
        print(f"Total input tokens: {total_input_tokens:,}")
        print(f"Total output tokens: {total_output_tokens:,}")
        if len(results) > 0:
            print(f"Average tokens per user: {(total_input_tokens + total_output_tokens) / len(results):,.0f}")
        
        # Show which fields had issues in partial parses
        if partial_parse_details:
            print("\n" + "-"*60)
            print("Partial Parse Field Analysis")
            print("-"*60)
            missing_field_counts = defaultdict(int)
            for detail in partial_parse_details:
                for field in detail['missing_field_names']:
                    missing_field_counts[field] += 1
            
            if missing_field_counts:
                print("Fields missing across partial parses:")
                for field, count in sorted(missing_field_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {field}: missing in {count} partial parse(s)")
        
        # Print feature grouping summary
        print("\n" + "="*60)
        print(f"{len(target_features)}-Feature Grouped Personas Summary")
        print("="*60)
        print(f"Total feature groups: {len(grouped_personas_list)}")
        
        # Count personas per feature
        total_grouped_personas = sum(g['total_personas_generated'] for g in grouped_personas_list)
        print(f"Total personas in groups: {total_grouped_personas}")
        
        # Show top features by persona count
        sorted_groups = sorted(grouped_personas_list, 
                              key=lambda x: x['total_personas_generated'], 
                              reverse=True)
        
        print("\nTop 10 features by persona count:")
        for idx, group in enumerate(sorted_groups[:10], 1):
            feature_map = group['group_prompt_variable_map']
            count = group['total_personas_generated']
            feature_str = ', '.join([f"{k}={v}" for k, v in feature_map.items()])
            print(f"{idx}. {feature_str}: {count} personas")
        
        # Count and show features with zero matches
        zero_groups = [g for g in grouped_personas_list if g['total_personas_generated'] == 0]
        if zero_groups:
            print(f"\n⚠️ Features with 0 personas ({len(zero_groups)}/{len(grouped_personas_list)}):")
            for group in zero_groups[:10]:  # Show first 10
                feature_map = group['group_prompt_variable_map']
                feature_str = ', '.join([f"{k}={v}" for k, v in feature_map.items()])
                print(f"  - {feature_str}")
            if len(zero_groups) > 10:
                print(f"  ... and {len(zero_groups) - 10} more")
        
        # Print detailed coverage report
        print_feature_coverage_report(grouped_personas_list, args.feature_set)
        
        print("="*60)

if __name__ == '__main__':
    args = get_args()
    main(args)

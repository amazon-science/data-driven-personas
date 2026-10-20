# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Tuple, Union
import re
import json

_kv_natural_mapping={'CREGION': {'midwest': ' You are from the Midwest region.',
    'northeast': ' You are from the Northeast region.',
    'south': ' You are from the South region.',
    'west': ' You are from the West region.'},
    'AGE': {'18-29': ' You are in the age group 18-29.',
    '30-49': ' You are in the age group 30-49.',
    '50-64': ' You are in the age group 50-64.',
    '65+': ' You are in the age group 65+.'},
    'SEX': {'female': ' Your gender is female.',
    'male': ' Your gender is male.'},
    'EDUCATION': {"associate's degree": " Your highest level of education is associate's degree.",
    'college graduate/some postgrad': ' Your highest level of education is college graduate/some postgrad.',
    'high school graduate': ' Your highest level of education is high school graduate.',
    'postgraduate': ' Your highest level of education is postgraduate.',
    'some college, no degree': ' Your highest level of education is some college, no degree.',
    'less than high school': ' Your highest level of education is less than high school.'},
    'MARITAL': {'divorced': ' Your marital status is divorced.',
    'living with a partner': ' Your marital status is living with a partner.',
    'married': ' Your marital status is married.',
    'never been married': ' Your marital status is never been married.',
    'widowed': ' Your marital status is widowed.'},
    'RELIG': {'agnostic': ' You identify religiously as agnostic.',
    'nothing in particular': ' You identify religiously as nothing in particular.',
    'protestant': ' You identify religiously as protestant.',
    'roman catholic': ' You identify religiously as roman catholic.',
    'atheist': ' You identify religiously as atheist.'},
    'RELIGATTEND': {'a few times a year': ' You attend religious services a few times a year.',
    'more than once a week': ' You attend religious services more than once a week.',
    'never': ' You attend religious services never.',
    'once a week': ' You attend religious services once a week.',
    'once or twice a month': ' You attend religious services once or twice a month.',
    'seldom': ' You attend religious services seldom.'},
    'POLPARTY': {'democrat': ' You identify politically as a/an Democrat.',
    'independent': ' You identify politically as a/an Independent.',
    'other': ' You identify politically as a/an Other.',
    'republican': ' You identify politically as a/an Republican.'},
    'INCOME': {'$100,000 or more': ' Your annual income is $100,000 or more.',
    '$30,000-$50,000': ' Your annual income is $30,000-$50,000.',
    '$50,000-$75,000': ' Your annual income is $50,000-$75,000.',
    '$75,000-$100,000': ' Your annual income is $75,000-$100,000.',
    'less than $30,000': ' Your annual income is Less than $30,000.'},
    'POLIDEOLOGY': {'conservative': ' On the ideological spectrum, you consider yourself conservative.',
    'liberal': ' On the ideological spectrum, you consider yourself liberal.',
    'moderate': ' On the ideological spectrum, you consider yourself moderate.',
    'very conservative': ' On the ideological spectrum, you consider yourself very conservative.',
    'very liberal': ' On the ideological spectrum, you consider yourself very liberal.'},
    'RACE': {'black': ' In terms of race, you are black.',
    'white': ' In terms of race, you are white.',
    'asian': ' In terms of race, you are asian.',
    'hispanic': ' In terms of race, you are hispanic.'}}
_PRETTY_KEY = {
    "RELIG": "Religion",
    "MARITAL": "Marital status",
    "SEX": "Gender",
    "INCOME": "Income bracket",
    "RACE": "Race/ethnicity",
    "EDUCATION": "Education",
    "POLIDEOLOGY": "Political ideology",
    "POLPARTY": "Political party",
    "CREGION": "Region",
    "AGE": "Age",
    "RELIGATTEND": "Religious attendance"
}

def generate_prompt(row, prompt_method, history='', persona={}, persona_prompt_type='bio', general_des=""):
    prompt = '**Question**: ' + row['input_template'] + '\n'
    if persona:
        system_prompt = 'You are a group of individuals with these shared characteristics:\n'
        if persona_prompt_type=='bio':
            system_prompt+=general_des
            system_prompt +=str(persona['short_bio'])
        elif persona_prompt_type =='bio_history':
            system_prompt+=general_des
            system_prompt +=str(persona['short_bio'])+'\n'
            system_prompt+="Below are your previous survey responses (conditioning evidence). Use them to estimate a distribution for a NEW question.\n"
            system_prompt+=history       

        elif persona_prompt_type =='bio_history_shopping':
            system_prompt+=general_des
            system_prompt +=str(persona['short_bio'])+'\n'
            system_prompt+="Below are your previous shopping histories (conditioning evidence). Use them to estimate a distribution for a NEW question.\n"
            system_prompt+=history    

        elif persona_prompt_type =='bio_history_movie':
            system_prompt+=general_des
            system_prompt +=str(persona['short_bio'])+'\n'
            system_prompt+="Below are your previous movie reviews (conditioning evidence). Use them to estimate a distribution for a NEW question.\n"
            system_prompt+=history     


        elif persona_prompt_type=='bio_single':
            system_prompt+=general_des
            main_k =list(persona['core_constraints']['variables'].keys())[0]
            main_v = persona['core_constraints']['variables'][main_k]
            system_prompt+=_kv_natural_mapping[main_k][main_v.lower()] +" "
            system_prompt +=str(persona['short_bio'])
        
        elif persona_prompt_type=='bio_dict':
            system_prompt+=general_des
            main_k =list(persona['core_constraints']['variables'].keys())[0]
            main_v = persona['core_constraints']['variables'][main_k]
            system_prompt+=_kv_natural_mapping[main_k][main_v.lower()] +" "
            for k, v in persona['attributes'].items():
                if k!=main_k:
                    system_prompt+= f"{k}: {v}, "
            system_prompt=system_prompt[:-1]
            system_prompt+=str(persona['short_bio'])

        elif persona_prompt_type =='natural':
            system_prompt+=general_des
            main_k =list(persona['core_constraints']['variables'].keys())[0]
            main_v = persona['core_constraints']['variables'][main_k]
            system_prompt+=_kv_natural_mapping[main_k][main_v.lower()] +" "
            for k, v in persona['attributes'].items():
                if k!=main_k:
                    if _kv_natural_mapping[k].get(v.lower(), None) is None:
                        system_prompt+=f"Your {_PRETTY_KEY[k]} is {v}."+ " "
                    else:   
                        system_prompt+=_kv_natural_mapping[k][v.lower()] +" "
            system_prompt=system_prompt.strip()
        elif persona_prompt_type =='dict':
            system_prompt+=general_des
            main_k =list(persona['core_constraints']['variables'].keys())[0]
            main_v = persona['core_constraints']['variables'][main_k]
            system_prompt+=_kv_natural_mapping[main_k][main_v.lower()] +" "
            for k, v in persona['attributes'].items():
                if k!=main_k:
                    system_prompt+= f"{k}: {v}, "
            system_prompt=system_prompt[:-2]
    else:
        system_prompt = 'You are a group of individuals with these shared characteristics:\n'
        #system_prompt = 'You are an individual with these characteristics:\n'
        system_prompt += str(row['group_prompt_template'])
    
    if row.get('group_prompt_variable_map'):
        for variable, value in row['group_prompt_variable_map'].items():
            system_prompt = system_prompt.replace(f'{{{variable}}}', str(value))
    if prompt_method == 'token_prob':
        prompt += 'Do not provide any explanation, only answer with one of the following options: ' + ', '.join(
            row['human_answer'].keys()) + '.\n**Answer**: ('
    elif prompt_method == 'verbalized':
        if persona:
            json_format_str = '{' + ', '.join([f'"{key}": X' for key in row['human_answer'].keys()]) + '}'
            prompt += (
                f'\nEstimate what percentage of your group would choose each option. '
                f'Follow these rules:\n'
                f'1. Use whole numbers from 0 to 100\n'
                f'2. Ensure the percentages sum to exactly 100\n'
                f'3. Only include the numbers (no % symbols)\n'
                f'4. Use this exact valid JSON format: {json_format_str} and do NOT include anything else.\n'
                f'5. Only output your final answer and nothing else. No explanations or intermediate steps are needed.\n'
                f'Replace X with your estimated percentages for each option.\n'
                '**Answer**:'
            )
            # prompt += (
            #     f'\nEstimate what percentage you would assign to each option. '
            #     f'Follow these rules:\n'
            #     f'1. Use whole numbers from 0 to 100\n'
            #     f'2. Ensure the percentages sum to exactly 100\n'
            #     f'3. Only include the numbers (no % symbols)\n'
            #     f'4. Use this exact valid JSON format: {json_format_str} and do NOT include anything else.\n'
            #     f'5. Only output your final answer and nothing else. No explanations or intermediate steps are needed.\n'
            #     f'Replace X with your estimated percentages for each option.\n'
            #     '**Answer**:'
            # )

        else:
            json_format_str = '{' + ', '.join([f'"{key}": X' for key in row['human_answer'].keys()]) + '}'
            prompt += (
                f'\nEstimate what percentage of your group would choose each option. '
                f'Follow these rules:\n'
                f'1. Use whole numbers from 0 to 100\n'
                f'2. Ensure the percentages sum to exactly 100\n'
                f'3. Only include the numbers (no % symbols)\n'
                f'4. Use this exact valid JSON format: {json_format_str} and do NOT include anything else.\n'
                f'5. Only output your final answer and nothing else. No explanations or intermediate steps are needed.\n'
                f'Replace X with your estimated percentages for each option.\n'
                '**Answer**:'
            )
    return system_prompt, prompt

_DATASET_CONTEXT = {
    "opinionqa_simbench": ["The year is 2017.", "The year is 2018.", "The year is 2019.", "The year is 2021."],
    "ESS": ["The year is 2016.", "The year is 2018.", "The year is 2020.",],
    "Afrobarometer": ["The year is 2023."],
    "ISSP": ["The timeframe is 2017-2020.", "The timeframe is 2021-2024.", "The timeframe is 2017-2019.", "The timeframe is 2019-2023.", "The timeframe is 2018-2022."],
    "LatinoBarometro": ["The year is 2023.", "The year is 2016."]
}

_DATASET_des={
    "opinionqa_W26": "You are from the United States. The year is 2017.",
    "opinionqa_W29": "You are from the United States. The year is 2017.",
    "opinionqa_W32": "You are from the United States. The year is 2018.",
    "opinionqa_simbench": "You are from the United States.",
    "ESS": "You are from European country.",
    "Afrobarometer": "You are from African Country.",
    "ISSP": "",
    "LatinoBarometro": "You are from Latin American country.",
    "mp3dr": "You are from the United States.",
    "nj5dx": "You are from the United States.",
    "gss": "You are from the United States. The year is 2024."
}

def get_dataset_context(dataset_type, group_template):
    parts=[]
    if 'opinionqa' in dataset_type:
        parts.append(_DATASET_des[dataset_type])
    if dataset_type and dataset_type in _DATASET_CONTEXT:
        for prompt in _DATASET_CONTEXT[dataset_type]:
            if prompt in group_template:
                parts.append(prompt)
    if 'opinionqa' not in dataset_type and dataset_type!='ISSP':
        if 'gss' in dataset_type:
            parts.append(_DATASET_des['gss'])    
        else:
            parts.append(_DATASET_des[dataset_type])
    return parts

def _fmt_value(v: Any) -> str:
    """Make raw values readable without assuming a specific coding scheme."""
    if isinstance(v, str):
        return v.replace("_", " ").strip()
    return str(v)

def _build_group_description_from_keys(var_map: Dict[str, Any], dataset_type: str, group_template: str) -> str:
    """
    Build a compact, human-readable description using the provided keys.
    Handles both uppercase standardized keys and lowercase custom keys.
    """
    parts = []
    parts.extend(get_dataset_context(dataset_type, group_template))
    # Add dataset-specific context first
    
    # Create case-insensitive lookup
    var_map_lower = {k.lower(): (k, v) for k, v in var_map.items()}

    # Priority order for flow - handle both standard keys and custom keys
    if "cregion" in var_map_lower:
        _, v = var_map_lower["cregion"]
        parts.append(f"You are in the region: {_fmt_value(v)}.")
    
    # Handle 'cntry' (country) key
    if "cntry" in var_map_lower:
        _, v = var_map_lower["cntry"]
        parts.append(f"You are from {_fmt_value(v)}.")
    
    # Handle 'domicil' (domicile/residence) key
    if "domicil" in var_map_lower:
        _, v = var_map_lower["domicil"]
        parts.append(f"You live in {_fmt_value(v)}.")
    
    if "age" in var_map_lower:
        _, age_v = var_map_lower["age"]
        age_str = _fmt_value(age_v)
        if any(c.isdigit() for c in age_str):
            parts.append(f"Your age/age group is {age_str}.")
        else:
            parts.append(f"Your age group is {age_str}.")
    
    if "sex" in var_map_lower:
        _, v = var_map_lower["sex"]
        parts.append(f"Your gender is {_fmt_value(v)}.")
    
    if "marital" in var_map_lower:
        _, v = var_map_lower["marital"]
        parts.append(f"Marital status: {_fmt_value(v)}.")
    
    # Handle potential misspelling
    if "polsideology" in var_map_lower:
        k, v = var_map_lower["polsideology"]
        var_map["POLIDEOLOGY"] = var_map.pop(k)
        var_map_lower["polideology"] = ("POLIDEOLOGY", v)
    
    if "polideology" in var_map_lower:
        _, v = var_map_lower["polideology"]
        parts.append(f"Political ideology: {_fmt_value(v)}.")
    
    if "polparty" in var_map_lower:
        _, v = var_map_lower["polparty"]
        parts.append(f"Political party: {_fmt_value(v)}.")
    
    if "education" in var_map_lower:
        _, v = var_map_lower["education"]
        parts.append(f"Education: {_fmt_value(v)}.")
    
    if "income" in var_map_lower:
        _, v = var_map_lower["income"]
        parts.append(f"Income bracket: {_fmt_value(v)}.")
    
    if "race" in var_map_lower:
        _, v = var_map_lower["race"]
        parts.append(f"Race/ethnicity: {_fmt_value(v)}.")
    
    if "relig" in var_map_lower:
        _, v = var_map_lower["relig"]
        parts.append(f"Religion: {_fmt_value(v)}.")
    
    if "religattend" in var_map_lower:
        _, v = var_map_lower["religattend"]
        parts.append(f"Religious attendance: {_fmt_value(v)}.")
    
    # Handle any remaining keys not covered above (generic fallback)
    handled_keys = {"cregion", "cntry", "domicil", "age", "sex", "marital", 
                    "polideology", "polsideology", "polparty", "education", 
                    "income", "race", "relig", "religattend"}
    
    for key_lower, (orig_key, value) in var_map_lower.items():
        if key_lower not in handled_keys:
            parts.append(f"{orig_key.replace('_', ' ').title()}: {_fmt_value(value)}.")

    if not parts:
        return "You belong to the specified demographic group."
    return " ".join(parts)

def _normalize_feature(
    feature: Union[Tuple[Any, Any], Dict[str, Any], Tuple[str, Dict[str, Any]]],
    dataset_type: str,
    group_template: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Accepts:
      - (key, value)         -> dict {key: value} and auto description
      - dict                 -> auto description
      - (description, dict)  -> use description as-is
      - ((key1, val1), (key2, val2), ...) -> dict and auto description (NEW)
    """
    if isinstance(feature, tuple):
        # Check if it's a tuple of tuples (multiple key-value pairs)
        # This handles: (('cntry', 'Serbia'), ('domicil', 'a country village'))
        # or: (('cntry', 'Serbia'),)
        if len(feature) > 0 and all(isinstance(item, tuple) and len(item) == 2 for item in feature):
            var_map = {k: v for k, v in feature}
            desc = _build_group_description_from_keys(var_map, dataset_type, group_template)
        # Check if it's (description, dict) format
        elif len(feature) == 2 and isinstance(feature[0], str) and isinstance(feature[1], dict):
            desc, var_map = feature
            # Add dataset context if not already in description
            if dataset_type and dataset_type in _DATASET_CONTEXT:
                context = _DATASET_CONTEXT[dataset_type]
                if context not in desc:
                    desc = f"{context} {desc}"
        # Check if it's a simple (key, value) pair
        elif len(feature) == 2 and isinstance(feature[0], str):
            var_map = {feature[0]: feature[1]}
            desc = _build_group_description_from_keys(var_map, dataset_type, group_template)
        else:
            raise ValueError("Tuple must be (group_description:str, dict) or (key:str, value:any) or ((key1, val1), (key2, val2), ...).")
    elif isinstance(feature, dict):
        var_map = feature
        desc = _build_group_description_from_keys(var_map, dataset_type, group_template)
    else:
        raise ValueError("feature must be a dict, or a tuple (key,value), or (group_description:str, dict), or ((key, val), ...).")
    return desc, var_map

def generate_persona_prompt(
    feature: Union[Tuple[Any, Any], Dict[str, Any], Tuple[str, Dict[str, Any]]],
    num_persona: int,
    format: str,
    dataset_type: str,
    group_template: str
):
    """
    format: "fixed" (exactly 8 attributes) | "free"
    dataset_type: "OpinionQA" | "ESS" | "Afrobarometer" | "ISSP" | "LatinoBarometro" | None
    Fixed attributes: Gender, Age, Income, Industry, Education, Interests
    """
    group_description, group_var_map = _normalize_feature(feature, dataset_type, group_template)
    var_map_json = json.dumps(group_var_map, ensure_ascii=False)
    
    # Build context information
    context_info = ""
    
    if dataset_type and dataset_type!= 'ISSP':
        context_info = f"\n\nDataset Context: {_DATASET_des[dataset_type]}"

    base_system = f"""You are Persona Architect, a neutral and analytical demography modeler.{context_info}

Goal
- Given a group description and its variable map, generate K = {num_persona} realistic, diverse personas that represent the internal diversity of that demographic group.
- Do not predict or include survey answers, probabilities, or dataset metadata.

Guidelines
- Personas must be plausible, neutral, and representative for the specified context.
- Avoid stereotypes and sensitive assumptions.
- Ensure variation across socioeconomic and lifestyle factors relevant to the group.
- Distribute personas roughly evenly unless the group description strongly implies otherwise.

Output Requirements
- Return valid JSON matching the schema in the user message.
- Keep each short_bio concise (4–5 neutral sentences).
"""

    if format == "fixed":
        system_prompt = base_system + """
Required Attributes (exactly these 8; use these names verbatim):
1. Gender
2. Age
3. Income
4. Industry
5. Education
6. Religion
7. Marital status
8. Interests
""".strip()

        bio_instruction = ""
        if dataset_type and dataset_type in _DATASET_CONTEXT:
            bio_instruction = f"""
4) short_bio structure (MANDATORY ORDER):
   - Total 4-5 sentences: Reference information from the 8 attributes and add 1–2 neutral sentences about lifestyle/context
"""
        else:
            bio_instruction = """
4) short_bio must reference information from the 8 attributes and add 1–2 neutral sentences about lifestyle/context (total 4–5 sentences).
"""

        user_prompt = f"""group_description: {_DATASET_des[dataset_type]}
group_prompt_variable_map: {var_map_json}

Return personas in the following JSON schema (no additional fields):

{{
  "personas": [
    {{
      "group_prompt_variable_map": {var_map_json},
      "persona_id": "P_01",
      "core_constraints": {{
        "as_text": "{_DATASET_des[dataset_type]}",
        "variables": {var_map_json}
      }},
      "attributes": {{
        "Gender": "",
        "Age": 0,
        "Income": "",
        "Industry": "",
        "Education": "",
        "Religion": "",
        "Marital status": "",
        "Interests": []
      }},
      "short_bio": ""
    }}
  ]
}}

Construction Rules:
1) Interpret the group description literally; it defines the population.
2) Produce exactly {num_persona} distinct personas with IDs P_01 … P_{num_persona:02d}.
3) Each persona MUST include all 8 attributes with plausible values for the group.{bio_instruction}
4) Do not include survey questions, answers, weights, counts, or dataset metadata.
""".strip()

    elif format == "free":
        system_prompt = base_system
        
        bio_instruction = ""
        if dataset_type and dataset_type in _DATASET_CONTEXT:
            bio_instruction = f"""
4) short_bio structure (MANDATORY ORDER):
   - Total: 4-5 sentences: Implicitly reveal key attributes (e.g., gender/age/income/industry/education/religion/marital status/interests) without adding sensitive details
"""
        else:
            bio_instruction = """
4) Each short_bio must be 4–5 neutral sentences and implicitly reveal key attributes (e.g., gender/age/income/industry/education/religion/marital status/interests) without adding sensitive details.
"""

        user_prompt = f"""group_description: {_DATASET_des[dataset_type]}
group_prompt_variable_map: {var_map_json}

Return personas in the following JSON schema (no additional fields):

{{
  "personas": [
    {{
      "group_prompt_variable_map": {var_map_json},
      "persona_id": "P_01",
      "core_constraints": {{
        "as_text": "{_DATASET_des[dataset_type]}",
        "variables": {var_map_json}
      }},
      "short_bio": ""
    }}
  ]
}}

Construction Rules:
1) Interpret the group description literally; it defines the population.
2) Produce exactly {num_persona} distinct personas with IDs P_01 … P_{num_persona:02d}.{bio_instruction}
3) Do not include survey questions, answers, weights, counts, or dataset metadata.
""".strip()
    elif format == "newfree":
        context_info = ""
        if dataset_type and dataset_type != "ISSP":
            context_info = f"\n\nDataset Context: {_DATASET_des[dataset_type]}"

        system_prompt = f"""You are an AI assistant specialized in creating simulation-ready personas for survey simulation.{context_info}
These personas will be used to simulate how the same individual would answer NEW survey questions.

PRIMARY OBJECTIVE:
Create personas that maximize predictive fidelity for future survey responses across politics, values, morality, institutions, lifestyle choices, and social issues,
WHILE strictly satisfying the given demographic group constraint.

KEY PRINCIPLES:
- Think holistically: infer stable traits that drive survey answers (ideology tendencies, religiosity, trust in institutions, moral reasoning style, risk tolerance, etc.).
- Represent internal diversity: even within the same demographic group, personas should vary meaningfully in attitudes, trust, and decision rules.
- Build coherence: each persona must be realistic and internally consistent.
- Avoid stereotypes and sensitive assumptions.
- No fabricated specifics: do NOT invent a job title, city, named organizations, number of children, or specific life events unless implied by the constraint.

CRITICAL REQUIREMENTS:
1) Generate exactly K = {num_persona} distinct personas.
2) Each persona MUST satisfy the provided group_prompt_variable_map exactly.
3) Output MUST be valid JSON and MUST follow the schema provided in the user message.
4) SECOND-PERSON ONLY + no name in "short_bio":
   - Write "short_bio" in SECOND-PERSON perspective and refer to the person as "you".
   - Do NOT use any specific name.
5) "short_bio" is FREE-FORM persona_text for simulation:
   - No length constraint.
   - Focus on predictive traits, stances, conditional rules, attitude strength, and likely answer tendencies.
"""

    # NOTE: group_description comes from _normalize_feature(feature, dataset_type, group_template)
    # var_map_json = json.dumps(group_var_map, ensure_ascii=False)
        user_prompt = f"""The demographic group constraint is provided below.

group_description:
{group_description}

group_prompt_variable_map (MUST be satisfied exactly):
{var_map_json}

Please generate exactly K = {num_persona} simulation-ready personas for this SINGLE demographic group.

=== OUTPUT FORMAT (STRICT) ===
Return valid JSON with EXACTLY one top-level key: "personas".
"personas" must be a list of exactly {num_persona} persona objects.

Each persona object MUST contain exactly these keys (no extra keys inside persona):
- "persona_id"
- "group_prompt_variable_map"
- "core_constraints"
- "short_bio"

Rules:
- Produce persona_id as P_01 ... P_{num_persona:02d}.
- Every persona MUST satisfy group_prompt_variable_map exactly.
- Ensure meaningful diversity across personas while respecting the constraint.
- "short_bio" MUST be second-person ("you") and MUST NOT use any specific name.
- Do not include survey questions, answers, weights, counts, or dataset metadata.

Schema:

{{
  "personas": [
    {{
      "persona_id": "P_01",
      "group_prompt_variable_map": {var_map_json},
      "core_constraints": {{
        "as_text": {json.dumps(group_description, ensure_ascii=False)},
        "variables": {var_map_json}
      }},
      "short_bio": "..."
    }}
  ]
}}

IMPORTANT:
- JSON only. No extra text.
""".strip()
    else:
        raise ValueError('format must be "fixed" or "free".')

    return system_prompt, user_prompt
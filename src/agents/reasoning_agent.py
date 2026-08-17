"""
Agent: Reasoning Agent
Takes retrieved evidence (similar historical cases) and generates a grounded
diagnosis using Claude, citing specific evidence rather than guessing.
"""

import os
import re
from dotenv import load_dotenv
import anthropic

load_dotenv('../.env')
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))


def build_reasoning_prompt(machine_id: int, retrieval_result: dict) -> str:
    cases_text = "\n\n".join([
        f"Case {i+1} (similarity distance: {c['similarity_distance']}): {c['description']}"
        for i, c in enumerate(retrieval_result['similar_cases'][:5])
    ])

    return f"""You are a predictive maintenance analyst. Based on real sensor data and retrieved historical cases, identify the most likely failing component for this machine.

CURRENT MACHINE STATE (Machine {machine_id}):
{retrieval_result['query_used']}

STATISTICALLY MOST COMMON FAILURE for this machine's model (from historical failure-rate data): {retrieval_result['statistically_likely_component']}

RETRIEVED SIMILAR HISTORICAL CASES (real past failures, ranked by similarity):
{cases_text}

Based ONLY on the evidence above, respond in this exact format:
LIKELY_COMPONENT: [component name]
CONFIDENCE: [a number from 0 to 100]
REASONING: [2-3 sentences explaining your diagnosis, citing which specific retrieved cases support it]

Do not invent information not present in the evidence above. If the evidence is ambiguous or conflicting, say so and lower your confidence accordingly."""


def diagnose(machine_id: int, retrieval_result: dict) -> dict:
    """
    The agent's main function: builds the prompt, calls Claude, and parses
    the structured response into a usable dict.
    """
    prompt = build_reasoning_prompt(machine_id, retrieval_result)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.content[0].text

    # Parse the structured response
    component_match = re.search(r'LIKELY_COMPONENT:\s*(\S+)', raw_text)
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', raw_text)
    reasoning_match = re.search(r'REASONING:\s*(.+)', raw_text, re.DOTALL)

    return {
        'machine_id': machine_id,
        'likely_component': component_match.group(1) if component_match else None,
        'confidence': int(confidence_match.group(1)) if confidence_match else None,
        'reasoning': reasoning_match.group(1).strip() if reasoning_match else raw_text,
        'raw_response': raw_text
    }
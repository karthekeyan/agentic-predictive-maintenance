"""
Agent: Reasoning Agent
Takes retrieved evidence (similar historical cases) and generates a grounded
diagnosis using Claude, citing specific evidence rather than guessing.
Respects 'insufficient_data' status from the Case Retrieval Agent - skips
diagnosis rather than reasoning over missing/invalid data (see Day 4 known issue).
"""

import os
import re
from dotenv import load_dotenv
import anthropic

load_dotenv('../.env')
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def build_reasoning_prompt(machine_id: int, retrieval_result: dict, classifier_result: dict = None) -> str:
    cases_text = "\n\n".join([
        f"Case {i+1} (similarity distance: {c['similarity_distance']}): {c['description']}"
        for i, c in enumerate(retrieval_result['similar_cases'][:5])
    ])

    classifier_text = "Not available."
    if classifier_result and classifier_result.get('status') == 'ok':
        classifier_text = (
            f"A trained machine learning model (XGBoost, validated with 0.82-0.94 recall "
            f"per component on a time-aware historical test) predicts: "
            f"'{classifier_result['predicted_label']}' "
            f"with {classifier_result['probability']*100:.2f}% confidence. "
            f"This model was trained specifically to predict failure within a 24-hour window "
            f"using real sensor, error, and maintenance history - treat this as strong evidence, "
            f"particularly when it disagrees with the statistical baseline below."
        )

    return f"""You are a predictive maintenance analyst. Based on real sensor data, a trained prediction model, and retrieved historical cases, identify the most likely failing component for this machine - or whether no failure is imminent.
CURRENT MACHINE STATE (Machine {machine_id}):
{retrieval_result['query_used']}
TRAINED MODEL PREDICTION (validated on historical data):
{classifier_text}
STATISTICALLY MOST COMMON FAILURE for this machine's model (general historical base rate, weaker evidence than the trained model prediction above): {retrieval_result['statistically_likely_component']}
RETRIEVED SIMILAR HISTORICAL CASES (real past failures, ranked by similarity):
{cases_text}
Based on the evidence above, respond in this exact format:
LIKELY_COMPONENT: [component name, or "none" if the trained model predicts no imminent failure and no other evidence contradicts it]
CONFIDENCE: [a number from 0 to 100]
REASONING: [2-3 sentences explaining your diagnosis. If the trained model's prediction differs from the statistical baseline or retrieved cases, explain how you weighed the evidence and why.]
The trained model prediction is generally the strongest evidence, since it was specifically validated for this task. Do not invent information not present in the evidence above. When referencing the trained model's confidence percentage in your reasoning, quote the exact figure given above (e.g. '99.96%') rather than rounding it to a whole number."""


def diagnose(machine_id: int, retrieval_result: dict, classifier_result: dict = None) -> dict:
    """
    The agent's main function: builds the prompt, calls Claude, and parses
    the structured response into a usable dict. Skips reasoning entirely if
    the Case Retrieval Agent flagged insufficient/invalid data.

    classifier_result: optional output from the trained ML model
    (predict_component_failure_24h) - if provided, included as evidence
    in the prompt.
    """
    if retrieval_result.get('status') == 'insufficient_data':
        return {
            'machine_id': machine_id,
            'likely_component': None,
            'confidence': None,
            'reasoning': f"Diagnosis skipped: {retrieval_result.get('reason')}",
            'raw_response': None
        }

    prompt = build_reasoning_prompt(machine_id, retrieval_result, classifier_result)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.content[0].text

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
"""
Agent: Decision & Routing Agent
Takes the diagnosis confidence and decides whether to auto-route the recommendation
directly to the maintenance team, or escalate it for human engineer review.
Confidence threshold is a configurable policy parameter, not derived from data.
"""

# Configurable threshold - a design/policy choice, documented as such
CONFIDENCE_THRESHOLD_AUTO = 70
CONFIDENCE_THRESHOLD_REVIEW = 50


def route_decision(machine_id: int, diagnosis_result: dict, recommendation_result: dict) -> dict:
    """
    Decides the routing path based on the Reasoning Agent's confidence score.

    - confidence >= 70: auto-route to maintenance team
    - 50 <= confidence < 70: flag for engineer review
    - confidence < 50 or None: escalate with high priority (low/no confidence)
    """
    confidence = diagnosis_result.get('confidence')

    if confidence is None:
        routing = 'escalate_urgent'
        reason = 'No diagnosis available (insufficient data or inconclusive evidence).'
    elif confidence >= CONFIDENCE_THRESHOLD_AUTO:
        routing = 'auto_route'
        reason = f'High confidence ({confidence}%) - evidence strongly supports this diagnosis.'
    elif confidence >= CONFIDENCE_THRESHOLD_REVIEW:
        routing = 'engineer_review'
        reason = f'Moderate confidence ({confidence}%) - recommend human verification before acting.'
    else:
        routing = 'escalate_urgent'
        reason = f'Low confidence ({confidence}%) - evidence is weak or conflicting, needs expert review.'

    return {
        'machine_id': machine_id,
        'routing_decision': routing,
        'reason': reason,
        'component': recommendation_result.get('component'),
        'recommendation': recommendation_result.get('recommendation'),
        'confidence': confidence
    }
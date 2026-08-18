"""
LangGraph Pipeline: wires all 7 agents into one automated flow.
State flows through each node in sequence; ends with a conditional branch
based on the Decision & Routing Agent's confidence-based decision.
"""

import sys
import operator
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

sys.path.append('src/agents')
sys.path.append('src/tools')

from telemetry_health_agent import check_machine_health
from prioritization_agent import prioritize_machine
from case_retrieval_agent import find_similar_past_cases
from reasoning_agent import diagnose
from recommendation_agent import recommend_action
from decision_routing_agent import route_decision


class PipelineState(TypedDict):
    machine_id: int
    model: str
    as_of: object
    telemetry_scored: object
    failures_df: object
    maint_df: object
    machines_df: object
    mtbf_table: object
    health_result: Optional[dict]
    priority_result: Optional[dict]
    retrieval_result: Optional[dict]
    diagnosis_result: Optional[dict]
    recommendation_result: Optional[dict]
    routing_result: Optional[dict]


def telemetry_health_node(state: PipelineState) -> dict:
    result = check_machine_health(state['machine_id'], state['telemetry_scored'], as_of=state['as_of'])
    return {'health_result': result}


def prioritization_node(state: PipelineState) -> dict:
    result = prioritize_machine(state['machine_id'], state['model'], state['health_result'], state['mtbf_table'])
    return {'priority_result': result}


def case_retrieval_node(state: PipelineState) -> dict:
    result = find_similar_past_cases(state['machine_id'], state['telemetry_scored'], state['as_of'], state['priority_result'])
    return {'retrieval_result': result}


def reasoning_node(state: PipelineState) -> dict:
    result = diagnose(state['machine_id'], state['retrieval_result'])
    return {'diagnosis_result': result}


def recommendation_node(state: PipelineState) -> dict:
    result = recommend_action(state['machine_id'], state['model'], state['diagnosis_result'],
                                state['failures_df'], state['maint_df'], state['machines_df'])
    return {'recommendation_result': result}


def decision_routing_node(state: PipelineState) -> dict:
    result = route_decision(state['machine_id'], state['diagnosis_result'], state['recommendation_result'])
    return {'routing_result': result}


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node('telemetry_health', telemetry_health_node)
    graph.add_node('prioritization', prioritization_node)
    graph.add_node('case_retrieval', case_retrieval_node)
    graph.add_node('reasoning', reasoning_node)
    graph.add_node('recommendation', recommendation_node)
    graph.add_node('decision_routing', decision_routing_node)

    graph.set_entry_point('telemetry_health')
    graph.add_edge('telemetry_health', 'prioritization')
    graph.add_edge('prioritization', 'case_retrieval')
    graph.add_edge('case_retrieval', 'reasoning')
    graph.add_edge('reasoning', 'recommendation')
    graph.add_edge('recommendation', 'decision_routing')
    graph.add_edge('decision_routing', END)

    return graph.compile()
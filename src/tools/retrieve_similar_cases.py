"""
Tool: retrieve_similar_cases
Searches the Knowledge Library (ChromaDB) for similar real historical failure cases.
Combines structured metadata filtering (dominant_sensor, and optionally component)
with semantic text search. Pure text search was found to miss numeric magnitude
(e.g., "voltage 12%" vs "voltage 1%" read as similar text) - dominant_sensor is
computed directly from the query's own numbers to fix this (Day 6 fix).
"""

import chromadb


def get_collection(db_path="../data/processed/chroma_db", collection_name="failure_cases"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)


def compute_dominant_sensor(vibration_dev, volt_dev, pressure_dev, rotate_dev):
    """Determines which sensor has the largest absolute deviation."""
    devs = {
        'vibration': abs(vibration_dev),
        'volt': abs(volt_dev),
        'pressure': abs(pressure_dev),
        'rotate': abs(rotate_dev)
    }
    return max(devs, key=devs.get)


def retrieve_similar_cases(query_description: str, component_filter: str = None,
                             dominant_sensor_filter: str = None, n_results: int = 3) -> list:
    """
    The actual tool function an agent calls.

    Args:
        query_description: plain-language description of the current situation
        component_filter: if known, restrict search to this component only
        dominant_sensor_filter: if known, restrict search to cases with this dominant sensor
        n_results: how many similar cases to return

    Returns:
        list of dicts, each with case_id, description, similarity_distance, metadata
    """
    collection = get_collection()

    query_params = {"query_texts": [query_description], "n_results": n_results}

    where_conditions = []
    if component_filter:
        where_conditions.append({"component": component_filter})
    if dominant_sensor_filter:
        where_conditions.append({"dominant_sensor": dominant_sensor_filter})

    if len(where_conditions) == 1:
        query_params["where"] = where_conditions[0]
    elif len(where_conditions) > 1:
        query_params["where"] = {"$and": where_conditions}

    results = collection.query(**query_params)

    matches = []
    for i in range(len(results['ids'][0])):
        matches.append({
            'case_id': results['ids'][0][i],
            'description': results['documents'][0][i],
            'similarity_distance': round(results['distances'][0][i], 4),
            'metadata': results['metadatas'][0][i]
        })
    return matches
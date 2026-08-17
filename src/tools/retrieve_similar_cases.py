"""
Tool: retrieve_similar_cases
Searches the Knowledge Library (ChromaDB) for similar real historical failure cases.
Combines structured metadata filtering (component) with semantic text search
for relevant results — pure text search alone was found to miss numeric relevance.
"""

import chromadb


def get_collection(db_path="../data/processed/chroma_db", collection_name="failure_cases"):
    """Connects to the existing ChromaDB Knowledge Library."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)


def retrieve_similar_cases(query_description: str, component_filter: str = None, n_results: int = 3) -> list:
    """
    The actual tool function an agent calls.

    Args:
        query_description: plain-language description of the current situation
        component_filter: if known, restrict search to this component only (e.g., "comp4")
        n_results: how many similar cases to return

    Returns:
        list of dicts, each with case_id, description, similarity_distance, metadata
    """
    collection = get_collection()

    query_params = {"query_texts": [query_description], "n_results": n_results}
    if component_filter:
        query_params["where"] = {"component": component_filter}

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
import json
import os
from typing import Any, Dict, List, cast

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("MODEL", "gemma4:latest")


QUERY_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "complexity": {
            "type": "string",
            "enum": ["simple", "complex"]
        },
        "confidence": {
            "type": "number"
        },
        "reason": {
            "type": "string"
        },
        "subqueries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "purpose": {
                        "type": "string",
                        "enum": [
                            "entity_lookup",
                            "relationship_lookup",
                            "detail_lookup",
                            "count_lookup",
                            "status_lookup",
                            "validation_lookup",
                            "metric_lookup",
                            "other"
                        ]
                    }
                },
                "required": ["query", "purpose"],
                "additionalProperties": False
            }
        }
    },
    "required": ["complexity", "confidence", "reason", "subqueries"],
    "additionalProperties": False
}


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise


def normalize_analysis(query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    complexity = str(analysis.get("complexity", "simple")).lower()

    if complexity not in {"simple", "complex"}:
        complexity = "simple"

    try:
        confidence = float(analysis.get("confidence", 0.5))
    except Exception:
        confidence = 0.5

    confidence = max(0.0, min(1.0, confidence))

    subqueries = analysis.get("subqueries", [])

    if not isinstance(subqueries, list):
        subqueries = []

    clean_subqueries: List[Dict[str, str]] = []

    for item in subqueries:
        if not isinstance(item, dict):
            continue

        sq = str(item.get("query", "")).strip()
        purpose = str(item.get("purpose", "other")).strip()

        if sq:
            clean_subqueries.append(
                {
                    "query": sq,
                    "purpose": purpose or "other",
                }
            )

    if complexity == "simple":
        clean_subqueries = [
            {
                "query": query,
                "purpose": "other",
            }
        ]

    if complexity == "complex" and not clean_subqueries:
        clean_subqueries = [
            {
                "query": query,
                "purpose": "other",
            }
        ]

    # Keep decomposition small.
    clean_subqueries = clean_subqueries[:4]

    return {
        "complexity": complexity,
        "confidence": confidence,
        "reason": str(analysis.get("reason", "")).strip(),
        "subqueries": clean_subqueries,
    }
    
    
def analyze_query_with_gemini(
    query: str,
    model: str | None = None,
) -> Dict[str, Any]:
    from google import genai

    model = model or MODEL
    client = genai.Client()

    prompt = f"""
You are a query decomposition assistant for a SQL/API data agent.

Your job:
- Decide whether the user query is simple or complex.
- If simple, return one subquery equal to the original query.
- If complex, split it into 2 to 4 smaller subqueries that can act as independent queries on its own.
- Do not write SQL.
- Do not invent API endpoints.
- Do not answer the user query.
- Keep subqueries short and retrieval-friendly.

Return only JSON.

Query:
{query}
""".strip()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_json_schema": QUERY_ANALYSIS_SCHEMA,
        },
    )

    raw = response.text or ""
    analysis = safe_json_loads(raw)

    return normalize_analysis(query, analysis)


def analyze_query_with_ollama(
    query: str,
    model: str | None = None,
) -> Dict[str, Any]:
    import ollama

    model = model or MODEL

    messages = [
        {
            "role": "system",
            "content": """
You are a query decomposition assistant for a SQL/API data agent.

Your job:
- Decide whether the user query is simple or complex.
- If simple, return one subquery equal to the original query.
- If complex, split it into 2 to 4 smaller subqueries that can act as independent queries on its own.
- Do not write SQL.
- Do not invent API endpoints.
- Do not answer the user query.
- Keep subqueries short and retrieval-friendly.

A complex query usually needs multiple pieces of evidence, such as:
entity lookup + relationship lookup + details/counts/metrics/validation.

Return only JSON.
""".strip(),
        },
        {
            "role": "user",
            "content": f"""
Analyze this query:

{query}
""".strip(),
        },
    ]

    response = ollama.chat(
        model=model,
        messages=messages,
        format=QUERY_ANALYSIS_SCHEMA,
        options={
            "temperature": 0.0,
            "num_ctx": 32768,
        },
        stream=False,
    )

    if isinstance(response, dict):
        raw = response["message"]["content"]
    else:
        raw = response.message.content

    raw = str(raw) if raw is not None else ""
    analysis = safe_json_loads(raw)

    return normalize_analysis(query, analysis)


def analyze_query(
    query: str,
    use_llm: bool = True,
    model: str | None = None,
    provider: str | None = None,
) -> Dict[str, Any]:
    if not use_llm:
        return normalize_analysis(
            query,
            {
                "complexity": "simple",
                "confidence": 1.0,
                "reason": "LLM analyzer disabled.",
                "subqueries": [
                    {
                        "query": query,
                        "purpose": "other",
                    }
                ],
            },
        )

    provider = (provider or LLM_PROVIDER).lower().strip()

    try:
        if provider == "gemini":
            return analyze_query_with_gemini(query=query, model=model)

        return analyze_query_with_ollama(query=query, model=model)

    except Exception as e:
        return normalize_analysis(
            query,
            {
                "complexity": "simple",
                "confidence": 0.0,
                "reason": f"Analyzer failed, fallback to simple query. Error: {e}",
                "subqueries": [
                    {
                        "query": query,
                        "purpose": "other",
                    }
                ],
            },
        )
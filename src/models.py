from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolStep:
    step: int
    action: str

    sql: Optional[str] = None

    api_method: Optional[str] = None
    api_url: Optional[str] = None
    api_params: Optional[Dict[str, Any]] = None

    status: Optional[str] = None


@dataclass
class ExampleRecord:
    query: str
    answer: str
    gold_sql: str
    gold_api: List[str]
    trace: List[ToolStep]


@dataclass
class ParsedApiCall:
    raw: str
    method: str
    path: str
    endpoint: str
    params: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None


@dataclass
class IndexedExample:
    query: str
    route: str
    intent: str
    domain: str

    sql_tables: List[str] = field(default_factory=list)
    api_calls: List[ParsedApiCall] = field(default_factory=list)
    api_endpoints: List[str] = field(default_factory=list)
    api_params: List[str] = field(default_factory=list)

    gold_sql: str = ""
    gold_api: List[str] = field(default_factory=list)

    search_text: str = ""
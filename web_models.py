from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr


class SearchRequest(BaseModel):
    queries: List[str]
    sources: Optional[List[str]] = Field(default_factory=lambda: ["openalex"])
    end_date: Optional[str] = ""
    max_workers: Optional[int] = 3
    batch_size: Optional[int] = 10
    google_serper_key: Optional[SecretStr] = None
    use_advanced_search: Optional[bool] = True
    max_depth: Optional[int] = 1
    relevance_doc_num: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.5


class SearchResponse(BaseModel):
    status: str
    total_papers: int
    query_results: Dict[str, List[Dict[str, Any]]]
    all_papers: Dict[str, Dict[str, Any]]
    query_source_map: Dict[str, str]
    search_tree: Optional[Dict[str, Any]] = None

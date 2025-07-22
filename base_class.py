from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class SearchConfig:
    """Configuration for academic search

    Attributes:
        max_depth (int): Maximum depth for recursive search.
        max_docs (int): Maximum number of documents to retrieve.
        similarity_threshold (float): Minimum similarity score to consider a document relevant.
        high_score_threshold (float): Threshold for high relevance scores.
        query_batch_size (int): Number of queries to process in a single batch.
        reference_batch_size (int): Number of references to process in a single batch.
        save_to_db (bool): Whether to save results to the database.
        do_rerank (bool): Whether to rerank the search results.
    """

    max_depth: int = 1
    max_docs: int = 50
    similarity_threshold: float = 0.6
    high_score_threshold: float = 0.75
    query_batch_size: int = 4
    reference_batch_size: int = 4
    save_to_db: bool = True
    do_rerank: bool = False


@dataclass
class SearchResult:
    """Container for search results from a single source

    Attributes:
        source (str): The source of the search results (e.g., ArXiv, PubMed).
        papers (Dict[str, Any]): A dictionary containing paper metadata.
        query2paper (Dict[str, List[str]]): Mapping of queries to the list of relevant paper IDs.
        arxiv_ids (Optional[Set[str]]): Set of ArXiv IDs for the retrieved papers.
        keyword (Optional[str]): The keyword used for the search.
        raw_query (Optional[Any]): The raw query used for the search.
        error (Optional[str]): Error message, if any occurred during the search.
        extra (Optional[Any]): Additional information or metadata.
    """

    source: str
    papers: Dict[str, Any]
    query2paper: Dict[str, List[str]] = field(default_factory=dict)
    arxiv_ids: Optional[Set[str]] = None
    keyword: Optional[str] = None
    raw_query: Optional[Any] = None
    error: Optional[str] = None
    extra: Optional[Any] = None

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

from typing import List, Optional, Callable, Dict


class SearchNode:
    """
    Represents a node in the academic search tree.
    Each node contains:
    - A search query
    - Relevant and irrelevant documents
    - References
    - Child nodes for query expansion
    """

    def __init__(
        self,
        query_str: str = "",
        query_weight: float = 1.0,
        status: str = "INIT",
        parent: Optional["SearchNode"] = None,
        source: List[str] = None,
        raw_query: str= "",
        **attrs
    ):
        # Query information
        self.query_str = query_str  # Search query string
        self.query_weight = query_weight  # Query importance weight
        self.status = status  # Node status (INIT, SEARCH, END, etc.)
        self.searched_queries = set()
        self.doc_used_to_gen_query = set()
        self.source = source or []  # Search channels for this query
        self.raw_query = raw_query
        # Search results
        self.docs = []  # Relevant documents
        self.irrelevant_docs = []  # Irrelevant documents
        self.relevance_refs = []  # Relevant reference documents
        self.irrelevant_refs = []  # Irrelevant reference documents
        self.references = []  # References from relevant docs
        self.children = []  # Child query nodes
        self.searched_docs = dict()
        self.reranked_top_docs = []
        self.hight_relevance_docs = set()  # 高相关度doc列表
        self.cal_sim_docs = dict()  # 计算过分数的doc
        # Node metadata
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        self.extra = attrs.get("extra", {})  # Additional attributes

    def convert_to_dict(self) -> dict:
        """Convert node to dictionary format for serialization"""
        self.extra["searched_docs"] = self.searched_docs
        self.extra["searched_queries"] = list(self.searched_queries)
        self.extra["hight_relevance_docs"] = list(self.hight_relevance_docs)
        return {
            "search_query": self.query_str,
            "query_weight": self.query_weight,
            "children": [child.convert_to_dict() for child in self.children],
            "docs": [dict(doc) for doc in self.docs],
            "irrelevant_docs": [dict(doc) for doc in self.irrelevant_docs],
            "references": [ref for ref in self.references],
            "depth": self.depth,
            "search_status": self.status,
            "source": self.source,
            "extra": self.extra,
        }

    def sort_doc(self) -> None:
        """Sort documents by similarity score in descending order"""
        self.docs = sorted(self.docs, key=lambda x: x["sim_score"], reverse=True)
        self.irrelevant_docs = sorted(
            self.irrelevant_docs, key=lambda x: x["sim_score"], reverse=True
        )

    def add_searched_query(self, queries):
        for query in queries:
            self.searched_queries.add(query)

    def add_signature_for_doc(self, docs):
        for doc in docs:
            if "paper_id" not in doc and "arxivId" in doc:
                doc["paper_id"] = doc["arxivId"]
            if doc["paper_id"] in self.searched_docs:
                self.searched_docs[doc["paper_id"]].update(doc)
            else:
                self.searched_docs[doc["paper_id"]] = doc

    def add_child(self, child: "SearchNode") -> None:
        """Add a child node"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    @property
    def has_results(self) -> bool:
        """Check if node has any search results"""
        return len(self.docs) > 0

    @property
    def total_docs(self) -> int:
        """Get total number of documents (relevant + irrelevant)"""
        return len(self.docs) + len(self.irrelevant_docs)
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

from collections import deque
from datetime import datetime, timedelta
from global_config import *
from graphviz import Digraph
from instruction import *
from local_db_v2 import db_path, ArxivDatabase
from log import logger
from search_engine import AcademicTreeSearchEngine
from search_node import SearchNode
from typing import List, Dict, Optional
import json
import random
import re
import time
import tqdm
import traceback
from rerank import Reranker


class AcademicSearchTree:
    """
    Tree-based academic paper search engine that:
    1. Performs iterative search using query expansion
    2. Filters results by relevance score
    3. Explores paper references for deeper search

    The search uses a tree structure where:
    - Root node represents the initial query
    - First level nodes are expanded queries
    - Subsequent levels are queries generated from document context

    The search process involves:
    1. Query expansion: Generate alternative formulations of the initial query
    2. Document retrieval: Search for papers matching each query
    3. Relevance calculation: Score papers based on relevance to the initial query
    4. Reference exploration: Find additional papers by following citations
    5. Query generation: Create new queries based on retrieved documents

    Search stops when either:
    - Enough highly relevant papers are found (> max_docs)
    - Maximum search depth is reached
    """

    def __init__(
        self,
        max_depth: int = 1,
        max_docs: int = 50,
        similarity_threshold: float = 0.6,
        search_engine=None,
    ):
        # Search parameters
        self.max_depth = max_depth
        self.max_docs = max_docs
        self.sim_threshold = similarity_threshold
        # Search state
        self.root = SearchNode()
        # Current search metadata
        self.search_time = None
        self.current_date = None
        self.high_score_thresh = 0.75
        self.search_engine = AcademicTreeSearchEngine()
        self.reranker = Reranker()

    def _cleanup_resources(self):
        """Perform cleanup of resources after search is completed"""
        try:
            # Release any resources that need explicit cleanup
            if (
                hasattr(self.search_engine, "_emd_model")
                and self.search_engine._emd_model
            ):
                # If there's a cleanup method available, call it
                if hasattr(self.search_engine._emd_model, "cleanup"):
                    self.search_engine._emd_model.cleanup()

            # Clear large in-memory data structures
            if hasattr(self.root, "cal_sim_docs"):
                self.root.cal_sim_docs.clear()

            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def meet_stop_condition(self, current_depth=0):
        """
        Determines if the search should stop based on relevance threshold and depth.

        The search stops when either:
        1. We've found enough highly relevant documents (> self.max_docs)
        2. We've reached the maximum search depth

        Args:
            current_depth: Current search depth

        Returns:
            Boolean indicating whether to stop the search
        """
        # Identify highly relevant documents
        relevance_docs = set(
            doc_id
            for doc_id, doc_info in self.root.searched_docs.items()
            if doc_info.get("sim_score", -1) > self.high_score_thresh
        )
        self.root.hight_relevance_docs = relevance_docs

        # Log current search state
        logger.info(
            f"Query: {self.user_query[:30]}... | "
            f"Depth: {current_depth}/{self.max_depth} | "
            f"Total docs: {len(self.root.searched_docs)} | "
            f"Highly relevant: {len(relevance_docs)}/{self.max_docs} | "
            f"Normal relevance: {len(self.root.docs)}"
        )

        # Check stopping conditions
        depth_exceeded = current_depth >= self.max_depth
        enough_relevant_docs = len(relevance_docs) > self.max_docs

        if enough_relevant_docs:
            logger.info(
                f"Stopping search: Found {len(relevance_docs)} highly relevant docs (threshold: {self.max_docs})"
            )
        elif depth_exceeded:
            logger.info(f"Stopping search: Reached maximum depth {self.max_depth}")

        # return enough_relevant_docs or depth_exceeded
        return depth_exceeded

    def _save_id2info(self, id2docs):
        """
        Save document information to local database with error handling.

        Args:
            id2docs: Dictionary mapping document IDs to document information
        """
        if not id2docs:
            logger.info("No documents to save to local DB")
            return

        logger.info(f"🤔 Saving {len(id2docs)} documents to local database")
        start_time = time.time()
        success_count = 0

        try:
            with ArxivDatabase(db_path) as db:
                for arxiv, info in id2docs.items():
                    if info.get("source", "") == "Search From Local":
                        continue
                    try:
                        # Create new info dict, removing keys containing "sim_score"
                        cleaned_info = {
                            k: v for k, v in info.items() if "sim_score" not in k
                        }
                        db.update_or_insert(arxiv, cleaned_info)
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to save document {arxiv}: {str(e)}")

            logger.info(
                f"😁 Saved {success_count}/{len(id2docs)} documents to local DB in {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Database operation failed: {traceback.format_exc()}")

    def query_fusion(self):
        """
        Expands the initial query into multiple search queries.
        Returns a list of expanded queries without creating search nodes.
        """
        logger.info("Running step1: query expansion")
        try:
            query_node_relations = {}
            expanded_queries_info = self.search_engine.expand_query(self.root.query_str)
            logger.info(f"expanded_queries_info: {expanded_queries_info}")
            expanded_queries_info["QUERY_NUM_PRUNED"] = QUERY_NUM_PRUNED
            self.root.extra["expanded_queries_info"] = expanded_queries_info
            expanded_queries = expanded_queries_info["expanded_queries"]

            # Add the original query to the expanded list
            if self.root.query_str not in expanded_queries:
                expanded_queries = [self.root.query_str] + expanded_queries

            for query in expanded_queries:
                node = SearchNode(
                    query_str=query,
                    status="START",
                )
                query_node_relations[query] = {
                    "own_node": node,
                    "parent_node": self.root,
                }

            return expanded_queries, query_node_relations
        except Exception as e:
            logger.error(f"Query fusion failed: {traceback.format_exc()}")
            # Fallback to just the original query if expansion fails
            return [self.root.query_str], {}

    def query_level_search(
        self,
        expanded_queries,
        query_node_relations,
        next_level,
        search_date,
        current_depth,
    ):
        """
        Performs search for all queries at the current level and processes the results.
        Creates search nodes based on the query results.

        Args:
            expanded_queries: List of expanded query strings
            next_level: List to populate with nodes for the next level
            search_date: End date for search
            current_depth: Current search depth

        Returns:
            Tuple of (created search nodes, next level nodes)
        """
        if not expanded_queries:
            logger.warning("No queries provided for query_level_search")
            return [], next_level

        logger.info(
            f"Running query_level_search with {len(expanded_queries)} queries at depth {current_depth}"
        )
        logger.info(f"expanded_queries: {expanded_queries}")
        logger.info(f"query_node_relations: {query_node_relations}")

        # Determine sources based on expanded_queries_info if available
        if hasattr(self.root, "extra") and "expanded_queries_info" in self.root.extra:
            expanded_info = self.root.extra["expanded_queries_info"]
            if "suitable_sources" in expanded_info:
                sources = expanded_info["suitable_sources"]
                logger.info(f"Using suitable sources from query expansion: {sources}")
            else:
                sources = SEARCH_ROUTE
                logger.info(
                    f"No suitable_sources found in expanded_queries_info, using default: {sources}"
                )
        else:
            # Use default sources based on depth
            sources = SEARCH_ROUTE if current_depth == 1 else ["arxiv"]
            logger.info(f"Using depth-based sources: {sources}")

        if "arxiv" not in sources:
            sources.insert(0, "arxiv")
            logger.info(f"Adding 'arxiv' to sources: {sources}")

        if current_depth > 1:
            logger.info(f"current_depth: {current_depth}, only search arxiv")
            sources = ["arxiv"]

        try:
            # Execute batch search across multiple sources
            batch_result, id2docs, query_source_map, query_keywords2raw = (
                self.search_engine.search_papers_mroute(
                    expanded_queries,
                    end_date=search_date,
                    searched_docs=self.root.searched_docs,
                    sources=sources,
                )
            )

            # Update search state
            # only store the query string, do not store the keywords
            self.root.searched_queries.update(expanded_queries)
            self.root.add_signature_for_doc(list(id2docs.values()))

            # Save to local DB if enabled
            if SAVE_ID2DOCS:
                self._save_id2info(id2docs)

            # Create nodes based on search results
            current_level_node = []

            for query, retrival_docs in batch_result.items():
                query_source = query_source_map[query]

                if not retrival_docs:
                    status = "Failed"
                else:
                    status = "Finshed"

                logger.info(f"{query} --- query_source: {query_source}: {status}")

                if query in query_node_relations:
                    assert (
                        query_keywords2raw[query] == query
                    ), f"query_keywords2raw: {query_keywords2raw[query]} != {query}"
                    node = query_node_relations[query]["own_node"]
                    assert (
                        node.query_str == query
                    ), f"node.query_str: {node.query_str} != {query}"
                    parent_node = query_node_relations[query]["parent_node"]
                    node.status = status
                    node.source = query_source
                    node.parent = parent_node
                    node.raw_query = query
                    parent_node.add_child(node)
                    current_level_node.append(node)
                else:
                    raw_query = query_keywords2raw[query]
                    parent_node = query_node_relations[raw_query]["parent_node"]
                    new_node = SearchNode(
                        query_str=query,
                        status=status,
                        parent=parent_node,
                        source=query_source,
                        raw_query=raw_query,
                    )
                    parent_node.add_child(new_node)
                    current_level_node.append(new_node)
            # Process each node's search results
            logger.info(f"current level node: {len(current_level_node)}")
            valid_doc_count = 0
            rel_doc_count = 0
            for node in tqdm.tqdm(
                current_level_node,
                total=len(current_level_node),
                desc="Processing search results",
            ):
                try:
                    raw_docs = batch_result.get(node.query_str, [])

                    if not raw_docs:
                        node.status = "Failed"
                        next_level.append(node)
                        continue

                    # Filter for docs with required fields
                    valid_docs = [
                        doc
                        for doc in raw_docs
                        if doc.get("title", "")
                        and doc.get("abstract", "")
                        and doc.get("paper_id", doc.get("arxivId"))
                        not in self.root.cal_sim_docs
                    ]

                    logger.info(
                        f"raw_docs: {len(raw_docs)}, valid_docs: {len(valid_docs)}"
                    )

                    valid_doc_count += len(valid_docs)

                    if not valid_docs:
                        logger.warning(
                            f"No valid docs found for query: {node.query_str}"
                        )
                        node.status = "NO Valid Docs"
                        continue

                    # Calculate similarity scores
                    relevant_docs, irrelevance_docs = (
                        self.search_engine.calculate_similarity(
                            query=self.root.query_str,
                            docs=valid_docs,
                            search_time=self.search_date,
                            score_thresh=self.sim_threshold,
                            source=f"from retrieval, query: [{node.source}] -- {node.query_str}",
                        )
                    )

                    # Update document collections
                    self.root.add_signature_for_doc(relevant_docs + irrelevance_docs)
                    self.root.cal_sim_docs.update(
                        {
                            one.get("paper_id", one.get("arxivId")): one
                            for one in relevant_docs + irrelevance_docs
                        }
                    )
                    # Update node status based on search results
                    if len(relevant_docs) == 0 and len(irrelevance_docs) == 0:
                        node.status = "Failed"
                        if node not in next_level:
                            next_level.append(node)

                    elif len(relevant_docs) > 0:
                        node.status = "Expand"
                        rel_doc_count += len(relevant_docs)
                    else:
                        node.status = "NO Relevance"

                    node.docs.extend(relevant_docs)
                    node.irrelevant_docs.extend(irrelevance_docs)

                except Exception as e:
                    logger.error(
                        f"Error processing node {node.query_str}: {traceback.format_exc()}"
                    )
                    node.status = "Error"

            logger.info(
                f"Query level search completed: {valid_doc_count} valid docs, {rel_doc_count} relevant docs, failed node mum: {len(next_level)}"
            )
            return current_level_node, next_level

        except Exception as e:
            logger.error(f"Batch search failed: {traceback.format_exc()}")
            return [], next_level

    def reference_level_search(self, level_node):
        """
        Explores references of retrieved documents to find additional relevant papers.

        Args:
            level_node: List of nodes whose documents' references will be explored

        Returns:
            Updated list of nodes with references added
        """
        if not level_node:
            logger.warning("No nodes provided for reference_level_search")
            return level_node

        logger.info(f"Running reference search on {len(level_node)} nodes")

        try:
            # Collect all relevant documents for reference exploration
            all_rel_docs = []
            for node in level_node:
                # Use both relevant and irrelevant docs for reference expansion
                expand_docs = node.docs + node.irrelevant_docs
                for doc in expand_docs:
                    all_rel_docs.append([node, doc])

            # Sort by similarity score (most relevant first)
            all_rel_docs = sorted(
                all_rel_docs, key=lambda x: x[1].get("sim_score", 0), reverse=True
            )

            # Limit to top documents to prevent excessive exploration
            docs_to_expand = min(len(all_rel_docs), DOCS_TO_EXPAND)
            all_rel_docs = all_rel_docs[:docs_to_expand]
            logger.info(
                f"Selected {docs_to_expand} documents for reference exploration"
            )

            # Process documents in batches
            batch_size = 2
            ref_count = 0
            relevant_ref_count = 0

            # Track which papers we've already processed to avoid duplicates
            processed_paper_ids = set()

            for index in tqdm.tqdm(
                range(0, len(all_rel_docs), batch_size),
                total=len(all_rel_docs) // batch_size
                + (1 if len(all_rel_docs) % batch_size else 0),
                desc="Reference Level Search",
            ):
                # Break early if stopping condition is met
                if self.meet_stop_condition():
                    logger.info(
                        "Stopping reference search early: stopping condition met"
                    )
                    # Mark remaining nodes as stopped
                    for i in range(index, len(all_rel_docs), batch_size):
                        if i < len(all_rel_docs):
                            node, _ = all_rel_docs[i]
                            node.status = "STOP"
                    return level_node

                # Process the current batch
                start_idx = index
                end_idx = min(index + batch_size, len(all_rel_docs))
                batch = all_rel_docs[start_idx:end_idx]

                for node, doc in batch:
                    try:
                        # Skip if we've already processed this paper
                        paper_id = doc.get("paper_id", "")
                        if paper_id in processed_paper_ids:
                            continue
                        processed_paper_ids.add(paper_id)

                        # Get document info from search state
                        doc_info = self.root.searched_docs.get(doc["paper_id"])
                        if not doc_info:
                            logger.warning(
                                f"Document info not found for {doc['paper_id']}"
                            )
                            continue

                        # Get references if not already available
                        refs = doc_info.get("references", [])
                        if not refs or not [
                            valid
                            for valid in refs
                            if valid.get("title") and valid.get("abstract")
                        ]:
                            doc_info = self.search_engine.get_doc_references(doc_info)
                            refs = doc_info.get("references", [])

                        if not refs:
                            logger.warning(
                                f"No references found for document: {doc_info.get('title', 'Unknown')}"
                            )
                            continue

                        # Update document signatures
                        self.root.add_signature_for_doc([doc_info])
                        self.root.add_signature_for_doc(refs)

                        # Filter valid references
                        valid_refs = [
                            ref
                            for ref in refs
                            if ref.get("title") and ref.get("abstract")
                        ]

                        ref_count += len(valid_refs)
                        doc_info["references"] = valid_refs

                        # Save to local DB if enabled
                        if SAVE_ID2DOCS:
                            self._save_id2info(
                                {ref["paper_id"]: ref for ref in valid_refs}
                            )
                            self._save_id2info({doc_info["paper_id"]: doc_info})

                        # Record references
                        node.references.extend([ref["paper_id"] for ref in valid_refs])

                        if not valid_refs:
                            logger.info(
                                f"No valid references found for {doc.get('title', 'Unknown')}"
                            )
                            continue

                        # Calculate similarity for references
                        relevant_refs, irrelevance_refs = (
                            self.search_engine.calculate_similarity(
                                self.root.query_str,
                                valid_refs,
                                search_time=self.search_date,
                                score_thresh=self.sim_threshold,
                                source=f"from reference, parent: {doc.get('arxivId', 'unknown')}",
                            )
                        )

                        relevant_ref_count += len(relevant_refs)

                        # Update document collections
                        self.root.add_signature_for_doc(
                            relevant_refs + irrelevance_refs
                        )
                        self.root.cal_sim_docs.update(
                            {
                                one["paper_id"]: one
                                for one in relevant_refs + irrelevance_refs
                            }
                        )

                        # Update node references
                        node.relevance_refs.extend(relevant_refs)
                        node.irrelevant_refs.extend(irrelevance_refs)

                        node.docs.extend(relevant_refs)
                        node.irrelevant_docs.extend(irrelevance_refs)

                    except Exception as e:
                        logger.error(
                            f"Error processing references for {doc.get('title', 'Unknown')}: {str(e)}"
                        )

            logger.info(
                f"Reference search completed: {ref_count} refs found, {relevant_ref_count} relevant"
            )
            return level_node

        except Exception as e:
            logger.error(f"Reference level search failed: {traceback.format_exc()}")
            return level_node

    def query_expand_from_context(self, level_node, next_level, search_queue):
        logger.info("Generate new query for next level expand ...")
        nex_level_prepare = []
        generate_new_query = []

        current_level_all_valid_docs = []
        for node in tqdm.tqdm(
            level_node, total=len(level_node), desc="query_expand_from_context"
        ):
            valid_docs = [
                [doc, node]
                for doc in node.docs
                if doc["paper_id"] not in self.root.doc_used_to_gen_query
            ]
            current_level_all_valid_docs.extend(valid_docs)

        logger.info(
            f"current_level_all_valid_docs: {len(current_level_all_valid_docs)}"
        )
        if len(current_level_all_valid_docs) == 0:
            logger.info("No relevance doc find, generate some new query")
            valid_docs_info = [[{}, self.root]]
        else:
            current_level_all_valid_docs_sort = sorted(
                current_level_all_valid_docs,
                key=lambda x: x[0]["sim_score"],
                reverse=True,
            )
            valid_docs = current_level_all_valid_docs_sort[
                :REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY
            ]

            valid_docs_info = [
                [self.root.searched_docs[doc["paper_id"]], node]
                for doc, node in valid_docs
            ]

            valid_docs_id = [doc["paper_id"] for doc, node in valid_docs]
            self.root.doc_used_to_gen_query.update(valid_docs_id)

        new_queries = self.search_engine.generate_queries_from_docs(
            self.root.query_str, valid_docs_info, list(self.root.searched_queries)
        )

        logger.info(f"generate {len(new_queries)} queries: {new_queries}")

        for query, parent_node in new_queries:
            if query not in generate_new_query:
                generate_new_query.append(query)
                child = SearchNode(query_str=query)
                nex_level_prepare.append([parent_node, child])
            else:
                logger.info(f"{query} already generated, skip")

        query_node_relations = {}
        querys_to_next_level = []

        logger.info(f"next_level: {len(next_level)}")
        if next_level:
            for one in next_level:
                query_node_relations[one.query_str] = {
                    "own_node": one,
                    "parent_node": one.parent,
                }
                querys_to_next_level.append(one.query_str)

        logger.info(f"nex_level_prepare: {len(nex_level_prepare)}")
        if nex_level_prepare:
            if QUERY_NUM_PRUNED < len(nex_level_prepare):
                nex_level_prepare_shuffle = random.sample(
                    nex_level_prepare, QUERY_NUM_PRUNED
                )
            else:
                nex_level_prepare_shuffle = nex_level_prepare
            for node_inf in nex_level_prepare_shuffle:
                parent_node, child_node = node_inf
                # child_node.parent = parent_node
                # parent_node.add_child(child_node)
                child_node.status = "Expand"

                querys_to_next_level.append(child_node.query_str)
                query_node_relations[child_node.query_str] = {
                    "own_node": child_node,
                    "parent_node": parent_node,
                }
            search_queue.append(querys_to_next_level)

        return level_node, search_queue, query_node_relations

    def search(self, initial_query: str, end_date="") -> List:
        """
        Main search method that:
        1. Initializes search tree with root query
        2. Performs iterative BFS search
        3. Expands search via reference exploration
        4. Returns ranked relevant documents

        Args:
            initial_query: User's search query
            end_date: Optional cutoff date for papers

        Returns:
            Dictionary of relevant documents
        """
        search_start_time = time.time()

        # Set search date
        self.search_date = ""

        # Initialize search tree
        logger.info("🌲 Initializing academic search tree")
        self.root = SearchNode(
            query_str=initial_query,
            status="INIT",
        )
        self.user_query = initial_query

        try:
            # Start with query fusion to generate initial queries
            # Instead of a list of nodes, now query_fusion returns a list of query strings
            expanded_queries, query_node_relations = self.query_fusion()
            search_queue = deque([expanded_queries])

            # Track search progress
            current_depth = 0
            iteration = 0

            # Main search loop
            while search_queue and not self.meet_stop_condition(current_depth):
                iteration += 1
                next_level = []
                level_queries = search_queue.popleft()
                current_depth += 1

                assert isinstance(
                    level_queries, list
                ), f"level_queries: {level_queries}"

                logger.info(f"=== Iteration {iteration}, Depth {current_depth} ===")
                logger.info(f"Processing {len(level_queries)} queries at this level")

                # Phase 1: Query-level search (now takes query strings instead of nodes)
                logger.info(f"Phase 1: Running query-level search")
                level_node, next_level = self.query_level_search(
                    level_queries,
                    query_node_relations,
                    next_level,
                    self.search_date,
                    current_depth,
                )

                # Check if we've found enough documents
                if self.meet_stop_condition(current_depth):
                    logger.info(
                        "Stopping after query-level search: stopping condition met"
                    )
                    for node in level_node:
                        node.status = "STOP"
                    break

                # Phase 2: Reference-based search
                if DO_REFERENCE_SEARCH:
                    logger.info(f"Phase 2: Running reference-level search")
                    level_node = self.reference_level_search(level_node=level_node)

                    if self.meet_stop_condition(current_depth):
                        logger.info(
                            "Stopping after reference search: stopping condition met"
                        )
                        break

                # Phase 3: Generate new queries for next iteration
                logger.info(f"Phase 3: Expanding to next level")
                level_node, search_queue, query_node_relations = (
                    self.query_expand_from_context(level_node, next_level, search_queue)
                )

                logger.info(
                    f"Added {len(query_node_relations)} nodes to search queue for next iteration"
                )
                logger.info(f"Search queue size: {len(search_queue)}")

            # Optional reranking
            # if RERANK:
            #     logger.info("Reranking final document list")

            #     reranked_top = self.reranker.rerank_query_and_doc_list(
            #         self.root.searched_docs, self.user_query
            #     )
            #     self.root.reranked_top_docs = reranked_top

            search_time = time.time() - search_start_time
            doc_count = len(self.root.searched_docs)
            logger.info(
                f"Search completed in {search_time:.2f}s. Found {doc_count} documents."
            )

            # Generate performance statistics
            high_rel_count = len(
                [
                    doc
                    for docid, doc in self.root.searched_docs.items()
                    if doc.get("sim_score", 0) > self.high_score_thresh
                ]
            )
            logger.info(
                f"Found {high_rel_count} highly relevant documents (score > {self.high_score_thresh})"
            )

            return self._collect_results()

        except Exception as e:
            logger.error(f"Search failed: {traceback.format_exc()}")
            # Return whatever results we've gathered so far
            return self._collect_results()
        finally:
            # Always clean up resources
            self._cleanup_resources()

    def _collect_results(self) -> Dict:
        """Collect search results with diversity optimization"""
        # Get all documents
        all_docs = self.root.searched_docs

        # Group documents by research approach/methodology
        docs_by_field = {}
        for doc_id, doc in all_docs.items():
            fields = doc.get("fieldsOfStudy", ["unknown"])
            for field in fields:
                if field not in docs_by_field:
                    docs_by_field[field] = []
                docs_by_field[field].append((doc_id, doc))

        # Select top docs from each field to ensure diversity
        diverse_results = {}
        for field, docs in docs_by_field.items():
            # Sort by relevance within field
            docs.sort(key=lambda x: x[1].get("sim_score", 0), reverse=True)
            # Take top N from each field
            for doc_id, doc in docs[:3]:  # Take top 3 from each field
                diverse_results[doc_id] = doc

        # Fill remaining slots with highest scoring docs overall
        remaining_slots = self.max_docs - len(diverse_results)
        if remaining_slots > 0:
            remaining_docs = {
                doc_id: doc
                for doc_id, doc in all_docs.items()
                if doc_id not in diverse_results
            }
            sorted_remaining = sorted(
                remaining_docs.items(),
                key=lambda x: x[1].get("sim_score", 0),
                reverse=True,
            )
            for doc_id, doc in sorted_remaining[:remaining_slots]:
                diverse_results[doc_id] = doc

        return diverse_results

    def _rank_query_doc_list(self, docs):
        """
        1. First, divide the data into buckets by `sim_score`, with each bucket divided into 0.5 buckets (0-0.5, 0.5-1.0, 1.0-1.5,...).
        2. In each bucket:
            - Sort by `citationCount` in descending order.
            - If `citationCount` is the same, sort by `year` in descending order.
        """

        from collections import defaultdict

        def bucketize(score):
            return round(score / 0.05) * 0.05

        bucketed_docs = defaultdict(list)
        for doc in docs:
            sim_score = doc.get("sim_score", 0.0)
            bucket_key = bucketize(sim_score)

            if doc["year"] is None:
                doc["year"] = -1
            bucketed_docs[bucket_key].append(doc)

        logger.debug(f"bucketed_docs: {bucketed_docs.keys()}")

        sorted_docs = []
        for sim_score in sorted(bucketed_docs.keys(), reverse=True):
            sorted_bucket = sorted(
                bucketed_docs[sim_score],
                key=lambda d: (-d.get("citationCount", 0), -d.get("year", 0)),
            )
            sorted_docs.extend(sorted_bucket)

        return sorted_docs

    def visualize_tree(
        self,
        filename: str = "search_tree",
        save_format: str = "pdf",
        view: bool = False,
    ):
        """
        Visualize the search tree and save it to a file, showing more information:
        - Search query for each node
        - Number of retrieved documents
        - Query weight (if applicable)

        Args:
            filename (str): Filename to save (without extension).
            save_format (str): File format, e.g., "pdf", "png", "svg".
            view (bool): Whether to automatically open the generated file.
        """
        dot = Digraph(comment="Search Tree")
        if not self.root:
            logger.error("Search tree is empty. No visualization will be created.")
            return

        search_queue = deque([(self.root, "0")])
        node_counter = 0

        def process_query(query_str, split=10):
            query_str_processed = ""
            query_str_spl = query_str.split(" ")
            for idx in range(0, len(query_str_spl), split):
                spl = " ".join(query_str_spl[idx : idx + split])
                query_str_processed += spl + "\n"
            return query_str_processed

        while search_queue:
            node, node_id = search_queue.popleft()
            if node_id == "0":
                node_label = process_query(node.query_str)
                dot.node(node_id, "TreeSearch\nStart", shape="hexagon")
            else:
                # if node.status in ["ACHIEVED AND STOP"]:
                #     continue
                num_docs = len(node.docs) if node.docs else 0
                num_relevalce_refs = (
                    len(node.relevance_refs) if node.relevance_refs else 0
                )
                num_references = len(node.references)
                remove_num_docs = (
                    len(node.irrelevant_docs) if node.irrelevant_docs else 0
                )
                source = node.source
                query_str_processed = process_query(node.query_str, 4)
                if node.raw_query != node.query_str:
                    raw_query_processed = process_query(node.raw_query, 4)
                    node_label = f"[{source}]: {query_str_processed}\n[RAW-QUERY]: {raw_query_processed}\nAllRelDocs: [{num_docs}] AllIrrelDocs: [{remove_num_docs}]\nAllRefs: [{num_references}] RelRefs: [{num_relevalce_refs}]"

                else:
                    node_label = f"[{source}]: {query_str_processed}\nAllRelDocs: [{num_docs}] AllIrrelDocs: [{remove_num_docs}]\nAllRefs: [{num_references}] RelRefs: [{num_relevalce_refs}]"

                if hasattr(node, "weight"):
                    node_label += f"\nWeight: {getattr(node, 'weight', 1.0):.2f}"

                node_label += f"\nStatus:{node.status}"
                dot.node(node_id, node_label, shape="box")

            for i, child in enumerate(node.children):
                node_counter += 1
                child_id = str(node_counter)
                dot.edge(node_id, child_id)
                search_queue.append((child, child_id))

        logger.info(f"filename: {filename}")
        filepath = dot.render(filename=filename, format=save_format, view=view)
        logger.info(f"搜索树已保存为文件: {filepath}")

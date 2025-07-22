# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

from global_config import *
from instruction import *
import traceback
from local_request_v2 import get_from_llm
from utils import fetch_string
import json
from log import logger
from typing import List, Dict, Any, Optional, Set
import concurrent.futures
from api_web import (
    google_search_arxiv_id,
    get_doc_info_from_semantic_scholar_by_arxivid,
    get_doc_info_from_api,
    search_paper_from_arxiv_by_arxiv_id,
    parallel_search_search_paper_from_arxiv,
    search_paper_via_query_from_openalex,
    search_paper_via_query_from_semantic,
    search_doc_via_url_from_openalex,
    search_from_pubmed,
    fetch_pubmed_json
)
from datetime import datetime, timedelta
import re
import numpy as np
import time

from dataclasses import dataclass
from base_class import SearchResult
from datetime import datetime
from collections import defaultdict

from local_db_v2 import db_path, ArxivDatabase

def get_info_from_local(id_list):
    already = []
    to_process = []
    if os.path.exists(db_path):
        with ArxivDatabase(db_path) as db:
            for _id in id_list:
                db_info = db.get(_id)
                if db_info is None:
                    to_process.append(_id)
                else:
                    already.append(_id)
        logger.info(f"already num is: {len(already)}, to_process num is :{len(to_process)}")
        return already,to_process
    else:
        return {},id_list


class MultiSearchAgent:
    """Agent for parallel multi-source academic paper search and result aggregation."""

    def __init__(self, max_workers: int = 3, batch_size: int = 10):
        """
        Initialize the multi-search agent.
        Args:
            max_workers: Maximum number of parallel search workers
            batch_size: Size of batches for paper detail retrieval
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.current_date = "2025-03-24"  # 当前日期，参考你的需求

    def extract_keywords(self, query: str, source: str = "semantic") -> List[str]:
        """Extract keywords from query optimized for a specific source."""
        query = query.lower()
        model_inp = template_extract_keywords_source_aware.format(
            user_query=query, source=source
        )
        for _ in range(4):
            try:
                response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
                pattern = r"\[Start\](.*?)\[End\]"
                match = re.search(pattern, response)
                if match:
                    keywords = match.group(1).strip()
                    logger.info(f"Extracted keywords for {source}: {keywords}")
                    return [kw.strip() for kw in keywords.split(",") if kw.strip()][:KEY_WORDS_NUM]
            except:
                logger.error(f"Failed to extract keywords: {traceback.format_exc()}")
        return []

    def _google_arxiv_search(
        self,
        queries: List[str],
        end_date: str = "",
        searched_docs: Dict[str, Any] = None,
    ) -> SearchResult:
        """Execute Google Scholar search for a list of queries."""
        try:

            if searched_docs is None:
                searched_docs = {}

            # 如果 end_date 为空，使用当前日期

            # if not end_date:
            #     end_date = self.current_date

            # Step 1: 并行搜索 arxiv_ids
            merged_papers = {}
            query2docs = {query: [] for query in queries}

            results = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_query = {
                    executor.submit(
                        google_search_arxiv_id, query, API_TRY_COUNT, 15, end_date
                    ): query
                    for query in queries
                }
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        results[query] = future.result(timeout=2)
                    except Exception as e:
                        logger.error(
                            f"Google search failed for query {query}: {str(e)}"
                        )
                        results[query] = []

            logger.info(f"google_search_arxiv_id results: {results}")

            # Step 2: 去重 arxiv_ids
            unique_arxiv: Set[str] = set()
            original_arxiv = []
            for arxiv_ids in results.values():
                original_arxiv.extend(arxiv_ids)
                for arxiv_id in arxiv_ids:
                    if arxiv_id not in searched_docs:
                        unique_arxiv.add(arxiv_id)

            logger.info(
                f"Original num: {len(original_arxiv)}, Unique num: {len(unique_arxiv)}"
            )

            # Step 3: 统一获取论文详情
            id2docs = {}
            if unique_arxiv:
                id2docs = parallel_search_search_paper_from_arxiv(
                    list(unique_arxiv),
                    max_workers=self.max_workers,
                    batch_size=self.batch_size,
                )

            # Step 4: 合并结果

            for query, arxiv_ids in results.items():
                for arxiv_id in arxiv_ids:
                    if arxiv_id in id2docs:
                        paper_info = id2docs[arxiv_id]
                        merged_papers[arxiv_id] = paper_info
                        query2docs[query].append(paper_info)
        except:
            logger.error(f"google search error: {traceback.format_exc()}")
        finally:
            return SearchResult(
                source="arxiv", papers=merged_papers, query2paper=query2docs
            )

    def _semantic_search(
        self, keyword: str, raw_query: str, end_date: str = "", max_papers: int = 15
    ) -> SearchResult:
        """Execute Semantic Scholar search."""
        logger.info(f"Searching Semantic Scholar for '{query}'")
        try:
            papers = search_paper_via_query_from_semantic(
                query=keyword, max_paper_num=max_papers
            )
            logger.info(f"Found {len(papers)} papers for '{keyword}' from Semantic")
            return SearchResult(
                source="semantic", papers=papers, keyword=keyword, raw_query=raw_query
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {traceback.format_exc()}")
            return SearchResult(source="semantic", papers={}, error=str(e))

    def _openalex_search(
        self, keyword: str, raw_query: str, end_date: str = ""
    ) -> SearchResult:
        """Execute OpenAlex search."""
        logger.info(f"Searching OpenAlex for '{keyword}'")
        try:
            papers = search_paper_via_query_from_openalex(keyword, per_page=10)
            logger.info(f"Found {len(papers)} papers for '{keyword}' from OpenAlex")
            return SearchResult(
                source="openalex", papers=papers, keyword=keyword, raw_query=raw_query
            )
        except Exception as e:
            logger.error(f"OpenAlex search failed: {traceback.format_exc()}")
            return SearchResult(
                source="openalex", papers={}, raw_query=raw_query, error=str(e)
            )

    def _pubmed_search(
        self, keyword: str, raw_query: str, max_results: int = 10
    ) -> SearchResult:
        """Execute PubMed search."""
        logger.info(f"Searching PubMed for '{keyword}'")
        try:
            papers = search_from_pubmed(keyword, max_results=max_results)
            logger.info(f"Found {len(papers)} papers for '{keyword}' from PubMed")
            return SearchResult(
                source="pubmed",
                papers={p["paper_id"]: p for p in papers},
                keyword=keyword,
                raw_query=raw_query,
            )
        except Exception as e:
            logger.error(f"PubMed search failed: {traceback.format_exc()}")
            return SearchResult(
                source="pubmed", papers={}, raw_query=raw_query, error=str(e)
            )

    def _merge_paper_info(
        self, existing: Dict[str, Any], new: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        """Merge paper information from different sources."""
        merged = existing.copy()

        for field in [
            "abstract",
            "title",
            "publicationYear",
            "authors",
            "fieldsOfStudy",
        ]:
            if field not in merged or not merged[field]:
                merged[field] = new.get(field)

        if "sources" not in merged:
            merged["sources"] = [existing.get("source", "unknown")]
        merged["sources"].append(source)
        merged["sources"] = list(set(merged["sources"]))

        for score_field in ["citationCount", "referenceCount"]:
            if score_field in new:
                if score_field not in merged:
                    merged[score_field] = new[score_field]
                else:
                    merged[score_field] = max(merged[score_field], new[score_field])

        return merged

    def _merge_search_results_grouped(
        self, results: List[SearchResult], source: str
    ) -> SearchResult:
        """Merge multiple search results grouped by raw_query."""
        # Step 1: Group results by raw_query
        grouped_results = defaultdict(list)
        for result in results:
            if result.raw_query:
                grouped_results[result.raw_query].append(result)
            else:
                logger.warning(f"Result without raw_query: {result}")

        # Step 2: Merge results within each group
        merged_search_keywords = {}
        merged_query2paper = {}
        for raw_query, group in grouped_results.items():
            logger.info(f"[{source}]: Merging results for raw_query: {raw_query}")
            for result in group:
                if not result.papers:
                    logger.info(f"No papers found in this result: {result}, skipping")
                    continue

                if raw_query not in merged_query2paper:
                    merged_query2paper[raw_query] = []

                merged_query2paper[raw_query].extend(list(result.papers.values()))

                if raw_query not in merged_search_keywords:
                    merged_search_keywords[raw_query] = []
                merged_search_keywords[raw_query].append(result.keyword)

        return SearchResult(
            source=source,
            papers=[],
            query2paper=merged_query2paper,
            extra={"merged_query_to_keywords": merged_search_keywords},
        )

    def _merge_search_results(
        self, results: List[SearchResult], source: str
    ) -> SearchResult:
        """Merge multiple search results from the same source."""
        merged_papers = {}
        merged_querys = []
        merged_raw_querys = []
        extra = {}

        for result in results:
            if not result.papers:
                logger.info(f"No papers found in this result: {result}, skipping")
                continue
            keyword = result.keyword
            raw_query = result.raw_query
            papers = result.papers
            merged_querys.append(keyword)
            merged_raw_querys.append(raw_query)
            extra[keyword] = len(result.papers)
            for paper_id, paper_info in result.papers.items():
                if paper_id in merged_papers:
                    merged_papers[paper_id] = self._merge_paper_info(
                        merged_papers[paper_id], paper_info, source
                    )
                else:
                    if "source" not in paper_info:
                        paper_info["source"] = source
                    else:
                        paper_info["source"] = f"{paper_info['source']}|{source}"
                    paper_info["sources"] = [source]
                    merged_papers[paper_id] = paper_info

        return SearchResult(
            source=source,
            papers=merged_papers,
            query="|".json(merged_querys),
            extra=extra,
        )

    def search_papers(
        self,
        querys: List[str],
        sources: List[str] = ["arxiv", "semantic", "openalex"],
        end_date: str = "",
        searched_docs: dict = {},
        rerank: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute parallel search across multiple sources with a list of queries.

        Args:
            querys: List of search queries
            sources: List of search sources to use
            end_date: End date filter
            searched_docs: Dictionary of already searched documents to avoid duplicates
            rerank: Whether to rerank results

        Returns:
            Dict containing aggregated results and stats
        """
        if not querys:
            logger.error("Query list is empty")
            return {}
        logger.info(f"Searching with query list: {querys} across sources: {sources}")

        # Validate sources
        search_funcs = {
            "arxiv": self._google_arxiv_search,
            "semantic": self._semantic_search,
            "openalex": self._openalex_search,
            "pubmed": self._pubmed_search,
        }
        valid_sources = []
        for source in sources:
            if source in search_funcs:
                valid_sources.append(source)
            else:
                logger.error(f"Unknown search source: {source}")

        if not valid_sources:
            logger.error("No valid search sources specified")
            return {}

        # Step 1: Extract keywords for each query for each source
        keyword_extraction_sources = {"openalex", "pubmed"}
        query_keywords_by_source = {}  # Maps source -> query -> keywords
        keywords_combine_query = {}
        query_keywords2raw = {}

        if set(valid_sources).intersection(keyword_extraction_sources):
            for source in valid_sources:
                if source in keyword_extraction_sources:
                    query_keywords_by_source[source] = {}
                    keywords_combine_query[source] = {}
                    query_keywords2raw[source] = {}
                    source_keywords_already = []
                    # Process each query separately
                    for query_idx, query in enumerate(querys):
                        # Extract keywords optimized for this specific source and query
                        source_keywords = self.extract_keywords(query, source)
                        if source_keywords:
                            source_keywords_valid = []
                            for one in source_keywords:
                                if one not in source_keywords_already:
                                    source_keywords_already.append(one)
                                    source_keywords_valid.append(one)
                                else:
                                    logger.info(
                                        f"Keyword '{one}' already exists in source keywords for {source}"
                                    )

                            query_keywords_by_source[source][
                                query
                            ] = source_keywords_valid
                            keywords_combine_query[source][query] = "|".join(
                                source_keywords_valid
                            )
                            query_keywords2raw[source][
                                "|".join(source_keywords_valid)
                            ] = query
                            logger.info(
                                f"Query {query_idx+1}: Extracted {len(source_keywords_valid)} keywords for {source}"
                            )
                        else:
                            # Fallback to default keywords if extraction fails
                            query_keywords_by_source[source][query] = [query]
                            keywords_combine_query[source][query] = query
                            query_keywords2raw[source][query] = query
                            logger.warning(
                                f"Query {query_idx+1}: No keywords extracted for {source}, falling back to original query"
                            )

        # Step 2: Prepare and execute all search tasks in parallel
        search_tasks = []
        future_to_source = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Step 1: Submit tasks

            for source in sources:
                if source == "arxiv":
                    # Google: Use the entire query list
                    future = executor.submit(search_funcs[source], querys, end_date, searched_docs)
                    search_tasks.append(future)
                    future_to_source[future] = (source, "google_arxiv_query")

                elif source in keyword_extraction_sources:
                    for query in querys:
                        for keyword in query_keywords_by_source[source][query]:
                            future = executor.submit(search_funcs[source], keyword, query, end_date)
                            search_tasks.append(future)
                            future_to_source[future] = (source, f"{query[:20]}...: {keyword}")
            # Step 2: Collect results with timeout and error logging
            results_by_source = {source: [] for source in valid_sources}

            for future in concurrent.futures.as_completed(search_tasks):  # total timeout for all tasks
                source, query_or_keyword = future_to_source[future]
                try:
                    start_time = time.time()
                    result = future.result(timeout=5)
                    duration = time.time() - start_time
                    results_by_source[source].append(result)
                    logger.info(f"Completed {source} search for '{query_or_keyword}' in {duration:.1f}s")
                except concurrent.futures.TimeoutError:
                    logger.error(f"{source} search timed out for '{query_or_keyword}'")
                except Exception as e:
                    logger.error(f"{source} search failed for '{query_or_keyword}': {traceback.format_exc()}")
                    logger.debug(f"Future state: {future}")

        # Step 4: Merge results for each source
        merged_results = {}
        for source in valid_sources:
            logger.info(
                f"Source is :{source}, results_by_source num is {len(results_by_source[source])}"
            )
            if results_by_source[source]:
                if source == "arxiv":
                    merged_results[source] = results_by_source[source][0]
                else:
                    merged_results[source] = self._merge_search_results_grouped(
                        results_by_source[source], source
                    )
        logger.info(f"merged_results: {merged_results}")

        # Step 5: Merge results from all sources
        final_papers = {}
        final_query2docs = {}
        query_source_map = {}  # Track which sources were used for each query
        query_keywords2raw = {}
        for source, result in merged_results.items():
            if not result.query2paper:
                logger.info(f"result is empty, skip: {result}")
                continue
            if source == "arxiv" and result.query2paper:
                # For Google/ArXiv results which already track query->paper relationships
                for query, papers in result.query2paper.items():
                    # Use the original query without source prefix
                    if query not in final_query2docs:
                        final_query2docs[query] = []
                    # Track which source provided results for this query
                    # Add papers to query results (without adding source to paper object)
                    final_query2docs[query].extend(papers)
                    query_keywords2raw[query] = query
                    query_source_map[query] = source
                    final_papers.update(
                        {
                            paper.get("paper_id", paper.get("arxivId", "")): paper
                            for paper in papers
                        }
                    )

            else:
                for query, papers in result.query2paper.items():
                    logger.info(f"papers: {len(papers)}: {papers[0]}")

                    query_extracted_keywords = result.extra.get(
                        "merged_query_to_keywords", {}
                    ).get(query, [])
                    query_extracted_keywords_str = "|".join(query_extracted_keywords)
                    if query_extracted_keywords_str not in final_query2docs:
                        final_query2docs[query_extracted_keywords_str] = []
                    final_query2docs[query_extracted_keywords_str].extend(papers)
                    query_keywords2raw[query_extracted_keywords_str] = query
                    query_source_map[query_extracted_keywords_str] = source
                    final_papers.update({paper["paper_id"]: paper for paper in papers})

        logger.info(f"All retrieved papers: {len(final_papers)}")
        logger.info(
            f"Retrieved papers details: {[{query:len(final_query2docs[query])} for query in final_query2docs]}"
        )
        logger.info(f"Query source mapping: {query_source_map}")
        logger.info(f"Query keywords2raw: {query_keywords2raw}")

        # Return the query-source mapping along with the results
        return final_query2docs, final_papers, query_source_map, query_keywords2raw


def _generate_query_from_reference(
    user_query, one_doc, searched_queries
) -> Optional[str]:
    """Generate new query from reference"""


    model_inp = template_context_query_generation.format(
        user_query=user_query,
        searched_queries=searched_queries,
        doc_title=one_doc.get("title", ""),
        doc_abstract=one_doc.get("abstract", ""),
        doc_field=one_doc.get("fieldsOfStudy", ""),
    )

    logger.info(f"_generate_query_from_reference model info: {model_inp}")

    for _ in range(LLM_TRY_COUNT):
        try:
            response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
            logger.info(f"response: {response}")
            response = fetch_string(response)
            query_list = json.loads(response)
            output = []
            for new_query in query_list:
                if new_query == "":
                    continue
                if new_query not in searched_queries:
                    output.append(new_query)
                else:
                    logger.info(f"{new_query} already exist in {searched_queries}")
            return output
        except:
            logger.error(
                f"Failed to parse response: {response}, will retry {SLEPP_TIME_LLM} seconds...; Error: {traceback.format_exc()}"
            )
            time.sleep(SLEPP_TIME_LLM)
    return []


def similarity_code_v4(query, doc, search_time):
    try:
        output = {}
        model_inp = (
            template_sim_between_query_doc_v2_inst.format(
                searchTime=search_time,
                userQuery=query,
                Title=doc["title"],
                Abstract=doc["abstract"],
                Author=(
                    "; ".join([one["name"] for one in doc["authors"]])
                    if doc["authors"] is not None
                    else ""
                ),
                fieldsOfStudy=(
                    ";".join(doc["fieldsOfStudy"])
                    if doc["fieldsOfStudy"] is not None
                    else ""
                ),
                publicationYear=doc["publicationYear"] if doc["publicationYear"] is not None else "",
            )
            + template_sim_between_query_doc_v2_example
        )
        response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
        response = fetch_string(response)
        response = json.loads(response.strip())
        overall_score = [
            response[key]
            for key in [
                "topic_match",
                "contextual_relevance",
                "depth_completeness",
            ]
        ]
        output["sim_score"] = np.mean(overall_score) / 5.0
        output["sim_info_details"] = response
        return output
    except:
        logger.error(f"similarity_code_v4 error {traceback.format_exc()}")
        return {}


def similarity_code_v5(query, doc):
    output = {}
    try:
        model_inp = evaluation_prompt.format(
            title=doc["title"],
            abstract=doc["abstract"],
            user_query=query,
        )
        response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)

        score_match = re.search(r"Score:\s*([0-1]\.\d+|\d+)", response)
        if score_match:
            score = float(score_match.group(1))
            output["sim_score"] = score
            output["sim_info_details"] = response
            return output
    except:
        logger.error(f"similarity_code_v5 error {traceback.format_exc()}")
        return {}


def _calculate_similarity_with_retry(
    query: str,
    search_time: str,
    doc: Dict,
    max_retries: int = LLM_TRY_COUNT,
    timeout: int = 20,
) -> float:
    """计算 query 与文档的相关性，带重试和超时"""
    output = {}
    for attempt in range(max_retries):
        response = None
        try:
            # # v4
            output = similarity_code_v4(query, doc, search_time)
            ## v5
            # output = similarity_code_v5(query,doc)
            if output:
                return output
        except Exception as e:
            logger.error(
                f"Similarity calculation failed for '{doc['title']}', attempt {attempt+1}/{max_retries}, error: {str(e)}, response: {response}"
            )
            time.sleep(timeout)  # 失败后等待 2 秒再重试
    return output  # 返回最低分，防止影响整体流程


class AcademicTreeSearchEngine:

    def __init__(
        self,
    ):
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.mretrival_processer = MultiSearchAgent()
        self._emd_model = None  # 用于存储懒加载的实例
        self._selector = None  # 用于存储懒加载的实例

    @property
    def emd_model(self):
        if self._emd_model is None:
            from embedding_agent import BGEM3EmbeddingAgent

            self._emd_model = BGEM3EmbeddingAgent()
        return self._emd_model

    @property
    def selector(self):
        if self._selector is None:
            from pasa_agent import Agent as PasaAgent

            selector_path = (
                "/share/project/shixiaofeng/data/model_hub/pasa/pasa-7b-selector"
            )
            self._selector = PasaAgent(selector_path)
        return self._selector

    def expand_query_native(self,query:str):
        judge_info = {"expanded_queries_info":{}}
        for _ in range(WEB_TRYNUM):
            try:
                # model_inp = template_query_fusion.format(user_query=query)
                model_inp = template_query_fusion_pasa.format(user_query=query)
                logger.debug(f"query correct model_inp: {model_inp}")
                response = get_from_llm(model_inp,
                                        4096,
                                        model_name=LLM_MODEL_NAME)
                response = fetch_string(response)
                logger.info(f"query correct response: {response}")
                response = json.loads(response)
                try:
                    judge_info["expanded_queries_info"]["expanded_queries"] = response
                    return judge_info
                except:
                    logger.error(
                        f"Query correction failed: {traceback.format_exc()}")
            except:
                logger.error(
                    f"Query correction failed: {traceback.format_exc()}")
        return judge_info



    def expand_query(self, query: str):
        """
        Enhanced query expansion with intent detection and source identification.

        This method:
        1. Analyzes the user's query intent and domain
        2. Identifies appropriate search sources (web, scholar, arxiv, etc.)
        3. Determines if query expansion would improve results
        4. Generates optimized expansions if needed

        Args:
            query: The original user query

        Returns:
            Dictionary containing query analysis and expansions
        """
        logger.info(f"Analyzing query intent: {query}")

        # Initialize response structure
        result = {
            "query_intent": "",
            "domain": "",
            "suitable_sources": [],
            "needs_expansion": False,
            "expansion_reason": "",
            "expanded_queries": [],
        }

        current_year = datetime.now().year
        # Step 1: Analyze query intent, domain and suitable sources
        try:
            intent_analysis = self._analyze_query_intent(query)
            if intent_analysis:
                result.update(intent_analysis)
                logger.info(f"Query intent analysis: {intent_analysis}")
            else:
                logger.warning("Query intent analysis failed, using defaults")
                result["query_intent"] = "general research"
                result["domain"] = "undefined"
                result["suitable_sources"] = ["arxiv"]
        except Exception as e:
            logger.error(f"Error in query intent analysis: {traceback.format_exc()}")
            result["query_intent"] = "general research"
            result["domain"] = "undefined"
            result["suitable_sources"] = ["arxiv", "openalex"]

        # Step 2: Determine if query needs expansion
        try:
            expansion_analysis = self._evaluate_expansion_need(query, result["domain"])
            if expansion_analysis:
                result["needs_expansion"] = expansion_analysis["needs_expansion"]
                result["expansion_reason"] = expansion_analysis["reason"]
                logger.info(
                    f"Query expansion needed: {result['needs_expansion']}, reason: {result['expansion_reason']}"
                )
            else:
                logger.warning("Expansion analysis failed, defaulting to no expansion")
                result["needs_expansion"] = False
                result["expansion_reason"] = "Analysis failed, keeping original query"
        except Exception as e:
            logger.error(f"Error in expansion analysis: {traceback.format_exc()}")
            result["needs_expansion"] = False
            result["expansion_reason"] = "Analysis error, keeping original query"

        # Step 3: Generate expanded queries if needed
        if result["needs_expansion"]:
            try:
                expanded_queries = self._generate_expanded_queries(
                    query, result["domain"], result["query_intent"]
                )
                result["expanded_queries"] = expanded_queries
                logger.info(f"Generated {len(expanded_queries)} expanded queries")
            except Exception as e:
                logger.error(
                    f"Error generating expanded queries: {traceback.format_exc()}"
                )
                result["expanded_queries"] = []
                result["expansion_reason"] += " (expansion generation failed)"
        return result

    def _analyze_query_intent(self, query: str):
        """
        Analyze the query to determine intent, domain and suitable search sources.

        Args:
            query: The user query

        Returns:
            Dictionary with query intent analysis
        """
        from datetime import datetime

        current_year = datetime.now().year
        previous_year = current_year - 1
        try:
            # Prompt for query intent analysis
            prompt = template_query_intent.format(query=query,current_year=current_year,previous_year=previous_year)

            for attempt in range(LLM_TRY_COUNT):
                try:
                    response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                    response = fetch_string(response)
                    result = json.loads(response)

                    # Validate the response has required fields
                    if all(
                        k in result
                        for k in ["query_intent", "domain", "suitable_sources","source_reason"]
                    ):
                        return result
                    logger.warning(f"Incomplete response from LLM: {result}")
                except Exception as e:
                    logger.warning(
                        f"LLM analysis failed (attempt {attempt+1}): {str(e)}"
                    )
                    time.sleep(SLEPP_TIME_LLM)

            logger.error("All attempts to analyze query intent failed")
            return None
        except Exception as e:
            logger.error(f"Error in query intent analysis: {traceback.format_exc()}")
            return None

    def _evaluate_expansion_need(self, query: str, domain: str):
        """
        Evaluate if the query would benefit from expansion.

        Args:
            query: The original query
            domain: The identified domain

        Returns:
            Dictionary with expansion decision and reason
        """
        try:
            # Prompt for evaluating if query needs expansion
            prompt = template_query_expand_judge_opt.format(query=query, domain=domain)

            for attempt in range(LLM_TRY_COUNT):
                try:
                    response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                    response = fetch_string(response)
                    result = json.loads(response)

                    # Validate the response has required fields
                    if "needs_expansion" in result and "reason" in result:
                        return result
                    logger.warning(f"Incomplete response from LLM: {result}")
                except Exception as e:
                    logger.warning(
                        f"LLM expansion evaluation failed (attempt {attempt+1}): {str(e)}"
                    )
                    time.sleep(SLEPP_TIME_LLM)

            logger.error("All attempts to evaluate expansion need failed")
            return None
        except Exception as e:
            logger.error(f"Error in expansion evaluation: {traceback.format_exc()}")
            return None

    def _generate_expanded_queries(self, query: str, domain: str, intent: str):
        """
        Generate expanded queries based on the original query, domain and intent.

        This method dynamically selects the appropriate query expansion strategy based on:
        1. Query complexity and specificity
        2. Domain characteristics
        3. Research intent (e.g., survey, methodology, application)

        Args:
            query: The original query
            domain: The identified domain
            intent: The query intent

        Returns:
            List of expanded queries
        """
        try:
            from datetime import datetime

            current_year = datetime.now().year
            previous_year = current_year - 1

            # Determine the appropriate template based on query analysis
            if FUSION_TEMP == "AUTOMATIC" and  self._is_survey_focused(intent):
                # For survey-focused queries, prioritize finding comprehensive reviews
                logger.info(f"Using survey-focused expansion for query: {query}")
                prompt = template_query_fusion_survery_forcus.format(
                    user_query=query,
                    user_input_N=5,
                    current_year=current_year,
                    previous_year=previous_year,
                )
                prompt_type = "survey"
            elif FUSION_TEMP == "AUTOMATIC" and self._is_complex_domain(domain):
                # For queries in complex or specialized domains, use domain-aware expansion
                logger.info(f"Using domain-aware expansion for query in {domain}")
                prompt = template_domain_aware_query_expansion.format(
                    user_input_N=5,
                    user_query=query,
                    intent=intent,
                    domain=domain,
                    current_year=current_year,
                    previous_year=previous_year,
                )
                prompt_type = "domain"
            elif FUSION_TEMP == "PASA":
                # Use PASA template if explicitly configured
                logger.info(f"Using PASA template for query expansion")
                prompt = template_query_fusion_pasa.format(user_query=query)
                prompt_type = "pasa"
            elif FUSION_TEMP == "WITHEXPLAIN":
                # Use withexplain template if explicitly configured
                logger.info(f"Using withexplain for query: {query}")
                prompt = (
                    template_query_fusion_with_score_inst
                    + template_query_fusion_with_score_user.format(
                        user_query=query, user_input_N=5
                    )
                )
                prompt_type = "withexplain"
            else:
                # Default to domain-aware expansion for all other cases
                logger.info(f"Using default domain-aware expansion")
                prompt = template_domain_aware_query_expansion.format(
                    user_input_N=5, user_query=query, intent=intent, domain=domain
                )
                prompt_type = "domain"

            # Track attempts and keep best result
            best_response = None
            best_query_count = 0

            logger.info(f"Expand query prompt for LLM: {prompt}")
            for attempt in range(LLM_TRY_COUNT):
                try:
                    response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                    response = fetch_string(response)
                    logger.info(f"Expanded queries response: {response}")

                    # Parse the response based on its format
                    try:
                        parsed_response = json.loads(response)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response: {str(e)}")
                        # Attempt to extract JSON from text if standard parsing fails
                        match = re.search(r"\{.*\}", response, re.DOTALL)
                        if match:
                            try:
                                parsed_response = json.loads(match.group(0))
                            except:
                                logger.warning("Failed to extract JSON from response")
                                continue
                        else:
                            # Try to extract a list if JSON object extraction failed
                            match = re.search(r"\[.*\]", response, re.DOTALL)
                            if match:
                                try:
                                    parsed_response = json.loads(match.group(0))
                                except:
                                    logger.warning(
                                        "Failed to extract JSON list from response"
                                    )
                                    continue
                            else:
                                continue

                    # Extract queries with unified format handling
                    expanded_queries = self._extract_queries_from_response(
                        parsed_response, prompt_type
                    )
                    logger.info(f"Extracted queries: {expanded_queries}")

                    if expanded_queries:
                        # Track best response by number of queries
                        if len(expanded_queries) > best_query_count:
                            best_response = expanded_queries
                            best_query_count = len(expanded_queries)

                        # Return immediately if we got a good number of queries
                        if len(expanded_queries) >= 3:
                            return expanded_queries

                except Exception as e:
                    logger.warning(
                        f"LLM expansion failed (attempt {attempt+1}): {str(e)}"
                    )
                    time.sleep(SLEPP_TIME_LLM)

            # If we tried all attempts but still have a valid best response, return it
            if best_response and len(best_response) > 0:
                logger.info(
                    f"Using best response from {LLM_TRY_COUNT} attempts: {len(best_response)} queries"
                )
                return best_response

            # Fallback to basic expansion if all attempts fail
            logger.error(
                "All attempts to generate expanded queries failed, using fallback"
            )
            return self._generate_fallback_queries(query, domain)

        except Exception as e:
            logger.error(f"Error generating expanded queries: {traceback.format_exc()}")
            # Emergency fallback
            return self._generate_emergency_fallback_queries(query)

    def _extract_queries_from_response(self, response, prompt_type):
        """
        Extract queries from different response formats and standardize.

        Args:
            response: The parsed JSON response
            prompt_type: The type of prompt used (survey, domain, pasa, withexplain)

        Returns:
            List of query strings
        """
        expanded_queries = []

        try:
            # Handle different response formats based on prompt type
            if isinstance(response, list):
                # Direct list of strings (PASA format)
                expanded_queries = [q for q in response if isinstance(q, str)]
            elif isinstance(response, dict):
                # Dictionary with expanded_queries field
                if "expanded_queries" in response:
                    queries_data = response["expanded_queries"]
                    if isinstance(queries_data, list):
                        for item in queries_data:
                            if isinstance(item, str):
                                expanded_queries.append(item)
                            elif isinstance(item, dict) and "query" in item:
                                expanded_queries.append(item["query"])
                # Some responses might have a different structure
                elif prompt_type == "withexplain" and "rewritten_queries" in response:
                    for item in response["rewritten_queries"]:
                        if isinstance(item, dict) and "rewritten_query" in item:
                            expanded_queries.append(item["rewritten_query"])

            # Log metadata if available (for monitoring/improvement)
            if isinstance(response, dict):
                if "summary" in response:
                    logger.info(f"Query expansion summary: {response['summary']}")
                if "domain_keywords" in response:
                    logger.info(f"Domain keywords: {response['domain_keywords']}")

            return expanded_queries
        except Exception as e:
            logger.error(f"Error extracting queries from response: {str(e)}")
            return []

    def _generate_fallback_queries(self, query, domain):
        """
        Generate fallback queries when regular expansion fails.

        Args:
            query: The original query
            domain: The identified domain

        Returns:
            List of fallback queries
        """
        logger.info(f"Using fallback query expansion for: {query}")
        return [
            f"survey papers on {query}",
            f"literature review {query}",
            f"state-of-the-art {query}",
            f"recent advances in {query}",
            f"{domain} {query} methodologies",
        ]

    def _generate_emergency_fallback_queries(self, query):
        """
        Generate emergency fallback queries when all else fails.

        Args:
            query: The original query

        Returns:
            List of minimal fallback queries
        """
        logger.info(f"Using emergency fallback expansion for: {query}")
        return [
            f"survey papers on {query}",
            f"literature review {query}",
            f"state-of-the-art {query}",
        ]

    def _is_survey_focused(self, intent: str) -> bool:
        """
        Determine if the query intent is focused on finding survey or review papers.
        Uses fast keyword matching first, then falls back to model-based detection if needed.

        Args:
            intent: The query intent string

        Returns:
            Boolean indicating if the intent is survey-focused
        """
        intent_lower = intent.lower()

        # Fast path: Check for explicit survey indicators
        survey_indicators = [
            "survey",
            "review",
            "overview",
            "state-of-the-art",
            "literature",
            "comprehensive",
            "summary",
            "taxonomy",
            "comparative",
            "meta-analysis",
        ]

        if any(indicator in intent_lower for indicator in survey_indicators):
            logger.info(f"Detected survey intent via keywords in: {intent}")
            return True

        # Fast path: Check for implicit survey patterns
        implicit_patterns = [
            r"what (is|are) the current",
            r"(summarize|summarizing) (recent|current)",
            r"broad (understanding|overview)",
            r"comprehensive (analysis|study)",
            r"(existing|available) (approaches|methods)",
            r"compare (different|various)",
            r"trends in",
        ]

        if any(re.search(pattern, intent_lower) for pattern in implicit_patterns):
            logger.info(f"Detected survey intent via patterns in: {intent}")
            return True

        # Medium path: Check for contextual clues
        contextual_indicators = [
            # Academic literature orientation
            ("literature", "field"),
            ("papers", "compare"),
            ("research", "directions"),
            ("developments", "field"),
            # Broad scope indicators
            ("comprehensive", "understanding"),
            ("overview", "approaches"),
            ("different", "techniques"),
            # Historical/evolutionary interest
            ("evolution", "development"),
            ("progress", "area"),
            ("history", "development"),
        ]

        if any(
            all(term in intent_lower for term in pair) for pair in contextual_indicators
        ):
            logger.info(f"Detected survey intent via contextual pairs in: {intent}")
            return True

        # Slow path: Use model for uncertain cases
        intent_cache_key = f"survey_intent:{intent_lower}"

        # Check if we have a cached result
        if hasattr(self, "_survey_intent_cache"):
            if intent_cache_key in self._survey_intent_cache:
                return self._survey_intent_cache[intent_cache_key]
        else:
            # Initialize cache if it doesn't exist
            self._survey_intent_cache = {}

        # Only call LLM for intents we're unsure about
        try:
            prompt = f"""Determine if this academic research intent is primarily focused on finding SURVEY or REVIEW papers rather than primary research:

Research intent: "{intent}"

A survey/review-focused intent typically seeks:
1. Comprehensive overviews of a research area
2. Comparisons of different approaches or methodologies
3. Summaries of the state-of-the-art
4. Historical development or evolution of concepts
5. Taxonomies or categorizations of approaches

Respond with only "Yes" if the intent is primarily seeking survey/review papers, or "No" if it's seeking specific primary research papers."""

            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            is_survey = "yes" in response.lower()

            # Cache the result for future use
            self._survey_intent_cache[intent_cache_key] = is_survey
            logger.info(f"Model determined survey intent as {is_survey} for: {intent}")
            return is_survey
        except Exception as e:
            # Fall back to more conservative check on error
            logger.error(f"Error determining survey intent: {str(e)}")
            return "overview" in intent_lower or "review" in intent_lower

    def _is_complex_domain(self, domain: str) -> bool:
        """
        Determine if the domain is specialized using both rules and model-based assessment.

        Args:
            domain: The domain string

        Returns:
            Boolean indicating if the domain is complex/specialized
        """
        domain_lower = domain.lower()

        # Fast path: Check against known complex domains first
        complex_domains = {
            "quantum computing",
            "genomics",
            "bioinformatics",
            "neuroscience",
            "computational linguistics",
            "cryptography",
            "nanomaterials",
            "immunology",
            "pharmacology",
            "astrophysics",
            "high energy physics",
            "theoretical computer science",
            "robotics",
            "material science",
        }

        # If it's in our known list, return immediately
        if any(complex_domain in domain_lower for complex_domain in complex_domains):
            return True

        # Fast path: Technical term indicators
        technical_indicators = [
            "quantum",
            "computational",
            "theoretical",
            "stochastic",
            "bayesian",
        ]

        if any(indicator in domain_lower for indicator in technical_indicators):
            return True

        # Fast path: Check multi-word domains that typically indicate complexity
        if len(domain_lower.split()) >= 3:
            return True

        # Slow path: Use the model for uncertain cases, but cache results
        domain_cache_key = f"domain_complexity:{domain_lower}"

        # Check if we have a cached result
        if hasattr(self, "_domain_complexity_cache"):
            if domain_cache_key in self._domain_complexity_cache:
                return self._domain_complexity_cache[domain_cache_key]
        else:
            # Initialize cache if it doesn't exist
            self._domain_complexity_cache = {}

        # For domains we're uncertain about, use LLM to assess complexity
        try:
            prompt = template_query_domain_complex.format(domain=domain)
            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            is_complex = "yes" in response.lower()

            # Cache the result for future use
            self._domain_complexity_cache[domain_cache_key] = is_complex
            return is_complex
        except Exception as e:
            # Fall back to heuristic on error
            logger.error(f"Error determining domain complexity: {str(e)}")
            return len(domain_lower.split()) >= 2  # More conservative fallback

    def search_papers_mroute(
        self, queries, end_date="", searched_docs=dict(), sources=["google"]
    ):
        # sources = ["google", "openalex"]
        output, id2docs, query_source_map, query_keywords2raw = (
            self.mretrival_processer.search_papers(
                querys=queries,
                end_date=end_date,
                searched_docs=searched_docs,
                sources=sources,
            )
        )
        return output, id2docs, query_source_map, query_keywords2raw

    def search_papers(self, queries, end_date="", searched_docs=dict()):
        results = {}
        if end_date == "":
            end_date = self.current_date

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=API_PARREL_REQUEST
        ) as executor:
            future_to_query = {
                executor.submit(
                    google_search_arxiv_id, query, API_TRY_COUNT, 10, end_date
                ): query
                for query in queries
            }
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results[query] = future.result(timeout=2)
                except Exception as e:
                    logger.error(f"Search failed for query {query}: {str(e)}")
                    results[query] = []

        logger.info(f"google_search_arxiv_id: {results}")

        unique_arxiv = set()
        original_arxiv = []
        for arxiv_ids in results.values():
            original_arxiv.extend(arxiv_ids)
            for arxiv_id in arxiv_ids:
                if arxiv_id not in searched_docs:
                    unique_arxiv.add(arxiv_id)

        logger.info(
            f"original num is: {len(original_arxiv)}, unique num is: {len(list(unique_arxiv))}"
        )

        id2docs = parallel_search_search_paper_from_arxiv(
            list(unique_arxiv), max_workers=API_PARREL_REQUEST, batch_size=8
        )

        output = {}
        for query, arxiv_ids in results.items():
            output[query] = [
                id2docs[arxiv_id] for arxiv_id in arxiv_ids if arxiv_id in id2docs
            ]
        return output, id2docs

    def calculate_sim_bge(
        self, query, docs, search_time="", score_thresh=BEGIN_SIM_THRESHOLD, source=""
    ):
        logger.info("calculate_sim_bge ...")
        relevace_docs = []
        irrelevace_docs = []

        golden_paper_info = [
            "Title:{}\nAbstract:{}Authors:{}".format(
                doc.get("title", ""),
                doc.get("abstract", ""),
                ";".join([one["name"] for one in doc.get("authors", [])]),
            )
            for doc in docs
        ]
        score_info_list = self.emd_model.get_score(
            query, golden_paper_info, batch_size=6
        )
        for doc, sim_score in zip(docs, score_info_list):
            simple_info = {
                "arxivId": doc["arxivId"],
                "paper_id": doc.get("paper_id", doc.get("arxivId")),
                "sim_score": sim_score,
                "source": source,
                "sim_info_details": {
                    "reason": "calculate sim from beg-m3",
                    "sim_score": sim_score,
                },
            }
            if sim_score >= score_thresh:
                relevace_docs.append(simple_info)
            else:
                irrelevace_docs.append(simple_info)

        return relevace_docs, irrelevace_docs

    def calculate_sim_pasa(
        self, query, docs, search_time="", score_thresh=PASS_SIM_THRESHOLD, source=""
    ):
        logger.info("calculate_sim_pasa..")
        relevace_docs = []
        irrelevace_docs = []

        prompt_template = (
            "You are an elite researcher in the field of AI, conducting research on {user_query}. "
            "Evaluate whether the following paper fully satisfies the detailed requirements of the user query "
            "and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\n"
            "Searched Paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
            "User Query: {user_query}\n\n"
            "Output format: Decision: True/False\nReason:... \nDecision:"
        )

        # 对doc进行过滤，如果存在字段缺失，那么这个数据丢弃
        docs = [doc for doc in docs if doc.get("title", "") and doc.get("abstract", "")]

        golden_paper_info = [
            prompt_template.format(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                user_query=query,
            )
            for paper in docs
        ]

        score_info_list = self.selector.batch_infer_score(golden_paper_info, 4)
        for doc, sim_score in zip(docs, score_info_list):
            simple_info = {
                "arxivId": doc.get("arxivId",""),
                "paper_id": doc.get("paper_id", doc.get("arxivId")),
                "sim_score": sim_score,
                "source": source,
                "sim_info_details": {
                    "reason": "calculate sim from pasa-scorer",
                    "sim_score": sim_score,
                },
            }
            if sim_score >= score_thresh:
                relevace_docs.append(simple_info)
            else:
                irrelevace_docs.append(simple_info)

        return relevace_docs, irrelevace_docs

    def rerank_score_bge(self, query, docs):
        logger.info("rerank_score_bge ...")

        golden_paper_info = [
            "Title:{}\nAbstract:{}Authors:{}".format(
                doc.get("title", ""),
                doc.get("abstract", ""),
                ";".join([one["name"] for one in doc.get("authors", [])]),
            )
            for doc in docs
        ]
        score_info_list = self.emd_model.get_score(
            query, golden_paper_info, batch_size=12
        )

        assert len(score_info_list) == len(golden_paper_info)
        paired_docs_scores = list(zip(docs, score_info_list))
        # 根据分数从高到低排序
        paired_docs_scores.sort(key=lambda x: x[1], reverse=True)

        scorted_docs = []
        for doc, score in paired_docs_scores:
            # logger.info(f"Score: {score}, Title: {doc.get('title', 'N/A')}")
            doc["rerank_score_bge"] = score
            scorted_docs.append(doc)

        return scorted_docs

    def calculate_similarity(
        self, query, docs, search_time="", score_thresh=0.5, source=""
    ):
        logger.debug(f"calculate_similarity, query: {query}; doc is: {docs[0].keys()}")
        relevace_docs = []
        irrelevace_docs = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=LLM_PARREL_NUM
        ) as executor:
            future_to_doc = {
                executor.submit(
                    _calculate_similarity_with_retry, query, search_time, doc
                ): doc
                for doc in docs
            }
            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                res = future.result(timeout=2)
                try:
                    if res:
                        doc.update(res)
                    else:
                        print(traceback.format_exc())
                        doc["sim_score"] = -1  # 失败则设为0，这个数据就不要了
                        doc["sim_info_details"] = {}
                except:
                    print(traceback.format_exc())
                    doc["sim_score"] = -1  # 失败则设为0，这个数据就不要了
                    doc["sim_info_details"] = {}
                finally:
                    simple_info = {
                        "arxivId": doc["arxivId"],
                        "paper_id": doc.get("paper_id", doc.get("arxivId")),
                        "sim_score": doc["sim_score"],
                        "sim_info_details": doc["sim_info_details"],
                        "source": source,
                    }
                    if doc["sim_score"] >= score_thresh:
                        relevace_docs.append(simple_info)
                    else:
                        irrelevace_docs.append(simple_info)

        return relevace_docs, irrelevace_docs

    def get_doc_references(self, doc_info):
        try:
            if "arxivId" in doc_info:
                if not doc_info.get("arxivId", ""):
                    return doc_info
                doc_info_new = get_doc_info_from_semantic_scholar_by_arxivid(
                    doc_info["arxivId"]
                )
                if doc_info_new is not None:
                    doc_info.update(doc_info_new)
                    return doc_info

            elif "PMID" in doc["info"]:
                # current doc has references, but the info is simple, get full info
                logger.info(f"source is pumbed, {doc['PMID']}")
                valid_pmid = [one["pmid"] for one in doc["references"]]
                already_info,valid_pmid = get_info_from_local(valid_pmid)
                pmid_info_lst = fetch_pubmed_json(valid_pmid)
                doc_info["references"] = already_info+pmid_info_lst

            elif "referenceWorksOpenAlex" in doc["info"]:
                references = search_doc_via_url_from_openalex(
                    doc_info["referenceWorksOpenAlex"]
                )
                doc_info["references"] = references
                return doc_info


        except Exception as e:
            logger.error(
                f"Failed to get references for {doc_info}: {traceback.format_exc()}"
            )
        return doc_info

    def generate_queries_from_docs(self, query, docs, searched_queries):
        results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=LLM_PARREL_NUM
        ) as executor:
            future_to_citation = {
                executor.submit(
                    _generate_query_from_reference,
                    query,
                    ref_doc,
                    searched_queries,
                ): [ref_doc, node]
                for ref_doc, node in docs
            }

            for future in concurrent.futures.as_completed(future_to_citation):
                ref_doc, node = future_to_citation[future]
                res = future.result(timeout=2)
                if res:
                    for new_q in res:
                        results.append([new_q, node])
        return results


# msearch_agent = MultiSearchAgent()
# query = [
#     "Provide me with some top-tier journal papers to expand my ideas on using synthetic data to augment supervised fine-tuning (SFT) while ensuring data quality and diversity, maintaining a balance between the two."
# ]
# res = msearch_agent.search_papers(query)
# print(res[0])
# print(res[1])

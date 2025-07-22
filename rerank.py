# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import re
from log import logger
import traceback
from datetime import datetime, timedelta
from local_request_v2 import get_from_llm
from global_config import RERANK_MODEL
from typing import List, Dict, Union
import json
import os


class Reranker(object):
    def rerank_query_and_doc_list(self,all_docs,user_query,score_name="sim_score"):
        """
        Reranks the top 20 documents based on authority and timeliness:
        1. Authority factors:
        - Conference/journal impact factor
        - Author prominence (h-index, citation count)
        2. Timeliness factors:
        - If query includes time constraints (recent, current, X years), respect those
        - Otherwise, favor more recent publications

        The reranking modifies the sim_score values in the document records.
        """
        logger.info("Reranking documents based on authority and timeliness...")
        self.user_query = user_query
        self.score_name = score_name

        if isinstance(all_docs, dict):
            all_docs = list(all_docs.values())
        # Get all documents and sort by original similarity score
        logger.info(f"Number of documents to rerank: {len(all_docs)}")

        # logger.info(f"all_docs: {all_docs}")
        sorted_docs = sorted(all_docs, key=lambda x: x.get(score_name, 0), reverse=True)

        # Only rerank top 20 documents
        top_docs = sorted_docs[:10]

        if not top_docs:
            logger.warning("No documents to rerank")
            return

        # Extract time constraints from the query if they exist
        time_constraints = self._extract_time_constraints(user_query)

        # Prepare prompt for LLM reranking
        prompt = self._prepare_reranking_prompt(top_docs, time_constraints)

        logger.info(f"prompt: {prompt}")
        try:
            # Use LLM to rerank documents
            reranked_results = self.llm_rerank_documents(prompt)

            # Update similarity scores based on reranking
            top_docs= self._update_documents_with_reranking(reranked_results, top_docs)

            logger.info("Document reranking completed successfully")
        except:
            logger.error(f"Error during document reranking: {traceback.format_exc()}")

        return top_docs

    def _extract_time_constraints(self, query):
        """
        Extract time constraints from the query string.
        Returns a dictionary with time constraint information.
        """
        time_constraints = {
            "has_time_requirement": False,
            "recency_required": False,
            "specific_timeframe": None,
            "year_limit": None
        }

        # Check for recency indicators
        recency_terms = ["recent", "latest", "newest", "current", "modern", "today", "last year"]
        if any(term in query for term in recency_terms):
            time_constraints["has_time_requirement"] = True
            time_constraints["recency_required"] = True

        # Check for specific timeframes (e.g., "in the last 3 years", "since 2020")
        year_pattern = r"(since|after|from|in the last|within)?\s*(\d{1,2})\s*(year|yr)s?"
        specific_year_pattern = r"(since|after|from)?\s*(20\d{2}|19\d{2})"

        year_match = re.search(year_pattern, query)
        if year_match:
            time_constraints["has_time_requirement"] = True
            time_constraints["specific_timeframe"] = f"last {year_match.group(2)} years"
            time_constraints["year_limit"] = int(year_match.group(2))

        specific_year_match = re.search(specific_year_pattern, query)

        if specific_year_match:
            time_constraints["has_time_requirement"] = True
            time_constraints["specific_timeframe"] = f"since {specific_year_match.group(2)}"
            time_constraints["year_limit"] = int(specific_year_match.group(2))

        logger.info(f"Extracted time constraints: {time_constraints}")
        return time_constraints

    def _prepare_reranking_prompt(self, docs, time_constraints):
        """
        Prepare a prompt for the LLM to rerank documents.
        """
        current_year = datetime.now().year

        prompt = (
            f"Please rerank the following {len(docs)} academic papers in response to the query: '{self.user_query}'\n\n"
            "Consider these factors in your reranking:\n"
            "1. Authority:\n"
            "   - Publication venue prestige (top conferences/journals rank higher)\n"
            "   - Author prominence (authors with higher h-index or citation counts rank higher)\n"
            "2. Timeliness:\n"
        )

        if time_constraints["has_time_requirement"]:
            if time_constraints["recency_required"]:
                prompt += "   - The query specifically asks for recent/current papers, so strongly prefer newer papers\n"

            if time_constraints["specific_timeframe"]:
                prompt += f"   - The query asks for papers {time_constraints['specific_timeframe']}, so prefer papers within this timeframe\n"
        else:
            prompt += "   - Generally prefer more recent papers, but don't overly penalize influential older papers\n"

        prompt += (
            "3. Maintain reasonable relevance to the original query\n\n"
            "For each paper, provide:\n"
            "1. A new numerical rank (1 being the highest)\n"
            "2. A brief justification (1-2 sentences)\n"
            "3. A new relevance score between 0-1 that incorporates both relevance and the factors above\n\n"
            "List of papers with original relevance scores (title, year, venue, authors, relevance):\n"
        )

        # Add document details to prompt
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown Title")
            year = doc.get("year", doc.get("publicationYear","Unknown Year"))
            venue = doc.get("venue", doc.get("journal", "Unknown Venue"))
            authors = ", ".join([a.get("name", "Unknown") for a in doc.get("authors", [])][:3])
            if len(doc.get("authors", [])) > 3:
                authors += " et al."
            sim_score = doc.get(self.score_name, 0.0)

            # Include timeliness information if relevant
            age_note = ""
            if year and year != "Unknown Year":
                paper_age = current_year - int(year)
                if time_constraints["has_time_requirement"] and time_constraints["year_limit"]:
                    if paper_age <= time_constraints["year_limit"]:
                        age_note = f" (meets {time_constraints['specific_timeframe']} requirement)"
                    else:
                        age_note = f" (outside {time_constraints['specific_timeframe']} requirement)"

            prompt += f"{i}. {title} ({year}{age_note}) - {venue}\n   Authors: {authors}\n   Original relevance: {sim_score:.3f}\n\n"

        prompt += "Please provide your reranking with new scores and concise justifications in the following format for each document:\n"
        prompt += "Document [index]: [score] - [justification]\n"
        prompt += "For example:\n"
        prompt += "Document 1: 9.5 - Highly relevant as it directly addresses the query topic with empirical evidence.\n"
        prompt += "Document 2: 7.0 - Somewhat relevant but focuses on a tangential aspect of the query.\n"

        return prompt

    def llm_rerank_documents(self, prompt):
        """
        Use an LLM to rerank documents based on the provided prompt.

        Returns a list of dictionaries containing reranked documents information.
        """
        logger.info(f"llm_rerank_documents: {prompt}")
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = get_from_llm(prompt, model_name=RERANK_MODEL)
                logger.info(f"response: {response}")
                # Parse the LLM response to extract reranking information
                reranked_results = self._parse_llm_reranking_response(response)

                # If we got valid results, return them
                if reranked_results:
                    logger.info(f"reranked_results: {reranked_results}")
                    return reranked_results

                # Otherwise, retry
                logger.warning(f"Empty reranking results received (attempt {retry_count + 1}/{max_retries}). Retrying...")
                retry_count += 1

                # Add a slight modification to the prompt to encourage different response
                if retry_count < max_retries:
                    prompt += f"\n\nPlease ensure you provide a complete reranking for all documents in the exact format requested."

            except Exception as e:
                logger.error(f"LLM reranking failed (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                retry_count += 1

                if retry_count < max_retries:
                    logger.info(f"Retrying reranking...")

        # If we've exhausted all retries, return empty list
        logger.error(f"Failed to get valid reranking results after {max_retries} attempts")
        return []

    def _parse_llm_reranking_response(self, response: str) -> List[Dict[str, Union[float, str]]]:
        """Parse LLM reranking response to extract scores and justifications.

        Args:
            response: LLM response string

        Returns:
            List of dictionaries with 'score' and 'justification' keys
        """
        results = []
        # Look for lines with the pattern "Document X: Y.Z - justification"
        pattern = r"Document\s+(\d+):\s+([\d.]+)\s+-\s+(.+)"

        for line in response.split("\n"):
            match = re.search(pattern, line)
            if match:
                document_idx = int(match.group(1)) - 1  # Convert to zero-based index
                score = float(match.group(2))
                justification = match.group(3).strip()

                # Ensure the document_idx is valid
                while len(results) <= document_idx:
                    results.append({})

                results[document_idx] = {
                    "rerank_score": score,
                    "justification": justification
                }

        return [r for r in results if r]  #


    def _update_documents_with_reranking(self, reranked_results, original_docs):
        """
        Update document sim_scores based on the reranking results.
        """
        if not reranked_results:
            logger.warning("No reranking results to apply")
            return

        # Update the original documents directly using their index position
        updates_applied = 0
        reraked_docs = []
        for idx, result in enumerate(reranked_results):
            if idx < len(original_docs):
                # Get reranking information
                rerank_score = result.get("rerank_score")
                justification = result.get("justification", "")

                if rerank_score is not None:
                    # Directly update the document at this position
                    doc = original_docs[idx]
                    cur = {
                        "title":doc["title"],
                        f"{self.score_name}":doc.get(self.score_name, 0.0),
                        "rerank_score":rerank_score,
                        "rerank_justification":justification,
                        "user_query":self.user_query,
                        "score_name":self.score_name
                    }
                    reraked_docs.append(cur)

                    updates_applied += 1


        logger.info(f"Applied reranking updates to {updates_applied} documents")

        return reraked_docs
def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = "".join(letters)
    return result.lower()


def cal_micro(pred_set, label_set):
    if not label_set and not pred_set:
        print("Warning: Both pred_set and label_set are empty.")
        return 0, 0, 0
    if not label_set:
        print("Warning: label_set is empty.")
        return 0, len(pred_set), 0
    if not pred_set:
        return 0, 0, len(label_set)
    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    return tp, fp, fn


#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Global configuration settings for Scholar Paper Agent Retrieval
# ==================================================================
import os
import json
import arxiv
from typing import Dict, List, Any

# Debug mode
DEBUG = False

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================
API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "your_openai_api_key_here",
)
ENDPOINT = os.getenv(
    "OPENAI_ENDPOINT",
    "https://api.openai.com/v1",
)
DEPLOYMENT_NAME = "gpt-4o"

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
SAVE_ID2DOCS = True
RELEVANCE_SCORE = 0.5
WEB_RETRY_NUM = 1

# Query threshold settings
QUERY_LOW_THRESHOLD = 0.2
QUERY_HIGH_THRESHOLD = 0.8
CORRECT_SCORE_THRESHOLD = 0.8
EXPAND_SCORE_THRESHOLD = 0.85
QUERY_TO_SEARCH_THRESHOLD = 0.85

# Generation settings
LENGTH_GEN_QUERY_FROM_CITATION = 12288

# =============================================================================
# WEB API CONFIGURATION
# =============================================================================
TRY_COUNT = 4
LLM_TRY_COUNT = 4
LLM_PARALLEL_NUM = 4
LLM_MODEL_NAME = "Qwen3-8B"


API_TRY_COUNT = 4
API_PARALLEL_REQUEST = 1

SLEEP_TIME_LLM = 2.0

# =============================================================================
# SEARCH HYPERPARAMETERS
# =============================================================================
DO_FUSION_JUDGE = True
FUSION_TEMPLATE = "AUTOMATIC"  # Options: "WITHEXPLAIN", "AUTOMATIC"

# Query processing settings
QUERY_NUM_PRUNED = 2  # Number of queries to use for search
RETRIEVAL_QUERY_BATCH_SIZE = 6  # Batch size for query processing to avoid excessive searching

# Document processing settings
DOCS_TO_EXPAND = 40
REFERENCE_DOC_PRUNED = 20  # Number of references to extract from each relevant document
REFERENCE_OCCUR_FREQUENCY = 0.6
REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY = 2  # Number of reference docs used to generate new queries

# Similarity thresholds
REFERENCE_DOC_SIM_THRESHOLD = 0.6
BEGIN_SIM_THRESHOLD = 0.5
PASS_SIM_THRESHOLD = 0.5

# Search routes configuration
SEARCH_ROUTES: List[str] = ["arxiv", "openalex"]

# =============================================================================
# EXTERNAL API KEYS
# =============================================================================
# Register at: https://google.serper.dev/search
GOOGLE_SERPER_KEY = os.getenv("GOOGLE_SERPER_KEY", "xxx")

# Semantic Scholar API key (currently invalid)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("S2_API_KEY", "")

# =============================================================================
# SEARCH FEATURES
# =============================================================================
DO_REFERENCE_SEARCH = False  # Toggle reference-based search
RERANK =os.getenv("DO_RERANK",True)

KEY_WORDS_NUM =2
LLM_PARREL_NUM=2
# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
PROXIES: Dict[str, str] = {
    "http": os.getenv("HTTP_PROXY", "http://localhost:1080"),
    "https": os.getenv("HTTPS_PROXY", "http://localhost:1080")
}

# ArXiv client configuration
ARXIV_CLIENT = arxiv.Client(delay_seconds=0.05)

# =============================================================================
# RERANKING CONFIGURATION
# =============================================================================
ENABLE_RERANK = False
RERANK_MODEL = "Qwen3-8B"

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config() -> bool:
    """
    Validate essential configuration settings.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [API_KEY, ENDPOINT]

    if not all(key and key != "your_openai_api_key_here" for key in required_keys):
        print("Warning: OpenAI API configuration is incomplete")
        return False

    if QUERY_LOW_THRESHOLD >= QUERY_HIGH_THRESHOLD:
        print("Error: QUERY_LOW_THRESHOLD must be less than QUERY_HIGH_THRESHOLD")
        return False

    return True

# Validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("Configuration validation passed")

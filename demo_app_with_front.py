# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import traceback
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os
import traceback

# 添加项目路径
from pipeline_spar import AcademicSearchTree
from search_engine import MultiSearchAgent

from log import logger

app = FastAPI(title="Scholar Paper Search API with Frontend", version="1.0.0")

# 请求模型
class SearchRequest(BaseModel):
    queries: List[str]
    sources: Optional[List[str]] = ["openalex"]
    end_date: Optional[str] = ""
    max_workers: Optional[int] = 3
    batch_size: Optional[int] = 10
    google_serper_key: Optional[str] = ""  # 添加Google Serper Key字段
    use_advanced_search: Optional[bool] = True  # 是否使用高级搜索（包含query改写和rerank）
    max_depth: Optional[int] = 1  # 搜索树最大深度
    relevance_doc_num: Optional[int] = 10  # 相关文档数量
    similarity_threshold: Optional[float] = 0.5  # 相似度阈值

# 响应模型
class SearchResponse(BaseModel):
    status: str
    total_papers: int
    query_results: Dict[str, List[Dict[str, Any]]]
    all_papers: Dict[str, Dict[str, Any]]
    query_source_map: Dict[str, str]
    search_tree: Optional[Dict[str, Any]] = None  # 搜索树结构（高级搜索模式）

# 初始化搜索引擎
multi_search_agent = MultiSearchAgent()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    html_file = "./index.html"
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Frontend file not found</h1>
                    <p>Please make sure index.html exists in the templates folder</p>
                </body>
            </html>
            """,
            status_code=404
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "message": "Scholar Paper Search API is running"}

@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    搜索学术论文API

    Args:
        request: 搜索请求参数

    Returns:
        SearchResponse: 搜索结果
    """
    try:
        logger.info(f"Received search request: {request}")

        # 验证输入
        if not request.queries:
            raise HTTPException(status_code=400, detail="Queries list cannot be empty")

        # 临时设置Google Serper Key环境变量
        if request.google_serper_key:
            os.environ["GOOGLE_SERPER_KEY"] = request.google_serper_key
            logger.info(f"Google Serper Key set from request: {request.google_serper_key}")

        if request.use_advanced_search:
            # 使用高级搜索（包含query改写、意图判断、rerank等完整pipeline）
            return await _advanced_search(request)
        else:
            # 使用简单搜索
            return await _simple_search(request)

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

def standardize_paper_data(paper_data, paper_id=None, source='unknown'):
    """
    标准化论文数据格式

    Args:
        paper_data: 原始论文数据（字典或对象）
        paper_id: 论文ID（可选，如果paper_data中没有）
        source: 数据来源标识

    Returns:
        dict: 标准化后的论文数据
    """
    try:
        # 确保paper_data是字典格式
        if not isinstance(paper_data, dict):
            if hasattr(paper_data, '__dict__'):
                paper_data = paper_data.__dict__
            else:
                logger.warning(f"Cannot convert paper_data to dict: {type(paper_data)}")
                # 创建基本字典结构
                paper_data = {
                    'title': str(paper_data) if paper_data else 'Unknown',
                    'abstract': '',
                    'paper_id': paper_id or 'unknown'
                }

        # 标准化论文数据格式
        standardized_paper = {
            'paper_id': paper_data.get('paper_id') or paper_data.get('arxivId') or paper_data.get('id') or paper_id or 'unknown',
            'title': paper_data.get('title', 'No title available'),
            'authors': paper_data.get('authors', []),
            'abstract': paper_data.get('abstract', ''),
            'year': paper_data.get('publicationYear') or paper_data.get('year', ''),
            'publicationYear': paper_data.get('publicationYear') or paper_data.get('year', ''),
            'url': paper_data.get('url', ''),
            'doi': paper_data.get('doi', ''),
            'citationCount': paper_data.get('citationCount', 0),
            'arxivId': paper_data.get('arxivId', ''),
            'arxiv_url': paper_data.get('arxiv_url', paper_data.get('arxivUrl', '')),
            'openalex_id': paper_data.get('openalex_id', ''),
            'fieldsOfStudy': paper_data.get('fieldsOfStudy', ''),
            'referenceCount': paper_data.get('referenceCount', 0),
            'isOpen': paper_data.get('isOpen', False),
            'source': paper_data.get('source', source),
            'sim_score': paper_data.get('sim_score', 0.0),
            'relevance_details': paper_data.get('sim_info_details', paper_data.get('relevance_details', {}))
        }

        # 处理作者字段 - 确保格式一致
        if standardized_paper['authors']:
            if isinstance(standardized_paper['authors'], list):
                author_names = []
                for author in standardized_paper['authors']:
                    if isinstance(author, dict) and 'name' in author:
                        author_names.append(author['name'])
                    elif isinstance(author, str):
                        author_names.append(author)
                    else:
                        author_names.append(str(author))
                standardized_paper['authors'] = author_names
            elif isinstance(standardized_paper['authors'], str):
                # 如果authors是字符串，尝试分割
                standardized_paper['authors'] = [name.strip() for name in standardized_paper['authors'].split(',') if name.strip()]

        # 处理URL字段 - 提供多种链接选项
        if standardized_paper['openalex_id'] and not standardized_paper['url']:
            standardized_paper['url'] = f"https://openalex.org/{standardized_paper['openalex_id']}"

        if standardized_paper['arxivId'] and not standardized_paper['arxiv_url']:
            standardized_paper['arxiv_url'] = f"https://arxiv.org/abs/{standardized_paper['arxivId']}"

        # 处理其他可能的URL字段
        if paper_data.get('pdf_url'):
            standardized_paper['pdf_url'] = paper_data['pdf_url']
        if paper_data.get('openaccess_url'):
            standardized_paper['openaccess_url'] = paper_data['openaccess_url']
        if paper_data.get('landing_page_url'):
            standardized_paper['landing_page_url'] = paper_data['landing_page_url']

        # 确保数值类型字段的类型正确
        try:
            standardized_paper['citationCount'] = int(standardized_paper['citationCount'] or 0)
        except (ValueError, TypeError):
            standardized_paper['citationCount'] = 0

        try:
            standardized_paper['referenceCount'] = int(standardized_paper['referenceCount'] or 0)
        except (ValueError, TypeError):
            standardized_paper['referenceCount'] = 0

        try:
            standardized_paper['sim_score'] = float(standardized_paper['sim_score'] or 0.0)
        except (ValueError, TypeError):
            standardized_paper['sim_score'] = 0.0

        # 确保year是字符串类型
        if standardized_paper['year']:
            standardized_paper['year'] = str(standardized_paper['year'])
            standardized_paper['publicationYear'] = standardized_paper['year']

        return standardized_paper

    except Exception as e:
        logger.error(f"Error standardizing paper data: {str(e)}")
        logger.error(f"Paper data: {paper_data}")
        # 返回错误占位符
        return {
            'paper_id': str(paper_id) if paper_id else 'error',
            'title': f'Error processing paper: {paper_id or "unknown"}',
            'abstract': f'Error: {str(e)}',
            'authors': [],
            'year': '',
            'publicationYear': '',
            'url': '',
            'doi': '',
            'citationCount': 0,
            'arxivId': '',
            'arxiv_url': '',
            'openalex_id': '',
            'fieldsOfStudy': '',
            'referenceCount': 0,
            'isOpen': False,
            'source': 'error',
            'sim_score': 0.0,
            'relevance_details': {}
        }


def process_paper_collection(papers_data, source='unknown', is_dict_format=True):
    """
    批量处理论文数据集合

    Args:
        papers_data: 论文数据集合（字典或列表）
        source: 数据来源标识
        is_dict_format: 是否为字典格式（True: {paper_id: paper_data}, False: [paper_data, ...])

    Returns:
        tuple: (papers_list, papers_dict) - 论文列表和论文字典
    """
    papers_list = []
    papers_dict = {}

    try:
        if is_dict_format and isinstance(papers_data, dict):
            # 处理字典格式 {paper_id: doc_info}
            for paper_id, paper_info in papers_data.items():
                try:
                    standardized_paper = standardize_paper_data(paper_info, paper_id, source)
                    papers_list.append(standardized_paper)
                    papers_dict[paper_id] = standardized_paper
                except Exception as paper_error:
                    logger.error(f"Error processing paper {paper_id}: {str(paper_error)}")
                    # 创建错误占位符
                    error_paper = {
                        'paper_id': str(paper_id),
                        'title': f'Error processing paper: {paper_id}',
                        'abstract': f'Error: {str(paper_error)}',
                        'authors': [],
                        'year': '',
                        'url': '',
                        'citationCount': 0,
                        'source': 'error'
                    }
                    papers_list.append(error_paper)
                    papers_dict[str(paper_id)] = error_paper

        elif isinstance(papers_data, list):
            # 处理列表格式 [paper_data, ...]
            for i, paper_info in enumerate(papers_data):
                try:
                    # 尝试获取paper_id，如果没有则生成一个
                    if isinstance(paper_info, dict):
                        paper_id = paper_info.get('paper_id', paper_info.get('arxivId', paper_info.get('id', f'paper_{i}')))
                    else:
                        paper_id = f'paper_{i}'

                    standardized_paper = standardize_paper_data(paper_info, paper_id, source)
                    papers_list.append(standardized_paper)
                    papers_dict[paper_id] = standardized_paper
                except Exception as paper_error:
                    logger.error(f"Error processing paper {i}: {str(paper_error)}")
                    continue
        else:
            logger.warning(f"Unexpected papers_data format: {type(papers_data)}")

    except Exception as e:
        logger.error(f"Error processing paper collection: {str(e)}")

    return papers_list, papers_dict


async def _advanced_search(request: SearchRequest) -> SearchResponse:
    """
    高级搜索模式，使用AcademicSearchTree进行完整的搜索流程
    包含query改写、意图判断、rerank等功能
    """
    try:
        # 为每个查询创建搜索树
        all_results = {}
        all_papers = {}
        query_source_map = {}
        search_trees = {}

        for query in request.queries:
            logger.info(f"Processing query with advanced search: {query}")

            try:
                # 创建学术搜索树实例
                search_agent = AcademicSearchTree(
                    max_depth=request.max_depth,
                    max_docs=request.relevance_doc_num,
                    similarity_threshold=request.similarity_threshold
                )

                # 执行搜索（包含完整pipeline）
                sorted_docs = search_agent.search(query, end_date=request.end_date)

                if not sorted_docs:
                    logger.warning(f"No documents found for query: {query}")
                    all_results[query] = []
                    query_source_map[query] = "advanced_search"
                    continue

                logger.info(f"Advanced search returned {len(sorted_docs)} documents for query: {query}")

                # 使用统一的数据处理函数
                if isinstance(sorted_docs, dict):
                    papers_list, papers_dict = process_paper_collection(sorted_docs, 'advanced_search', is_dict_format=True)
                elif isinstance(sorted_docs, list):
                    papers_list, papers_dict = process_paper_collection(sorted_docs, 'advanced_search', is_dict_format=False)
                else:
                    logger.warning(f"Unexpected sorted_docs format: {type(sorted_docs)}")
                    papers_list, papers_dict = [], {}

                all_results[query] = papers_list
                all_papers.update(papers_dict)
                query_source_map[query] = "advanced_search"

                # 保存搜索树结构
                try:
                    if hasattr(search_agent, 'root') and search_agent.root:
                        search_trees[query] = search_agent.root.convert_to_dict()
                    else:
                        logger.warning(f"No search tree root found for query: {query}")
                except Exception as tree_error:
                    logger.error(f"Error converting search tree to dict: {str(tree_error)}")

            except Exception as query_error:
                logger.error(f"Error processing query '{query}': {str(query_error)}")
                logger.error(f"Query error traceback: {traceback.format_exc()}")
                # 为失败的查询添加空结果
                all_results[query] = []
                query_source_map[query] = "error"

        # 构造响应
        response = SearchResponse(
            status="success",
            total_papers=len(all_papers),
            query_results=all_results,
            all_papers=all_papers,
            query_source_map=query_source_map,
            search_tree=search_trees
        )

        logger.info(f"Advanced search completed successfully. Found {len(all_papers)} papers")
        return response

    except Exception as e:
        logger.error(f"Advanced search failed: {str(e)}")
        logger.error(f"Advanced search traceback: {traceback.format_exc()}")
        # 返回错误响应
        return SearchResponse(
            status="error",
            total_papers=0,
            query_results={},
            all_papers={},
            query_source_map={},
            search_tree={"error": str(e)}
        )


async def _simple_search(request: SearchRequest) -> SearchResponse:
    """
    简单搜索模式，使用MultiSearchAgent进行基础搜索
    """
    try:
        # 更新搜索引擎参数
        multi_search_agent.max_workers = request.max_workers
        multi_search_agent.batch_size = request.batch_size

        # 执行搜索
        query_results, all_papers, query_source_map, query_keywords2raw = multi_search_agent.search_papers(
            querys=request.queries,
            sources=request.sources,
            end_date=request.end_date,
            searched_docs={},
            rerank=True
        )

        # 标准化处理结果数据，确保与前端期望的格式一致
        standardized_query_results = {}
        standardized_all_papers = {}

        # 处理query_results - 每个query对应一个论文列表
        for query, papers in query_results.items():
            papers_list, _ = process_paper_collection(papers, 'simple_search', is_dict_format=False)
            standardized_query_results[query] = papers_list

        # 处理all_papers - 字典格式 {paper_id: paper_data}
        _, standardized_all_papers = process_paper_collection(all_papers, 'simple_search', is_dict_format=True)

        # 构造响应
        response = SearchResponse(
            status="success",
            total_papers=len(standardized_all_papers),
            query_results=standardized_query_results,
            all_papers=standardized_all_papers,
            query_source_map=query_source_map,
            search_tree=None  # 简单搜索不生成搜索树
        )

        logger.info(f"Simple search completed successfully. Found {len(standardized_all_papers)} papers")
        return response

    except Exception as e:
        logger.error(f"Simple search failed: {str(e)}")
        logger.error(f"Simple search traceback: {traceback.format_exc()}")
        # 返回错误响应而不是重新抛出异常
        return SearchResponse(
            status="error",
            total_papers=0,
            query_results={},
            all_papers={},
            query_source_map={},
            search_tree={"error": str(e)}
        )


@app.get("/sources")
async def get_available_sources():
    """获取可用的搜索源"""
    return {
        "available_sources": ["arxiv", "openalex", "pubmed"],  # 移除semantic scholar
        "description": {
            "arxiv": "ArXiv papers via Google Scholar",
            "openalex": "OpenAlex database",
            "pubmed": "PubMed medical papers"
        }
    }

@app.get("/search-modes")
async def get_search_modes():
    """获取搜索模式信息"""
    return {
        "modes": {
            "simple": {
                "name": "Simple Search",
                "description": "Basic multi-source search with reranking",
                "features": ["Multi-source search", "Basic reranking", "Fast results"]
            },
            "advanced": {
                "name": "Advanced Search",
                "description": "Complete pipeline with query rewriting, intent analysis, and advanced reranking",
                "features": [
                    "Query rewriting and expansion",
                    "Intent analysis and classification",
                    "Reference-based search",
                    "Advanced reranking algorithms",
                    "Search tree visualization",
                    "Iterative refinement"
                ]
            }
        }
    }

if __name__ == "__main__":
    # 创建templates目录
    templates_dir = "./api/templates"
    os.makedirs(templates_dir, exist_ok=True)

    uvicorn.run(
        "demo_app_with_front:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
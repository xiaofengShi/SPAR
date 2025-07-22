from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from search_engine import MultiSearchAgent
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scholar Paper Search API", version="1.0.0")

# 请求模型
class SearchRequest(BaseModel):
    queries: List[str]
    sources: Optional[List[str]] = ["arxiv", "semantic", "openalex"]
    end_date: Optional[str] = ""
    max_workers: Optional[int] = 3
    batch_size: Optional[int] = 10

# 响应模型
class SearchResponse(BaseModel):
    status: str
    total_papers: int
    query_results: Dict[str, List[Dict[str, Any]]]
    all_papers: Dict[str, Dict[str, Any]]
    query_source_map: Dict[str, str]

# 初始化搜索引擎
search_agent = MultiSearchAgent()

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

        # 更新搜索引擎参数
        search_agent.max_workers = request.max_workers
        search_agent.batch_size = request.batch_size

        # 执行搜索
        query_results, all_papers, query_source_map, query_keywords2raw = search_agent.search_papers(
            querys=request.queries,
            sources=request.sources,
            end_date=request.end_date,
            searched_docs={},
            rerank=True
        )

        # 构造响应
        response = SearchResponse(
            status="success",
            total_papers=len(all_papers),
            query_results=query_results,
            all_papers=all_papers,
            query_source_map=query_source_map
        )

        logger.info(f"Search completed successfully. Found {len(all_papers)} papers")
        return response

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/sources")
async def get_available_sources():
    """获取可用的搜索源"""
    return {
        "available_sources": ["arxiv", "semantic", "openalex", "pubmed"],
        "description": {
            "arxiv": "ArXiv papers via Google Scholar",
            "semantic": "Semantic Scholar database",
            "openalex": "OpenAlex database",
            "pubmed": "PubMed medical papers"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "search_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
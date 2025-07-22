import requests
import json

class ScholarSearchClient:
    """简易的搜索API客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def search_papers(self, queries, sources=None, end_date="", max_workers=3, batch_size=10):
        """搜索论文"""
        if sources is None:
            sources = ["arxiv", "semantic"]

        payload = {
            "queries": queries,
            "sources": sources,
            "end_date": end_date,
            "max_workers": max_workers,
            "batch_size": batch_size
        }

        try:
            response = requests.post(f"{self.base_url}/search", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Search request failed: {e}")
            return None

    def get_sources(self):
        """获取可用搜索源"""
        try:
            response = requests.get(f"{self.base_url}/sources")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get sources failed: {e}")
            return None

    def health_check(self):
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    client = ScholarSearchClient()

    # 健康检查
    print("Health check:", client.health_check())

    # 获取可用源
    print("Available sources:", client.get_sources())

    # 搜索论文
    queries = ["machine learning for NLP", "transformer models"]
    results = client.search_papers(
        queries=queries,
        sources=["arxiv", "semantic"],
        max_workers=2
    )

    if results:
        print(f"Found {results['total_papers']} papers")
        for query, papers in results['query_results'].items():
            print(f"Query '{query}': {len(papers)} papers")
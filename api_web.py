# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

# pip install biopython
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from global_config import GOOGLE_SERPER_KEY, PROXIES, ARXIV_CLIENT
from local_db_v2 import db_path, ArxivDatabase
from log import logger
from typing import Optional, Dict, List, Any  # 引入类型提示
from utils import get_md5, keep_letters
import arxiv
import json
import logging
import os
import random
import re
import requests
import time
import traceback

Entrez.email = "xxx@163.com"  # 替换为你的邮箱


def fetch_pubmed_json(pmid_list):
    """
    Retrieve structured PubMed paper information with enhanced metadata extraction.

    Extracts comprehensive information including:
    - ArXiv identifiers (when available)
    - Publication dates (year, month, day)
    - Research fields/MeSH terms
    - DOI identifiers
    - Journal information
    - Citation counts (when available)
    - Reference information

    Args:
        pmid_list: List of PubMed IDs to retrieve

    Returns:
        List of dictionaries containing structured paper information
    """
    if not pmid_list:
        logger.warning("Empty PMID list provided to fetch_pubmed_json")
        return []

    try:
        ids = ",".join(pmid_list)
        handle = Entrez.efetch(
            db="pubmed",
            id=ids,
            rettype="abstract",
            retmode="xml",
            httppost={"proxies": PROXIES}  # 使用局部代理
        )
        records = Entrez.read(handle)
        handle.close()

        paper_list = []
        for article in records["PubmedArticle"]:
            try:
                # print(article)
                # Get basic article metadata
                article_data = article["MedlineCitation"]["Article"]
                pmid = str(article["MedlineCitation"]["PMID"])


                # Extract title
                title = article_data["ArticleTitle"]

                # Extract abstract with fallbacks
                abstract = ""
                if "Abstract" in article_data:
                    abstract_parts = article_data["Abstract"].get("AbstractText", [])
                    if isinstance(abstract_parts, list):
                        abstract = " ".join([str(part) for part in abstract_parts])
                    else:
                        abstract = str(abstract_parts)

                # Extract authors with fallbacks
                authors = []
                if "AuthorList" in article_data:
                    for author in article_data["AuthorList"]:
                        if "LastName" in author and "ForeName" in author:
                            authors.append(f"{author['LastName']} {author['ForeName']}")
                        elif "LastName" in author:
                            authors.append(author["LastName"])
                        elif "CollectiveName" in author:
                            authors.append(author["CollectiveName"])

                # Extract publication date
                pub_date = {}
                if "PubDate" in article_data["Journal"]["JournalIssue"]:
                    date_info = article_data["Journal"]["JournalIssue"]["PubDate"]
                    if "Year" in date_info:
                        pub_date["year"] = date_info["Year"]
                    if "Month" in date_info:
                        pub_date["month"] = date_info["Month"]
                    if "Day" in date_info:
                        pub_date["day"] = date_info["Day"]

                # Format publication date
                publication_date = ""
                if "year" in pub_date:
                    publication_date = pub_date["year"]
                    if "month" in pub_date:
                        month_map = {
                            "Jan": "01",
                            "Feb": "02",
                            "Mar": "03",
                            "Apr": "04",
                            "May": "05",
                            "Jun": "06",
                            "Jul": "07",
                            "Aug": "08",
                            "Sep": "09",
                            "Oct": "10",
                            "Nov": "11",
                            "Dec": "12",
                        }
                        month = pub_date["month"]
                        if month in month_map:
                            month = month_map[month]
                        publication_date += month.zfill(2)
                        if "day" in pub_date:
                            publication_date += pub_date["day"].zfill(2)

                # Extract journal information
                journal_info = {}
                if "Journal" in article_data:
                    journal = article_data["Journal"]
                    # Convert Title to plain string
                    journal_title = journal.get("Title", "")
                    journal_info["name"] = str(journal_title) if journal_title else ""

                    # Convert ISSN to plain string
                    journal_issn = journal.get("ISSN", "")
                    journal_info["issn"] = str(journal_issn) if journal_issn else ""

                    if "JournalIssue" in journal:
                        issue = journal["JournalIssue"]

                        # Convert Volume to plain string
                        volume = issue.get("Volume", "")
                        journal_info["volume"] = str(volume) if volume else ""

                        # Convert Issue to plain string
                        issue_num = issue.get("Issue", "")
                        journal_info["issue"] = str(issue_num) if issue_num else ""

                # Extract DOI and other identifiers
                identifiers = {}
                if "ArticleIdList" in article:
                    for id_item in article["ArticleIdList"]:
                        id_type = id_item.attributes.get("IdType", "")
                        if id_type:
                            identifiers[id_type] = str(id_item)

                # Extract ArXiv ID if available
                arxiv_id = ""
                arxiv_url = ""
                if "arxiv" in identifiers:
                    arxiv_id = identifiers["arxiv"]
                    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

                # Extract MeSH terms and keywords
                mesh_terms = []
                if "MeshHeadingList" in article["MedlineCitation"]:
                    for mesh in article["MedlineCitation"]["MeshHeadingList"]:
                        if "DescriptorName" in mesh:
                            mesh_terms.append(str(mesh["DescriptorName"]))

                keywords = []
                if "KeywordList" in article["MedlineCitation"]:
                    for keyword_list in article["MedlineCitation"]["KeywordList"]:
                        keywords.extend([str(keyword) for keyword in keyword_list])

                # Get citation information using related articles
                citation_count = 0
                try:
                    # Check if citation data available in the article
                    if "CommentsCorrectionsList" in article["PubmedData"]:
                        for comment in article["PubmedData"]["CommentsCorrectionsList"]:
                            if "RefType" in comment and comment["RefType"] == "Cites":
                                citation_count += 1

                    # If citation count is still 0, try to get it from PubMed Central
                    if citation_count == 0 and "pmc" in identifiers:
                        pmc_id = identifiers["pmc"]
                        # This requires a separate API call to PMC
                        try:
                            cite_handle = Entrez.elink(
                                dbfrom="pmc",
                                db="pubmed",
                                linkname="pmc_pubmed_cited",
                                id=pmc_id,
                            )
                            cite_results = Entrez.read(cite_handle)
                            cite_handle.close()

                            if cite_results and len(cite_results[0]["LinkSetDb"]) > 0:
                                citation_count = len(
                                    cite_results[0]["LinkSetDb"][0]["Link"]
                                )
                        except Exception as e:
                            logger.warning(
                                f"Error getting citation data from PMC: {str(e)}"
                            )
                except Exception as e:
                    logger.warning(f"Error extracting citation information: {str(e)}")

                # Get references from the article
                references = []
                try:
                    # First try to extract from the bibliography if available
                    if "PubmedData" in article and "ReferenceList" in article["PubmedData"]:
                        for ref_list in article["PubmedData"]["ReferenceList"]:
                            for ref in ref_list["Reference"]:
                                # logger.info(f"refinfo: {ref}")
                                ref_info = {
                                    "title": str(ref.get("ArticleTitle", "")) if ref.get("ArticleTitle") else "",
                                    "citation": str(ref.get("Citation", "")) if ref.get("Citation") else "",
                                    "pmid": ""
                                }

                                # Extract PMID if available
                                if "ArticleIdList" in ref:
                                    for id_item in ref["ArticleIdList"]:
                                        if id_item.attributes.get("IdType", "") == "pubmed":
                                            ref_info["pmid"] = str(id_item)
                                # logger.info(f"ref_info extract: {ref_info}")
                                references.append(ref_info)

                    # If no references found, try another approach using cross-references
                    if not references and "pmc" in identifiers:
                        pmc_id = identifiers["pmc"]
                        try:
                            ref_handle = Entrez.elink(
                                dbfrom="pmc",
                                db="pubmed",
                                linkname="pmc_pubmed_refs",
                                id=pmc_id
                            )
                            ref_results = Entrez.read(ref_handle)
                            ref_handle.close()

                            if ref_results and len(ref_results[0]["LinkSetDb"]) > 0:
                                ref_pmids = [link["Id"] for link in ref_results[0]["LinkSetDb"][0]["Link"]]

                                # Get basic info for each reference
                                if ref_pmids:
                                    ref_handle = Entrez.esummary(db="pubmed", id=",".join(ref_pmids))
                                    ref_summaries = Entrez.read(ref_handle)
                                    ref_handle.close()

                                    for summary in ref_summaries:
                                        logger.info(f"summary: {summary}")
                                        ref_info = {
                                            "title": str(summary.get("Title", "")),
                                            "citation": f"{summary.get('Source', '')} {summary.get('PubDate', '')}",
                                            "pmid": str(summary.get("Id", ""))
                                        }
                                        references.append(ref_info)
                        except Exception as e:
                            logger.warning(f"Error getting references from PMC: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error extracting references: {str(e)}")
                # Combine fields of study from MeSH terms and keywords
                fields_of_study = ";".join(set(mesh_terms + keywords))

                # Create structured paper record
                if arxiv_id:
                    paper_id = arxiv_id
                else:
                    paper_id = pmid
                paper_info = {
                    "paper_id": paper_id,
                    "PMID": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": [{"name": author} for author in authors],
                    "author_str": "; ".join(authors),
                    "year": publication_date,
                    "journal": journal_info,
                    "fieldsOfStudy": fields_of_study,
                    "mesh_terms": mesh_terms,
                    "keywords": keywords,
                    "doi": identifiers.get("doi", ""),
                    "pmc": identifiers.get("pmc", ""),
                    "arxivId": arxiv_id,
                    "arxivUrl": arxiv_url,
                    "citationCount": citation_count,
                    "referenceCount": len(references),
                    "references": references,
                    "source": "Search From PubMed",
                }

                # Add external IDs
                if identifiers:
                    paper_info["external_ids"] = identifiers

                paper_list.append(paper_info)

            except Exception as e:
                logger.error(f"Error processing PubMed article {pmid}: {str(e)}")
                logger.error(traceback.format_exc())

        return paper_list

    except Exception as e:
        logger.error(f"Error fetching PubMed data: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def search_from_pubmed(query, max_results=10):
    """在 PubMed 上搜索论文"""
    # 1. 在 PubMed 上执行搜索
    handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            httppost={"proxies": PROXIES}  # 添加代理信息
    )
    record = Entrez.read(handle)
    handle.close()
    # 获取搜索到的论文 ID 列表（PMID）
    pmid_list = record["IdList"]
    # 获取格式化信息
    structured_papers = fetch_pubmed_json(pmid_list)
    return structured_papers


def fetch_reference_openalex(ref_id):
    """获取单个文献的详细信息"""
    try:
        ref_url = f"https://api.openalex.org/works/{ref_id}"
        ref_response = requests.get(ref_url)

        if ref_response.status_code != 200:
            logger.error(
                f"Reference Failed to retrieve reference data: {ref_response.status_code}"
            )
            return None

        ref_data = ref_response.json()
        is_open = ref_data.get("open_access", {}).get("is_oa", False)
        if not is_open:
            logger.error("Openalex Reference Not open access, skip....")
            return None

        ref_title = ref_data.get("title", "")
        ref_authors = [
            a["author"]["display_name"] for a in ref_data.get("authorships", [])
        ]

        ref_abstract_inverted_index = ref_data.get("abstract_inverted_index", None)
        ref_abstract = (
            " ".join(ref_abstract_inverted_index.keys())
            if ref_abstract_inverted_index
            else ""
        )

        primary_topic = ref_data.get("primary_topic", {}).get("display_name", "")
        field = (
            ref_data.get("primary_topic", {}).get("field", {}).get("display_name", "")
        )
        subfield = (
            ref_data.get("primary_topic", {})
            .get("subfield", {})
            .get("display_name", "")
        )
        domain = (
            ref_data.get("primary_topic", {}).get("domain", {}).get("display_name", "")
        )

        publication_year = ref_data.get("publication_year", "")

        fields_of_study = "; ".join(
            filter(None, [primary_topic, field, subfield, domain])
        )
        reference_count = len(ref_data.get("referenced_works", []))
        reference_works = ref_data.get("referenced_works", [])
        cited_by_count = ref_data.get("cited_by_count", 0)
        oa_url = ref_data.get("open_access", {}).get("oa_url", "")

        output = {
            "openalex_url": ref_data.get("id", ""),
            "oa_url": oa_url,
            "title": ref_title if ref_title is not None else "",
            "authors": ref_authors,
            "abstract": ref_abstract,
            "fieldsOfStudy": fields_of_study,
            "publicationYear": publication_year,
            "referenceWorks": reference_works,
            "referenceCount": reference_count,
            "cited_by_count": cited_by_count,
            "isOpen": is_open,
        }
        return output
    except:
        return None


def search_doc_via_url_parallel_openalex(referenced_works, max_workers=4, timeout=10):
    """并行获取多个文献信息"""
    references = []
    if len(referenced_works) > 100:
        referenced_works = random.sample(referenced_works, 20)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ref = {
            executor.submit(fetch_reference_openalex, ref_id): ref_id
            for ref_id in referenced_works
        }
        for future in as_completed(future_to_ref):
            ref_doc = future.result(timeout=timeout)
            if ref_doc is not None:  # 过滤掉 None 值
                references.append(ref_doc)
    return references


def search_doc_via_url_from_openalex(data):
    referenced_works = data
    references = []
    for ref_id in referenced_works:
        try:
            ref_url = f"https://api.openalex.org/works/{ref_id}"
            ref_response = requests.get(ref_url, timeout=10)
            if ref_response.status_code == 200:
                ref_data = ref_response.json()
                is_open = ref_data.get("open_access", {}).get("is_oa", False)
                if not is_open:
                    logger.info("Openalex Not open access, skip....")
                    continue
                ref_title = ref_data.get("title", "")
                ref_authors = [
                    a["author"]["display_name"] for a in ref_data.get("authorships", [])
                ]

                ref_abstract_inverted_index = ref_data.get(
                    "abstract_inverted_index", None
                )
                ref_abstract = (
                    " ".join(ref_abstract_inverted_index.keys())
                    if ref_abstract_inverted_index is not None
                    else ""
                )
                if ref_abstract == "" and ref_title == "":
                    logger.info("title and abstract is empty, skip")
                    continue

                primary_topic = ref_data.get("primary_topic", {}).get(
                    "display_name", ""
                )
                field = (
                    ref_data.get("primary_topic", {})
                    .get("field", {})
                    .get("display_name", "")
                )
                subfield = (
                    ref_data.get("primary_topic", {})
                    .get("subfield", {})
                    .get("display_name", "")
                )
                domain = (
                    ref_data.get("primary_topic", {})
                    .get("domain", {})
                    .get("display_name", "")
                )

                publication_year = ref_data.get("publication_year", "")

                fieldsOfStudy = ""
                for one in [primary_topic, field, subfield, domain]:
                    if one != "":
                        fieldsOfStudy += one + "; "

                reference_count = len(ref_data.get("referenced_works", []))
                reference_works = ref_data.get("referenced_works", [])
                cited_by_count = ref_data.get("cited_by_count", 0)

                # ---
                oa_url = ref_data.get("open_access", {}).get("oa_url", "")

                paper_id = ""
                if oa_url is not None and "arxiv" in oa_url:
                    arxivId = oa_url.split("/")[-1].split("v")[0]
                    arxivUrl = oa_url
                    paper_id = arxivId
                else:
                    arxivId = ""
                    arxivUrl = ""

                url = ref_data.get("id", "")
                if url:
                    openalex_id = url.split("/")[-1]
                else:
                    openalex_id = get_md5(keep_letters(ref_title))

                if paper_id == "":
                    paper_id = openalex_id

                formated_data = {
                    "paper_id": paper_id,
                    "url": url,
                    "openalex_id": openalex_id,
                    "arxivId": arxivId,
                    "arxivUrl": arxivUrl,
                    "title": ref_title,
                    "abstract": ref_abstract,
                    "authors": [{"name": author} for author in ref_authors],
                    "citationCount": (
                        cited_by_count if cited_by_count is not None else 0
                    ),
                    "fieldsOfStudy": fieldsOfStudy,
                    "referenceCount": reference_count,
                    "referenceWorksOpenAlex": reference_works,
                    "publicationYear": publication_year,
                    "isOpen": is_open,
                    "source": "Search From OpenAlex",
                }
                references.append(formated_data)
            else:
                logger.error(
                    f"\nFailed to retrieve reference data: {ref_response.status_code}"
                )
        except:
            logger.error(
                f"openalex get info from url is error: {traceback.format_exc()}"
            )
            pass
    return references


def search_paper_via_query_from_openalex(
    keyword, page=1, per_page=10, search_reference=False
):
    """从openalex获取相关论文，要输入关键词，不支持自然语言的输入，效果不好"""
    try:
        logger.info("search_paper_via_query_from_openalex")
        base_url = "https://api.openalex.org/works"
        current_year = datetime.now().year
        start_year = current_year - 20
        # print(f"start_year: {start_year}")
        params = {
            # "search": keyword,  # 全文搜索
            "sort": "cited_by_count:desc",  # 按照被引用的次数降序排序
            "filter": f"open_access.is_oa:true,publication_year:>{start_year},title_and_abstract.search:{keyword}",  # 过滤近20年的文章，只选择还在公开状态的文章,只搜索标题和摘要
            "page": page,
            "per-page": per_page,
        }
        for i in range(4):
            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                data = response.json()
                search_docs = {}
                for work in data["results"]:
                    try:

                        is_open = work.get("open_access", {}).get("is_oa", False)
                        if not is_open:
                            logger.info("Openalex Work Not open access, skip....")
                            continue

                        title = work.get("title", "")
                        if title is None or title == "":
                            logger.info("title is empty, skip....")
                            continue

                        authors = [
                            a["author"]["display_name"]
                            for a in work.get("authorships", [])
                        ]
                        # 处理摘要
                        abstract_inverted_index = work.get(
                            "abstract_inverted_index", None
                        )
                        abstract = (
                            " ".join(abstract_inverted_index.keys())
                            if abstract_inverted_index is not None
                            else ""
                        )

                        if abstract is None or abstract == "":
                            logger.info("abstract is None, skip....")
                            continue

                        cited_by_count = work.get("cited_by_count", 0)

                        if work["primary_topic"] is None:
                            work["primary_topic"] = {}

                        primary_topic = work.get("primary_topic", {}).get(
                            "display_name", ""
                        )
                        field = (
                            work.get("primary_topic", {})
                            .get("field", {})
                            .get("display_name", "")
                        )
                        subfield = (
                            work.get("primary_topic", {})
                            .get("subfield", {})
                            .get("display_name", "")
                        )
                        domain = (
                            work.get("primary_topic", {})
                            .get("domain", {})
                            .get("display_name", "")
                        )

                        fieldsOfStudy = ""
                        for one in [primary_topic, field, subfield, domain]:
                            if one != "":
                                fieldsOfStudy += one + "; "

                        reference_count = len(work.get("referenced_works", []))
                        reference_works = work.get("referenced_works", [])
                        publication_year = work.get("publication_year", "")

                        oa_url = work.get("open_access", {}).get("oa_url", "")

                        paper_id = ""
                        if oa_url is not None and "arxiv" in oa_url:
                            arxivId = oa_url.split("/")[-1].split("v")[0]
                            arxivUrl = oa_url
                            paper_id = arxivId
                        else:
                            arxivId = ""
                            arxivUrl = ""

                        url = work.get("id", "")
                        if url:
                            openalex_id = url.split("/")[-1]

                        else:
                            openalex_id = get_md5(keep_letters(title))

                        if paper_id == "":
                            paper_id = openalex_id

                        formated_data = {
                            "paper_id": paper_id,
                            "url": url,
                            "openalex_id": openalex_id,
                            "arxivId": arxivId,
                            "arxivUrl": arxivUrl,
                            "title": title,
                            "abstract": abstract,
                            "authors": [{"name": author} for author in authors],
                            "citationCount": (
                                cited_by_count if cited_by_count is not None else 0
                            ),
                            "fieldsOfStudy": fieldsOfStudy,
                            "referenceCount": reference_count,
                            "referenceWorksOpenAlex": reference_works,
                            "publicationYear": publication_year,
                            "isOpen": is_open,
                            "source": "Search From OpenAlex",
                        }
                        # print(formated_data)
                        # print(f"Title: {title}")
                        # print(f"Authors: {authors}")
                        # print(f"Abstract: {abstract}")
                        # print(f"Cited by: {cited_by_count}")
                        # print(f"Concepts: {concepts}")
                        # print(f"References Count: {reference_count}")
                        if search_reference:
                            formated_data["reference"] = (
                                search_doc_via_url_parallel_openalex(reference_works)
                            )
                        search_docs[paper_id] = formated_data
                    except:
                        logger.error(f"Openalex parse error: {traceback.format_exc()}")
                return search_docs

        logger.info(f"Failed to retrieve data: {response.status_code}, {response}")
        return {}
    except:
        logger.error(f"Failed to retrieve data from OpenAlex: {traceback.format_exc()}")
        return {}


def search_paper_via_query_from_semantic(query, max_paper_num=15, end_date=None):
    """
    从semantic sholar获取相关关query，失败率较高，query是对应的关键词，自然语言查询表现不好
    find schema info here:
    https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_relevance_search
    citationCount: 引用本文的论文数量
    referenceCount: 本文引用的论文的数量
    """
    logger.info("run search_paper_via_query_from_semantic...")
    if "Search queries: " in query:
        query = query.split("Search queries: ")[1]

    current_year = datetime.now().year
    start_year = current_year - 20

    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y%m%d")
        end_date = end_date.strftime("%Y%m%d")

    else:
        end_date = current_year

    # print(f"{start_year}:{end_date}")

    query_params = {
        "query": query,
        "limit": max_paper_num,
        "minCitationCount": 0,
        "sort": "citationCount:desc",
        "publicationDateOrYear": f"{start_year}0101:{end_date}",
        "fields": "title,year,abstract,authors.name,url,externalIds,fieldsOfStudy,openAccessPdf,citationCount,referenceCount,references.title,references.abstract,references.authors,references.year,references.fieldsOfStudy,references.citationCount,references.openAccessPdf,references.externalIds",
    }
    from global_config import S2_API_KEY

    api_key = S2_API_KEY
    # Define headers with API key
    headers = {"x-api-key": api_key}
    # Send the API request
    for _ in range(4):
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=query_params,
            timeout=10,
        )
        # headers=headers)
        time.sleep(0.1)

        if response.status_code == 200:
            response_data = response.json()
        else:
            response_data = None
            logger.info(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
        if (
            response_data is None
            or len(response_data) == 0
            or "data" not in response_data
        ):
            logger.info("Semantic retrieval failed")
        else:
            docs = response_data["data"]
            papers_info = {}
            for paper in docs:
                try:
                    paper["source"] = "Search From SemanticScholar"
                    if paper["fieldsOfStudy"] is None:
                        paper["fieldsOfStudy"] = ""
                    else:
                        paper["fieldsOfStudy"] = ";".join(
                            one for one in paper["fieldsOfStudy"]
                        )

                    paper["publicationYear"] = paper["year"]
                    if "externalIds" in paper and "ArXiv" in paper["externalIds"]:
                        paper_id = (
                            paper["externalIds"]["ArXiv"].split("/")[-1].split("v")[0]
                        )
                        paper["arxivId"] = paper_id
                        paper["arxivUrl"] = f"https://arxiv.org/abs/{paper_id}"
                    else:
                        paper_id = paper["paperId"]
                        paper["arxivId"] = ""
                        paper["arxivUrl"] = ""
                    paper["paper_id"] = paper_id
                    papers_info[paper_id] = paper
                except:
                    logger.error(f"semantic parse error: {traceback.format_exc()}")

            return papers_info
    return {}


def google_search_arxiv_id(query, try_num=4, num=10, end_date=""):
    """从google搜索arxiv id, 要是用api，免费额度2500"""
    # refer from: https://serper.dev/playground
    url = "https://google.serper.dev/search"
    search_query = f"{query} site:arxiv.org"
    # logger.info(f"end_date: {end_date}")
    if end_date != "":
        try:
            end_date = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"

    payload = json.dumps(
        {
            "q": search_query,
            "num": num,
            # "autocorrect": True,
            "page": 1,
            # "type":"search"
        }
    )

    GOOGLE_SERPER_KEY = os.getenv("GOOGLE_SERPER_KEY", "xxx")
    logger.info(f"use GOOGLE_SERPER_KEY: {GOOGLE_SERPER_KEY}")
    headers = {"X-API-KEY": GOOGLE_SERPER_KEY, "Content-Type": "application/json"}
    assert headers["X-API-KEY"] != "your google keys", "add your google search key!!!"

    for _ in range(try_num):
        try:
            response = requests.request(
                "POST", url, headers=headers, data=payload, timeout=10, proxies=PROXIES
            )
            if response.status_code == 200:
                results = json.loads(response.text)
                logger.info(f"results: {results}")
                arxiv_id_list = []
                for paper in results["organic"]:
                    link = paper["link"]
                    match = re.search("arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", link)
                    if match:
                        arxiv_id = match.group(1)
                        arxiv_id_list.append(arxiv_id)
                res = list(set(arxiv_id_list))
                logger.info(f"google_search_arxiv_id success: {len(res)}")
                return res
            else:
                logger.error(f"google_search_arxiv_id response: {response}")
        except:
            logger.error(
                f"google search failed, query: {query}; Error: {traceback.format_exc()}"
            )
            continue
    return []


def get_doc_info_from_semantic_scholar_by_arxivid(
    arxiv_id: str, try_num: int = 2, timeout: float = 3, raise_on_error: bool = False
) -> Optional[Dict[str, Any]]:
    """
    根据 arXiv ID 从 Semantic Scholar 获取论文信息，信息非常丰富
    Args:
        arxiv_id (str): arXiv 论文 ID，例如 "1706.03762"。
        try_num (int): 最大重试次数，默认 5。
        timeout (float): 单次请求超时时间（秒），默认 20.0。
        raise_on_error (bool): 是否在失败时抛出异常而非警告，默认 False。
    Returns:
        Optional[Dict[str, Any]]: 论文信息的字典，失败时返回 None。
    Raises:
        requests.RequestException: 如果 raise_on_error=True，且所有尝试失败。
    """
    # example: https://api.semanticscholar.org/v1/paper/arXiv:2503.05565

    url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"

    for attempt in range(try_num):
        try:
            response = requests.get(url, timeout=timeout, proxies=PROXIES)
            if response.status_code == 200:
                data = response.json()
                # 检查返回的 arXiv ID 是否匹配
                if data.get("arxivId") != arxiv_id:
                    msg = f"Returned arXiv ID '{data.get('arxivId')}' does not match requested ID: {arxiv_id}"
                    if raise_on_error:
                        raise ValueError(msg)
                    logger.error(msg)
                    return None
                # 过滤有效的参考文献（必须有 arXiv ID）
                if len(data["references"]) == 0:
                    data["references"] = []
                references_valid_ids = set(
                    [
                        ref["arxivId"].split("/")[-1].split("v")[0]
                        for ref in data.get("references", [])
                        if ref.get("arxivId") is not None
                    ]
                )
                logger.info(
                    f"references_valid_ids {arxiv_id}: {len(references_valid_ids)}"
                )

                if len(references_valid_ids) == 0:
                    data["references"] = []
                else:
                    # semanctic中的reference不含abstract信息
                    references_id2doc = parallel_search_search_paper_from_arxiv(
                        list(references_valid_ids), max_workers=2
                    )
                    logger.info(
                        f"references_id2doc {arxiv_id}: {list(references_id2doc.keys())}"
                    )

                    references_valid = [
                        references_id2doc[doc_id]
                        for doc_id in list(references_valid_ids)
                        if doc_id in references_id2doc
                    ]
                    # 保留原始参考文献
                    # data["references_raw"] = data.get("references", [])
                    data["references"] = references_valid

                data["citationCount"] = len(data["citations"])

                if data["authors"] is None:
                    data["authors"] = []

                data["authors"] = data["authors"][:10]

                if data["title"] is None:
                    data["title"] = ""
                if data["abstract"] is None:
                    data["abstract"] = ""

                if data["fieldsOfStudy"] is None:
                    data["fieldsOfStudy"] = []
                data["fieldsOfStudy"] = ";".join(
                    one for one in data["fieldsOfStudy"][:5]
                )

                if data["venue"] is None:
                    data["venue"] = ""

                if data["year"] is None:
                    data["publicationYear"] = ""
                else:
                    data["publicationYear"] = data.pop("year")
                # del data["citations"]
                # del data["references_raw"]
                # del data["authors"]
                # del data["fieldsOfStudy"]
                data["source"] = "Search From SemanticScholar"

                return data
            elif response.status_code == 404:
                msg = f"Attempt {attempt+1}: Paper not found for arXiv ID: {arxiv_id} (Status 404)"
                if raise_on_error:
                    raise requests.RequestException(msg)
                logger.error(msg)
            elif response.status_code == 429:
                msg = f"Attempt {attempt+1}: Rate limit exceeded (Status 429) - {response.text}"
                if raise_on_error:
                    raise requests.RequestException(msg)
                logger.error(msg)
            else:
                msg = f"Attempt {attempt+1}: API request failed (Status {response.status_code}) - {response.text}"
                if raise_on_error:
                    raise requests.RequestException(msg)
                logger.error(msg)

        except requests.exceptions.RequestException as e:
            msg = f"Attempt {attempt+1}: Request exception - {str(e)}"
            if raise_on_error and attempt == try_num - 1:
                raise requests.RequestException(
                    f"All {try_num} attempts failed: {str(e)}"
                )
            logger.error(msg)

    final_msg = f"All {try_num} attempts failed for arXiv ID: {arxiv_id}"
    if raise_on_error:
        raise requests.RequestException(final_msg)
    logger.error(final_msg)
    return None


def get_openalex_info(arxiv_id):
    # 没跑通
    url = f"https://api.openalex.org/works/https://arxiv.org/abs/{arxiv_id}"
    response = requests.get(url, proxies=PROXIES)
    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title"),
            "cited_by_count": data.get("cited_by_count"),
            "referenced_works": data.get("referenced_works"),
            "citing_works": data.get("citing_works"),
        }
    else:
        return {
            "error": f"OpenAlex API request failed with status {response.status_code}"
        }


def search_paper_from_arxiv_by_arxiv_id(arxiv_id, try_num=5):
    """
    Search paper by arxiv id from Arxiv.
    :param arxiv_id: arxiv id of the paper
    :return: paper list
    """
    logger.info("search_paper_from_arxiv_by_arxiv_id ... ")
    search = arxiv.Search(
        query="",
        id_list=[arxiv_id],
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = None
    for _ in range(try_num):
        try:
            results = list(ARXIV_CLIENT.results(search, offset=0))
        except:
            logger.error(f"Failed to search arxiv id: {arxiv_id}")

    if results is None:
        return None
    res = None
    for arxiv_id_result in results:
        entry_id = arxiv_id_result.entry_id.split("/")[-1]
        entry_id = entry_id.split("v")[0]
        if entry_id == arxiv_id:
            res = {
                "paper_id": arxiv_id,
                "arxivId": arxiv_id,
                "arxivUrl": arxiv_id_result.entry_id,
                "title": arxiv_id_result.title.replace("\n", " "),
                "abstract": arxiv_id_result.summary.replace("\n", " "),
                "authors": [
                    {"name": author.name} for author in arxiv_id_result.authors
                ],
                "year": arxiv_id_result.published.strftime("%Y%m%d"),
                "fieldsOfStudy": arxiv_id_result.categories,
                "source": "Search From Arxiv",
            }
            break
    return res


def search_paper_from_arxiv_by_arxiv_id_bsz(arxiv_ids, max_retries=3):
    """
    批量查询多个 Arxiv ID，加入重试机制
    :param arxiv_ids: 论文 ID 列表
    :param max_retries: 最大重试次数
    :return: Dict[str, Dict] -> {arxivId: paper_info}
    """
    for attempt in range(max_retries):
        try:
            search = arxiv.Search(
                id_list=arxiv_ids,
                max_results=100,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )
            try:
                results = list(ARXIV_CLIENT.results(search))
                for result in results:
                    logger.info(f"Arxiv success paper: {result.title}")
            except arxiv.HTTPError as e:
                logger.error(f"Arxiv HTTP Error: {e}")

            # results = list(ARXIV_CLIENT.results(search))
            logger.info(
                f"search_paper_from_arxiv_by_arxiv_id_bsz results num : {len(results)}, id num: {len(arxiv_ids)}"
            )
            return {
                paper.entry_id.split("/")[-1].split("v")[0]: {
                    "paper_id": paper.entry_id.split("/")[-1].split("v")[0],
                    "arxivId": paper.entry_id.split("/")[-1].split("v")[0],
                    "arxivUrl": paper.entry_id,
                    "title": paper.title.replace("\n", " "),
                    "abstract": paper.summary.replace("\n", " "),
                    "authors": [{"name": author.name} for author in paper.authors],
                    "publicationYear": paper.published.strftime("%Y%m%d"),
                    "fieldsOfStudy": ";".join(one for one in paper.categories),
                    "source": "Search From Arxiv",
                }
                for paper in results
            }

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(
                f"Attempt {attempt + 1} failed for IDs {arxiv_ids}. Error: {e}"
            )
            time.sleep(2**attempt)  # 指数退避策略，避免短时间内频繁请求

    # 如果所有尝试都失败，则返回空字典
    return {}


def parallel_search_search_paper_from_arxiv(
    arxiv_id_list, batch_size=10, max_workers=5
):
    """
    使用线程池并行查询 Arxiv 论文
    :param arxiv_id_list: 论文 ID 列表
    :param batch_size: 每个批次查询多少个 ID
    :param max_workers: 并行线程数
    :return: 查询结果
    """

    results = {}
    failed_ids = []  # 记录失败的 ID

    arxiv_id_list_to_process = []

    # from local_db import LOCAL_DB, local_db_file
    # for _id in arxiv_id_list:
    #     if _id in LOCAL_DB:
    #         results[_id] = LOCAL_DB[_id]
    #         results[_id]["source"] = "Search From Local"
    #     else:
    #         arxiv_id_list_to_process.append(_id)

    with ArxivDatabase(db_path) as db:
        for _id in arxiv_id_list:
            try:
                db_info = db.get(_id)
                if db_info is None:
                    arxiv_id_list_to_process.append(_id)
                else:
                    results[_id] = db_info
                    results[_id]["source"] = "Search From Local"
            except:
                arxiv_id_list_to_process.append(_id)
                pass


    logger.info(f"Local db exist doc num: {len(results)}")
    logger.info(
        f"Now Request arxiv for {len(arxiv_id_list_to_process)} docs, batch_size is:{batch_size}, max_workers is: {max_workers}"
    )
    if not arxiv_id_list_to_process:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                search_paper_from_arxiv_by_arxiv_id_bsz,
                arxiv_id_list_to_process[i : i + batch_size],
            ): arxiv_id_list_to_process[i : i + batch_size]
            for i in range(0, len(arxiv_id_list_to_process), batch_size)
        }

        for future in as_completed(future_to_batch):
            batch_ids = future_to_batch[future]
            try:
                batch_results = future.result()
                if batch_results:
                    results.update(batch_results)
                else:
                    failed_ids.extend(batch_ids)  # 记录失败的 ID
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"Unexpected error for batch {batch_ids}: {e}")
                failed_ids.extend(batch_ids)

    # if results:
    #     logger.info("save to local db")
    #     with ArxivDatabase(db_path) as db:
    #         for _id, info in results.items():
    #             db.update_or_insert(_id,info)

    save = False
    if save:
        with open(local_db_file, "a") as fw:
            for _id, info in results.items():
                fw.write(json.dumps(info) + "\n")

    if failed_ids:
        logger.error(f"Failed to retrieve data for IDs: {failed_ids}")

    return results


def get_doc_info_from_api(query, try_num=5, end_date=""):
    """
    输入query，调用google搜索，获取arxiv_id，再从arxiv获取文章信息
    """
    try:
        arxiv_lst = google_search_arxiv_id(
            query=query, try_num=try_num, end_date=end_date
        )
        logger.info(f"arxiv_lst: {len(arxiv_lst)}")

        doc_info = parallel_search_search_paper_from_arxiv(arxiv_lst)
        return doc_info

    except:
        logger.error(f"gen foc info from api error: {traceback.format_exc()}")
        return []


def get_doc_info_from_api(query, try_num=5, end_date=""):
    """
    输入query，调用google搜索，获取arxiv_id，再从arxiv获取文章信息
    """
    try:
        arxiv_lst = google_search_arxiv_id(
            query=query, try_num=try_num, end_date=end_date
        )
        logger.info(f"arxiv_lst: {len(arxiv_lst)}")

        doc_info = parallel_search_search_paper_from_arxiv(arxiv_lst)
        return doc_info

    except:
        logger.error(f"gen foc info from api error: {traceback.format_exc()}")
        return []


### 测试google + arxiv
# from pprint import pprint

# user_query = """current research on lora"""
# # info = {"query": user_query}
# doc_info_lst = get_doc_info_from_api(user_query)
# pprint(doc_info_lst)

# print(search_paper_via_query_from_semantic("Lora"))
# res = search_paper_via_query_from_openalex("Lora")
# print([one["title"] for one in res.values()])

# print(search_from_pubmed("synthetic data for supervised fine-tuning"))
# pubmed_search = search_from_pubmed("lung caner")
# print(len(pubmed_search))
# print(pubmed_search[0])


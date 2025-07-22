# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import os
import json
import sqlite3
from typing import Dict, Any, List, Optional, Iterator
from contextlib import contextmanager
from tqdm import tqdm

from log import logger


class ArxivDatabase:
    """A SQLite database manager for arXiv papers.

    Provides functionality to store and retrieve arXiv paper metadata using SQLite.
    Supports batch operations and context management.

    Attributes:
        db_path (str): Path to the SQLite database file
        batch_size (int): Number of records to process in a single transaction
    """

    def __init__(self, db_path: str, batch_size: int = 1000):
        """Initialize database connection and create table if needed.

        Args:
            db_path (str): Path to SQLite database file
            batch_size (int): Size of batches for bulk operations
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self._connection = None
        self._cursor = None
        self.connect()
        self._create_table()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection, reconnecting if needed."""
        if self._connection is None:
            self.connect()
        return self._connection

    @property
    def cursor(self) -> sqlite3.Cursor:
        """Get database cursor, reconnecting if needed."""
        if self._cursor is None:
            self.connect()
        return self._cursor

    def connect(self):
        """Establish database connection with optimized settings."""
        self._connection = sqlite3.connect(self.db_path)
        self._connection.execute(
            "PRAGMA journal_mode=WAL")  # Optimize write performance
        self._connection.execute(
            "PRAGMA synchronous=NORMAL")  # Balance durability and speed
        self._cursor = self._connection.cursor()

    def _create_table(self):
        """Create the arXiv documents table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS arxiv_docs (
                arxiv_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Transaction failed: {str(e)}")
            raise

    def load_from_jsonl(self, jsonl_file: str):
        """Load data from JSONL file in batches.

        Args:
            jsonl_file (str): Path to JSONL file containing arXiv records
        """
        if not os.path.exists(jsonl_file):
            logger.warning(f"JSONL file {jsonl_file} does not exist.")
            return

        batch = []
        total_processed = 0

        def process_batch(records: List[tuple]):
            if not records:
                return
            self.cursor.executemany(
                "INSERT OR REPLACE INTO arxiv_docs (arxiv_id, data) VALUES (?, ?)",
                records,
            )
            self.conn.commit()

        try:
            with open(jsonl_file, "r") as fr:
                for line in tqdm(fr, desc="Loading records"):
                    try:
                        info = json.loads(line)
                        arxiv_id = info.get("arxivId")
                        if arxiv_id:
                            batch.append((arxiv_id, json.dumps(info)))

                        if len(batch) >= self.batch_size:
                            process_batch(batch)
                            total_processed += len(batch)
                            batch = []
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON line: {line[:100]}...")
                        continue

            # Process remaining records
            if batch:
                process_batch(batch)
                total_processed += len(batch)

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        finally:
            logger.info(f"Loaded {total_processed} records from {jsonl_file}")

    def update_or_insert(self, arxiv_id: str, data: Dict[str, Any]):
        """Update or insert a single record."""
        try:
            data_json = json.dumps(data)
            with self.transaction():
                self.cursor.execute(
                    "INSERT OR REPLACE INTO arxiv_docs (arxiv_id, data) VALUES (?, ?)",
                    (arxiv_id, data_json),
                )
        except Exception as e:
            logger.error(f"Error updating record {arxiv_id}: {str(e)}")
            raise

    def get(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get a single record by arXiv ID."""
        try:
            self.cursor.execute(
                "SELECT data FROM arxiv_docs WHERE arxiv_id = ?", (arxiv_id, ))
            result = self.cursor.fetchone()
            return json.loads(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error fetching record {arxiv_id}: {str(e)}")
            return None

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all records using generator to manage memory."""
        try:
            self.cursor.execute("SELECT arxiv_id, data FROM arxiv_docs")
            return {
                row[0]: json.loads(row[1])
                for row in self.cursor.fetchall()
            }
        except Exception as e:
            logger.error(f"Error fetching all records: {str(e)}")
            return {}

    def iter_records(self, batch_size: int = 1000) -> Iterator[Dict[str, Any]]:
        """Iterate over records in batches to manage memory."""
        offset = 0
        while True:
            self.cursor.execute(
                "SELECT arxiv_id, data FROM arxiv_docs LIMIT ? OFFSET ?",
                (batch_size, offset),
            )
            batch = self.cursor.fetchall()
            if not batch:
                break
            for row in batch:
                yield {"arxiv_id": row[0], "data": json.loads(row[1])}
            offset += batch_size

    def get_record_count(self) -> int:
        """Get total number of records."""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM arxiv_docs")
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting records: {str(e)}")
            return 0

    def close(self):
        """Close database connection safely."""
        if self._connection:
            try:
                self._connection.commit()
                self._cursor.close()
                self._connection.close()
            except Exception as e:
                logger.error(f"Error closing database: {str(e)}")
            finally:
                self._connection = None
                self._cursor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


# db_path = "./database/arxiv_data.db"
db_path = "./database/recovered.db"

# with ArxivDatabase(db_path) as db:
#     all_records = db.get_all()
#     print(f"Total Records: {len(all_records)}")

# 示例用法
# if __name__ == "__main__":
#     # 初始化数据库
#     db_path = "/share/project/shixiaofeng/code/scholar-paper-agent-retrieval/database/arxiv_data.db"
#     jsonl_file = "/share/project/shixiaofeng/code/scholar-paper-agent-retrieval/database/id2docs.jsonl"

#     db = ArxivDatabase(db_path)

#     # 从现有 JSONL 文件加载数据
#     db.load_from_jsonl(jsonl_file)

#     # 示例：添加或更新一条记录
#     sample_data = {
#         "arxivId": "000",
#         "title": "Citation Count Analysis for Papers with Preprints",
#         "authors": ["Author1", "Author2"]
#     }

#     db.update_or_insert(sample_data["arxivId"], sample_data)

#     # 示例：获取一条记录
#     record = db.get("2501.07810")
#     print("Sample Record:", record)

#     # 示例：获取所有记录
#     all_records = db.get_all()
#     print(f"Total Records: {len(all_records)}")
#     # pprint(all_records)

#     # 关闭数据库
#     db.close()

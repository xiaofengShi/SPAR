import tempfile
import unittest
from pathlib import Path

from local_db_v2 import ArxivDatabase


class ArxivDatabaseInitializationTests(unittest.TestCase):
    def test_initializes_database_when_parent_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_file = Path(temp_dir) / "database" / "arxiv_data.db"

            with ArxivDatabase(str(db_file)) as database:
                self.assertEqual(database.get_record_count(), 0)

            self.assertTrue(db_file.is_file())


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from tools import init_db


class InitDbTests(unittest.TestCase):
    def test_get_dsn_appends_database_name(self) -> None:
        self.assertEqual(
            init_db.get_dsn("financial_rag"),
            "postgresql://postgres:postgres@localhost:5432/financial_rag",
        )

    def test_get_dsn_keeps_existing_database_name(self) -> None:
        self.assertTrue(
            init_db.get_dsn("postgres").startswith(
                "postgresql://postgres:postgres@localhost:5432/"
            )
        )

    @patch("tools.init_db.psycopg2.connect")
    def test_ensure_database_creates_missing_database(
        self, connect_mock
    ) -> None:
        connect_mock.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = (
            None
        )

        init_db.ensure_database("financial_rag")

        connect_mock.assert_called()

    @patch("tools.init_db.psycopg2.connect")
    def test_create_queue_table_runs_schema(self, connect_mock) -> None:
        connect_mock.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = (
            None
        )

        init_db.create_queue_table("financial_rag")

        connect_mock.assert_called()


if __name__ == "__main__":
    unittest.main()

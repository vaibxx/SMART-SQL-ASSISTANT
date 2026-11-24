"""Exercise the sidebar, chat input and displayed answer using the CI PostgreSQL service."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psycopg2
from streamlit.testing.v1 import AppTest


class SalesQuestionTest(unittest.TestCase):
    def test_question_to_displayed_result(self):
        password = os.environ["UI_TEST_DB_PASSWORD"]
        with psycopg2.connect(host="localhost", dbname="ui_test", user="postgres", password=password) as db:
            with db.cursor() as cursor:
                cursor.execute("CREATE TABLE sales (product TEXT, units INTEGER)")
                cursor.executemany("INSERT INTO sales VALUES (%s, %s)", [("Atlas", 8), ("Orbit", 12), ("Atlas", 7)])

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "mainapp.py"), default_timeout=60)
        app.secrets["GEMINI_API_KEY"] = "offline-test-placeholder"
        sql = "SELECT product, SUM(units) AS units FROM sales GROUP BY product ORDER BY units DESC, product LIMIT 1"
        with patch("google.generativeai.GenerativeModel.generate_content", return_value=SimpleNamespace(text="SQL: " + sql)) as model:
            app.run()
            self.assertFalse(list(app.exception))
            app.radio[0].set_value("PostgreSQL").run()
            settings = {"Username": "postgres", "Password": password, "Database Name": "ui_test"}
            for field in app.text_input:
                if field.label in settings:
                    field.set_value(settings[field.label])
            next(button for button in app.button if button.label == "Connect to Database").click().run()
            self.assertEqual(app.session_state["connection_status"], "connected")
            question = "Which product sold the most units?"
            app.chat_input[0].set_value(question).run()
            self.assertFalse(list(app.exception))
            model.assert_called_once()
            self.assertIn(question, model.call_args.args[0])
            self.assertIn("Table: sales", model.call_args.args[0])
            self.assertEqual(app.dataframe[0].value.to_dict("records"), [{"product": "Atlas", "units": 15}])


if __name__ == "__main__":
    unittest.main()

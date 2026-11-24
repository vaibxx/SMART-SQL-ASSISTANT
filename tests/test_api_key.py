"""A visitor can paste their own Gemini key: no secrets file needed, and it can be forgotten."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "mainapp.py")


class PastedKeyTest(unittest.TestCase):
    def test_paste_and_forget_key_without_secrets(self):
        app = AppTest.from_file(APP, default_timeout=60)
        app.run()
        self.assertFalse(list(app.exception))
        self.assertFalse(app.session_state["gemini_configured"])

        key_box = next(t for t in app.text_input if t.label == "Gemini API key")
        key_box.set_value("visitor-test-key").run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.session_state["user_api_key"], "visitor-test-key")
        self.assertTrue(app.session_state["gemini_configured"])

        next(b for b in app.button if b.label == "Forget my key").click().run()
        self.assertEqual(app.session_state["user_api_key"], "")


if __name__ == "__main__":
    unittest.main()

# SMART-SQL-ASSISTANT

[![SQL assistant UI test](https://github.com/vaibxx/SMART-SQL-ASSISTANT/actions/workflows/sales-ui.yml/badge.svg)](https://github.com/vaibxx/SMART-SQL-ASSISTANT/actions/workflows/sales-ui.yml)

The automated UI test connects to PostgreSQL through the Streamlit sidebar, submits a sales question in chat, and checks the displayed answer. It uses a fixed Gemini response so contributors can run the test without a paid API key.

### ▶ [Use it in your browser](https://vaibxx.github.io/SMART-SQL-ASSISTANT/)

Paste your own Gemini API key, ask questions in plain English, and get SQL, tables and charts. A real PostgreSQL engine ([PGlite](https://github.com/electric-sql/pglite)) runs inside the page with a sample store database, or upload your own CSV. Your key stays in the tab's memory, goes only to Google's Gemini API, and is never stored. Writes are dry-run in a rolled-back transaction first, so you see exactly how many rows will change before confirming. The web app lives in [`docs/`](docs/).

## Two ways to use it

| | 🌐 Browser app | 🔌 Full app (Python) |
|---|---|---|
| Where | [vaibxx.github.io/SMART-SQL-ASSISTANT](https://vaibxx.github.io/SMART-SQL-ASSISTANT/) | Your machine, `streamlit run mainapp.py` |
| Setup | None | About 2 minutes |
| Data | Built-in sample store, or upload a CSV | **Your own MySQL or PostgreSQL** (host, port, username, password, database) or a JSON schema |
| Gemini key | Paste in the page, held in tab memory only | Paste in the sidebar, held in the session only |
| Writes | Dry-run preview, then confirm | Preview, confirm, then commit or roll back |

### 🔌 Connect your own MySQL or PostgreSQL

Browsers can't open database connections, so connecting a real database uses the full Python app.

```sh
git clone https://github.com/vaibxx/SMART-SQL-ASSISTANT && cd SMART-SQL-ASSISTANT
pip install -r requirements.txt
streamlit run mainapp.py
```

Then, in the sidebar:

1. Paste your Gemini API key ([get a free one](https://aistudio.google.com/apikey)). It stays in your session only and is never saved.
2. Pick **MySQL** or **PostgreSQL** and enter **Host, Port, Username, Password and Database name**.
3. Click **Connect to Database** and ask questions in the chat.

Any database you can reach from your machine works: `localhost`, or hosted ones such as Neon, Supabase, Railway or AWS RDS. Use a read-only database user if you only want to explore.

[Run Grain Check locally](grain-check/README.md)

A Streamlit application that enables natural language interactions with your databases using Google's Gemini AI. Transform plain English into SQL queries, visualize data, and manage your database through an intuitive chat interface.

✨ Features
🗣️ Natural Language to SQL

    AI-Powered Queries: Convert plain English to SQL using Google Gemini
    Smart Understanding: Handles complex queries with joins, aggregations, and filters
    Schema-Aware: Automatically understands your database structure

📊 Data Visualization

    Auto-Chart Generation: Creates bar charts, line graphs, pie charts, and more
    Smart Analysis: Suggests appropriate visualizations based on your data
    Interactive Plots: Powered by Plotly for rich, interactive charts

🔒 Safety & Control

    Safe Operations: Confirmation for DELETE operations with data preview
    Input Validation: Protection against SQL injection and harmful queries
    Transaction Management: Commit/Rollback controls for data integrity

🛠️ Flexible Database Support

    Multiple Databases: MySQL and PostgreSQL support
    Schema-Only Mode: Work with JSON schemas without live connections
    Dynamic Schema Loading: Automatic schema discovery from connected databases

🚀 Quick Start
Prerequisites

    Python 3.8+
    Google Gemini API key
    MySQL or PostgreSQL database (optional)

Configuration

    Get Google Gemini API Key:
        Visit Google AI Studio
        Create an API key for Gemini
        Enter it in the sidebar of the application

    Database Connection:
        Choose MySQL or PostgreSQL
        Enter connection details in the sidebar
        Or upload a JSON schema file


Supported Operations
Operation	SQL Generated	Safety Features
SELECT	✅	Auto-execution with results
INSERT	✅	Smart missing field detection
UPDATE	✅	Preview before execution
DELETE	✅	Confirmation with data preview
JOINs	✅	Schema-aware relationship handling
Aggregations	✅	Group by, count, sum, avg, etc.

🛡️ Safety Features

    SQL Injection Prevention: Input sanitization and validation
    Dangerous Operation Blocking: Prevents DROP, TRUNCATE, etc.
    Delete Confirmation: Always shows data preview before deletion
    Transaction Control: Manual commit/rollback for data changes
    Connection Validation: Regular connection health checks

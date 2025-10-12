# SMART-SQL-ASSISTANT
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

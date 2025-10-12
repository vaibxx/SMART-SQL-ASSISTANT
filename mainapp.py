import streamlit as st
import pandas as pd
import mysql.connector
import psycopg2
import json
import google.generativeai as genai
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Dict, List, Optional, Tuple
import datetime
import io
import os

class DatabaseManager:
    
    def __init__(self):
        self.connection = None
        self.engine = None
        self.db_type = None
        self.is_connected = False
        
    def connect_mysql(self, host: str, port: str, username: str, password: str, database: str) -> bool:
        try:
            port = int(port) if port else 3306
            connection_string = f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}'
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            
            # Test the connection
            self.connection.execute(text("SELECT 1"))
            
            self.db_type = 'mysql'
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"MySQL Connection Error: {str(e)}")
            self.is_connected = False
            return False
    
    def connect_postgres(self, host: str, port: str, username: str, password: str, database: str) -> bool:
        try:
            port = int(port) if port else 5432
            connection_string = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            
            # Test the connection
            self.connection.execute(text("SELECT 1"))
            
            self.db_type = 'postgres'
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"PostgreSQL Connection Error: {str(e)}")
            self.is_connected = False
            return False

    def check_connection(self) -> bool:
        if not self.is_connected or not self.connection:
            return False
        
        try:
            # Test connection with a simple query
            self.connection.execute(text("SELECT 1"))
            return True
        except Exception:
            self.is_connected = False
            return False

    def load_schema_from_db(self) -> Dict:
        if not self.connection:
            return {}
        
        schema_info = {}
        inspector = inspect(self.engine)
        
        try:
            tables = inspector.get_table_names()
            
            for table in tables:
                schema_info[table] = {
                    'columns': [],
                    'primary_keys': [],
                    'foreign_keys': []
                }
                
                # Get columns
                columns = inspector.get_columns(table)
                for col in columns:
                    col_info = {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable'],
                        'default': col.get('default'),
                        'autoincrement': col.get('autoincrement', False)
                    }
                    schema_info[table]['columns'].append(col_info)
                
                # Get primary keys
                primary_keys = inspector.get_pk_constraint(table)
                if primary_keys and 'constrained_columns' in primary_keys:
                    schema_info[table]['primary_keys'] = primary_keys['constrained_columns']
                
                # Get foreign keys
                foreign_keys = inspector.get_foreign_keys(table)
                for fk in foreign_keys:
                    fk_info = {
                        'constrained_columns': fk['constrained_columns'],
                        'referred_table': fk['referred_table'],
                        'referred_columns': fk['referred_columns']
                    }
                    schema_info[table]['foreign_keys'].append(fk_info)
                    
        except Exception as e:
            st.error(f"Error loading schema: {str(e)}")
            
        return schema_info
    
    def load_schema_from_json(self, json_file) -> Dict:
        try:
            if hasattr(json_file, 'read'):
                schema_data = json.load(json_file)
            else:
                schema_data = json.loads(json_file)
            return schema_data
        except Exception as e:
            st.error(f"Error loading JSON schema: {str(e)}")
            return {}
    
    def format_schema_for_llm(self, schema_info: Dict) -> str:
        schema_text = "Database Schema:\n\n"
        
        for table_name, table_info in schema_info.items():
            schema_text += f"Table: {table_name}\n"
            
            # Columns
            schema_text += "  Columns:\n"
            for col in table_info['columns']:
                col_desc = f"    - {col['name']} ({col['type']})"
                if not col['nullable']:
                    col_desc += " NOT NULL"
                if col['autoincrement']:
                    col_desc += " AUTO_INCREMENT"
                if col['default'] is not None:
                    col_desc += f" DEFAULT {col['default']}"
                schema_text += col_desc + "\n"
            
            # Primary Keys
            if table_info['primary_keys']:
                schema_text += f"  Primary Key: {', '.join(table_info['primary_keys'])}\n"
            
            # Foreign Keys
            if table_info['foreign_keys']:
                schema_text += "  Foreign Keys:\n"
                for fk in table_info['foreign_keys']:
                    schema_text += f"    - {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}({', '.join(fk['referred_columns'])})\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def execute_query(self, query: str, params: Dict = None) -> Tuple[bool, pd.DataFrame, str]:
        if not self.connection:
            return False, pd.DataFrame(), "No database connection"
        
        try:
            # Start transaction for non-SELECT queries
            if not query.strip().upper().startswith(('SELECT', 'WITH', 'SHOW', 'DESC', 'DESCRIBE')):
                self.connection.execute(text("START TRANSACTION"))
            
            result = self.connection.execute(text(query), params or {})
            
            if query.strip().upper().startswith(('SELECT', 'WITH', 'SHOW', 'DESC', 'DESCRIBE')):
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return True, df, "Query executed successfully"
            else:
                df = pd.DataFrame()
                return True, df, f"Query executed successfully. Rows affected: {result.rowcount}"
                
        except Exception as e:
            # Rollback on error
            try:
                if not query.strip().upper().startswith(('SELECT', 'WITH', 'SHOW', 'DESC', 'DESCRIBE')):
                    self.connection.execute(text("ROLLBACK"))
            except:
                pass
            return False, pd.DataFrame(), f"Query execution error: {str(e)}"
    
    def commit_transaction(self) -> bool:
        try:
            self.connection.execute(text("COMMIT"))
            return True
        except Exception as e:
            st.error(f"Commit error: {str(e)}")
            return False
    
    def rollback_transaction(self) -> bool:
        try:
            self.connection.execute(text("ROLLBACK"))
            return True
        except Exception as e:
            st.error(f"Rollback error: {str(e)}")
            return False

class GeminiSQLAssistant:
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def generate_sql_response(self, user_input: str, schema_info: str, chat_history: List) -> Dict:
        
        system_prompt = f"""
        You are an expert SQL assistant. Your role is to help users interact with databases through natural language.

        DATABASE SCHEMA:
        {schema_info}

        INSTRUCTIONS:
        1. Analyze the user's request and determine the best approach:
           - For data retrieval: Generate SELECT queries
           - For data modification: Generate INSERT/UPDATE/DELETE queries with safety checks
           - For analysis: Generate SQL queries and suggest visualizations
           - For schema info: Provide table/column information

        2. CRITICAL: For any request that needs data from the database, you MUST generate executable SQL queries.
        3. Format your response as follows:
           - SQL Query: Start with "SQL:" followed by the executable SQL query
           - Visualization: If appropriate, add "VIZ:" followed by chart type and description
           - Explanation: Provide brief explanation after the SQL

        4. Safety guidelines:
           - For DELETE: Always suggest previewing data first
           - For INSERT: Handle missing required fields gracefully
           - Never generate harmful queries

        5. For visualization requests, suggest appropriate chart types:
           - bar, line, pie, scatter, histogram, box

        CHAT HISTORY:
        {self.format_chat_history(chat_history)}

        USER REQUEST: {user_input}

        Respond with executable SQL and clear explanations.
        """
        
        try:
            response = self.model.generate_content(system_prompt)
            return self.parse_llm_response(response.text, user_input)
        except Exception as e:
            return {"type": "error", "content": f"Gemini API error: {str(e)}"}
    
    def parse_llm_response(self, response: str, user_input: str) -> Dict:
        response = response.strip()
        
        sql_patterns = [
            r'SQL:\s*```sql\s*(.*?)\s*```',
            r'SQL:\s*(.*?)(?=\n\n|\n[A-Z]+:|\n*$)',
            r'```sql\s*(.*?)\s*```',
            r'SELECT.*?;',
            r'INSERT.*?;',
            r'UPDATE.*?;',
            r'DELETE.*?;'
        ]
        
        sql_query = None
        for pattern in sql_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                sql_query = matches[0].strip()
                sql_query = re.sub(r'^SQL:\s*', '', sql_query)
                break
        
        viz_match = re.search(r'VIZ:\s*(\w+)\s*-\s*(.*)', response, re.IGNORECASE)
        viz_type = viz_match.group(1).lower() if viz_match else None
        viz_desc = viz_match.group(2) if viz_match else None
        
        if not viz_type:
            user_input_lower = user_input.lower()
            if any(word in user_input_lower for word in ['plot', 'chart', 'graph', 'visualize']):
                if 'histogram' in user_input_lower or 'distribution' in user_input_lower:
                    viz_type = 'histogram'
                elif 'pie' in user_input_lower:
                    viz_type = 'pie'
                elif 'scatter' in user_input_lower:
                    viz_type = 'scatter'
                elif 'box' in user_input_lower:
                    viz_type = 'box'
                elif 'line' in user_input_lower:
                    viz_type = 'line'
                else:
                    viz_type = 'bar'
        
        if sql_query:
            return {
                "type": "sql_query", 
                "content": sql_query,
                "viz_type": viz_type,
                "viz_desc": viz_desc,
                "full_response": response
            }
        elif response.startswith("CLARIFY:"):
            clarification = response.replace("CLARIFY:", "").strip()
            return {"type": "clarify", "content": clarification}
        else:
            return {"type": "message", "content": response}
    
    def format_chat_history(self, chat_history: List) -> str:
        if not chat_history:
            return "No previous conversation."
        
        history_text = "Previous conversation:\n"
        for msg in chat_history[-6:]:  # Last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            history_text += f"{role}: {content}\n"
        
        return history_text

class VisualizationEngine:
    
    @staticmethod
    def create_visualization(df: pd.DataFrame, chart_type: str = None, title: str = None):
        if df.empty:
            st.warning("No data available for visualization")
            return
        
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if chart_type == "bar":
                if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                    x_col = categorical_cols[0]
                    y_col = numeric_cols[0]
                    fig = px.bar(df, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}")
                else:
                    fig = px.bar(df, title=title or "Bar Chart")
                st.plotly_chart(fig)
                
            elif chart_type == "line":
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0] if df[numeric_cols[0]].nunique() > df[numeric_cols[1]].nunique() else numeric_cols[1]
                    y_col = numeric_cols[1] if x_col == numeric_cols[0] else numeric_cols[0]
                    fig = px.line(df, x=x_col, y=y_col, title=title or f"{y_col} over {x_col}")
                else:
                    fig = px.line(df, title=title or "Line Chart")
                st.plotly_chart(fig)
                
            elif chart_type == "pie":
                if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                    names_col = categorical_cols[0]
                    values_col = numeric_cols[0]
                    fig = px.pie(df, names=names_col, values=values_col, title=title or f"Distribution of {values_col}")
                else:
                    st.info("Pie chart requires one categorical and one numeric column")
                    st.dataframe(df)
                st.plotly_chart(fig)
                
            elif chart_type == "scatter":
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                    fig = px.scatter(df, x=x_col, y=y_col, title=title or f"{y_col} vs {x_col}")
                    st.plotly_chart(fig)
                else:
                    st.info("Scatter plot requires at least two numeric columns")
                    st.dataframe(df)
                
            elif chart_type == "histogram":
                if len(numeric_cols) >= 1:
                    x_col = numeric_cols[0]
                    fig = px.histogram(df, x=x_col, title=title or f"Distribution of {x_col}")
                    st.plotly_chart(fig)
                else:
                    st.info("Histogram requires at least one numeric column")
                    st.dataframe(df)
                
            elif chart_type == "box":
                if len(numeric_cols) >= 1:
                    y_col = numeric_cols[0]
                    fig = px.box(df, y=y_col, title=title or f"Distribution of {y_col}")
                    st.plotly_chart(fig)
                else:
                    st.info("Box plot requires at least one numeric column")
                    st.dataframe(df)
                
            else:
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            st.dataframe(df) 

class StreamlitApp:
    
    def __init__(self):
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager()
        
        self.db_manager = st.session_state.db_manager
        self.gemini_assistant = None
        self.viz_engine = VisualizationEngine()
        
        self.initialize_session_state()
        
        # Initialize Gemini from secrets
        self.initialize_gemini_from_secrets()
        
    def initialize_gemini_from_secrets(self):
        try:
            # Check if Gemini API key exists in secrets
            if 'GEMINI_API_KEY' in st.secrets:
                api_key = st.secrets['GEMINI_API_KEY']
                if api_key and api_key.strip():
                    self.gemini_assistant = GeminiSQLAssistant(api_key)
                    st.session_state.gemini_configured = True
                else:
                    st.session_state.gemini_configured = False
            else:
                st.session_state.gemini_configured = False
        except Exception as e:
            st.error(f"Error initializing Gemini from secrets: {str(e)}")
            st.session_state.gemini_configured = False
    
    def initialize_session_state(self):
        defaults = {
            'chat_history': [],
            'schema_info': {},
            'pending_delete': None,
            'pending_insert': None,
            'connection_status': "disconnected",
            'last_query_data': None,
            'last_viz_type': None,
            'last_viz_desc': None,
            'gemini_configured': False
        }
        
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def setup_sidebar(self):
        with st.sidebar:
            st.title("🔧 Configuration")
            
            # Gemini Status
            gemini_status = "🟢 Configured" if st.session_state.gemini_configured else "🔴 Not Configured"
            st.subheader(f"{gemini_status}")
            
            if not st.session_state.gemini_configured:
                st.error("""
                **Gemini API Key not found!**
                
                Please add your Gemini API key to Streamlit secrets:
                1. Create `.streamlit/secrets.toml` file
                2. Add: `GEMINI_API_KEY = "your_actual_key_here"`
                3. Redeploy the app
                """)
            
            status_color = "🟢" if self.db_manager.is_connected else "🔴"
            st.subheader(f"{status_color} Connection Status: {st.session_state.connection_status}")
            
            db_type = st.radio("Database Type", ["MySQL", "PostgreSQL"])
            
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", "localhost")
                port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
            with col2:
                username = st.text_input("Username", "root")
                password = st.text_input("Password", type="password", value="")
            
            database = st.text_input("Database Name", "classicmodels")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Connect to Database", type="primary", use_container_width=True):
                    with st.spinner("Connecting to database..."):
                        if db_type == "MySQL":
                            success = self.db_manager.connect_mysql(host, port, username, password, database)
                        else:
                            success = self.db_manager.connect_postgres(host, port, username, password, database)
                        
                        if success:
                            st.session_state.schema_info = self.db_manager.load_schema_from_db()
                            st.session_state.connection_status = "connected"
                            st.success("✅ Database connected successfully!")
                            st.rerun()
                        else:
                            st.session_state.connection_status = "disconnected"
                            st.error("❌ Failed to connect to database")
            
            with col2:
                if st.button("Disconnect", use_container_width=True):
                    self.db_manager.is_connected = False
                    self.db_manager.connection = None
                    self.db_manager.engine = None
                    st.session_state.connection_status = "disconnected"
                    st.success("Disconnected from database")
                    st.rerun()
            
            st.divider()
            st.subheader("Alternative: Upload Schema")
            schema_file = st.file_uploader("Upload Schema JSON", type=['json'])
            if schema_file is not None:
                if st.button("Load Schema from JSON", use_container_width=True):
                    st.session_state.schema_info = self.db_manager.load_schema_from_json(schema_file)
                    if st.session_state.schema_info:
                        st.success(f"✅ Schema loaded! {len(st.session_state.schema_info)} tables found.")
                    else:
                        st.error("❌ Failed to load schema from JSON")
            
            with st.expander("Sample Schema (for testing)"):
                if st.button("Load Sample Schema"):
                    sample_schema = {
                        "customers": {
                            "columns": [
                                {"name": "customerNumber", "type": "INT", "nullable": False, "autoincrement": False},
                                {"name": "customerName", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "contactLastName", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "contactFirstName", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "phone", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "addressLine1", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "addressLine2", "type": "VARCHAR(50)", "nullable": True},
                                {"name": "city", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "state", "type": "VARCHAR(50)", "nullable": True},
                                {"name": "postalCode", "type": "VARCHAR(15)", "nullable": True},
                                {"name": "country", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "salesRepEmployeeNumber", "type": "INT", "nullable": True},
                                {"name": "creditLimit", "type": "DECIMAL(10,2)", "nullable": True}
                            ],
                            "primary_keys": ["customerNumber"],
                            "foreign_keys": []
                        },
                        "products": {
                            "columns": [
                                {"name": "productCode", "type": "VARCHAR(15)", "nullable": False},
                                {"name": "productName", "type": "VARCHAR(70)", "nullable": False},
                                {"name": "productLine", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "productScale", "type": "VARCHAR(10)", "nullable": False},
                                {"name": "productVendor", "type": "VARCHAR(50)", "nullable": False},
                                {"name": "productDescription", "type": "TEXT", "nullable": False},
                                {"name": "quantityInStock", "type": "SMALLINT", "nullable": False},
                                {"name": "buyPrice", "type": "DECIMAL(10,2)", "nullable": False},
                                {"name": "MSRP", "type": "DECIMAL(10,2)", "nullable": False}
                            ],
                            "primary_keys": ["productCode"],
                            "foreign_keys": []
                        }
                    }
                    st.session_state.schema_info = sample_schema
                    st.success("✅ Sample schema loaded!")
            
            # Removed the Gemini API key input section since it's now in secrets
            
            if st.session_state.schema_info:
                st.divider()
                st.subheader("Database Schema")
                with st.expander("View Schema Details"):
                    schema_text = self.db_manager.format_schema_for_llm(st.session_state.schema_info)
                    st.text(schema_text)
                
                table_list = list(st.session_state.schema_info.keys())
                st.write(f"**Tables ({len(table_list)}):**")
                for table in table_list[:10]:
                    st.write(f"• {table}")
                if len(table_list) > 10:
                    st.write(f"• ... and {len(table_list) - 10} more tables")
            
            if self.db_manager.is_connected:
                st.divider()
                st.subheader("Transaction Controls")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Commit", use_container_width=True):
                        if self.db_manager.commit_transaction():
                            st.success("Transaction committed!")
                with col2:
                    if st.button("↩️ Rollback", use_container_width=True):
                        if self.db_manager.rollback_transaction():
                            st.success("Transaction rolled back!")
    
    def handle_chat_message(self, user_input: str):
        if not st.session_state.gemini_configured:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "❌ **Gemini API not configured. Please check the setup instructions.**"
            })
            return
        
        if not st.session_state.schema_info:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "❌ **Please connect to a database or load schema from JSON first**"
            })
            return
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        schema_text = self.db_manager.format_schema_for_llm(st.session_state.schema_info)
        
        with st.spinner("🤔 Analyzing your request..."):
            response = self.gemini_assistant.generate_sql_response(
                user_input, schema_text, st.session_state.chat_history
            )
        
        self.process_assistant_response(response, user_input)
    
    def process_assistant_response(self, response: Dict, user_input: str):
        
        if response["type"] == "sql_query":
            self.handle_sql_query(response, user_input)
            
        elif response["type"] == "clarify":
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response["content"]
            })
            
        elif response["type"] == "error":
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"❌ Error: {response['content']}"
            })
            
        else:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response["content"]
            })
    
    def handle_sql_query(self, response: Dict, user_input: str):
        query = response["content"]
        viz_type = response.get("viz_type")
        viz_desc = response.get("viz_desc")
        full_response = response.get("full_response", "")
        
        if not self.db_manager.is_connected:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Generated SQL (Database not connected - connect to execute):**\n```sql\n{query}\n```\n\n{full_response}"
            })
            return
        
        if not self.db_manager.check_connection():
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "❌ **Database connection lost. Please reconnect in the sidebar.**"
            })
            self.db_manager.is_connected = False
            st.session_state.connection_status = "disconnected"
            return
        
        if query.upper().strip().startswith('DELETE'):
            st.session_state.pending_delete = {
                'query': query,
                'user_input': user_input,
                'viz_type': viz_type,
                'viz_desc': viz_desc,
                'full_response': full_response
            }
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ **Delete Operation Requested**\n\n```sql\n{query}\n```\n\nPlease confirm this delete operation:"
            })
            return
        
        success, result, message = self.db_manager.execute_query(query)
        
        if full_response and full_response != query:
            response_content = f"{full_response}\n\n**Executed Query:**\n```sql\n{query}\n```\n\n**Result:** {message}"
        else:
            response_content = f"**Executed Query:**\n```sql\n{query}\n```\n\n**Result:** {message}"
        
        if success:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_content
            })
            
            if not result.empty:
                st.session_state.last_query_data = result
                st.session_state.last_viz_type = viz_type
                st.session_state.last_viz_desc = viz_desc
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ **Query Execution Failed**\n\n{response_content}\n\n**Error Details:** {message}"
            })

    def render_chat_interface(self):
        st.title("🤖 SQL Assistant with Gemini")
        
        # Display Gemini status
        if not st.session_state.gemini_configured:
            st.error("""
            ## 🔧 Setup Required
            
            **To use this SQL Assistant, you need to:**
            
            1. **Get a Gemini API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. **Add it to Streamlit Secrets**:
               - Local: Create `.streamlit/secrets.toml` with `GEMINI_API_KEY = "your_key"`
               - Cloud: Add to Streamlit Cloud secrets in app settings
            3. **Refresh the app**
            """)
        
        # Display connection status
        if not self.db_manager.is_connected and st.session_state.schema_info:
            st.warning("""
            **Schema loaded but no database connection** 
            - You can generate SQL queries and see the proposed SQL
            - Connect to a database in the sidebar to execute queries
            """)
        elif not self.db_manager.is_connected and not st.session_state.schema_info:
            st.error("**Please configure either:**\n- Database connection in sidebar\n- Or upload schema JSON file")
        
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    if (message["role"] == "assistant" and 
                        st.session_state.last_query_data is not None and
                        i == len(st.session_state.chat_history) - 1):
                        
                        self.display_query_results()
        
        if st.session_state.pending_delete:
            self.render_delete_confirmation()
        
        # Chat input
        st.divider()
        self.render_chat_input()
    
    def display_query_results(self):
        df = st.session_state.last_query_data
        viz_type = st.session_state.last_viz_type
        viz_desc = st.session_state.last_viz_desc
        
        if df is not None and not df.empty:
            st.subheader("📊 Query Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Total Cells", len(df) * len(df.columns))
            
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"query_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            if viz_type and len(df) > 0:
                st.subheader("📈 Visualization")
                if viz_desc:
                    st.caption(viz_desc)
                self.viz_engine.create_visualization(df, viz_type, viz_desc)
            
            st.session_state.last_query_data = None
            st.session_state.last_viz_type = None
            st.session_state.last_viz_desc = None
    
    def render_delete_confirmation(self):
        st.divider()
        pending = st.session_state.pending_delete
        
        st.warning("⚠️ **Pending Delete Operation**")
        st.code(pending['query'])
        
        if st.button("🔍 Preview Data to be Deleted", use_container_width=True):
            preview_query = pending['query'].replace('DELETE', 'SELECT *', 1)
            success, result, message = self.db_manager.execute_query(preview_query)
            
            if success and not result.empty:
                st.write("**Data that will be deleted:**")
                st.dataframe(result)
                st.info(f"This will affect {len(result)} row(s)")
            else:
                st.error(f"Could not preview data: {message}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Delete", type="primary", use_container_width=True):
                success, result, message = self.db_manager.execute_query(pending['query'])
                if success:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"✅ **Delete Operation Completed**\n\n{message}"
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ **Delete Operation Failed**\n\nError: {message}"
                    })
                st.session_state.pending_delete = None
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel Delete", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Delete operation cancelled."
                })
                st.session_state.pending_delete = None
                st.rerun()
    
    def render_chat_input(self):
        with st.expander("💡 Usage Tips & Examples", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Data Retrieval:**
                - `show all customers`
                - `find orders from last week`
                - `count products by category`
                - `display employee details`
                
                **Analysis:**
                - `plot sales by month`
                - `show credit limit distribution`
                - `analyze revenue by country`
                """)
            with col2:
                st.markdown("""
                **Data Modification:**
                - `insert new customer John Doe`
                - `update price for product X`
                - `remove customer with id 123`
                
                **Schema Exploration:**
                - `what tables are available?`
                - `describe customers table`
                - `show relationships between tables`
                """)
        
        user_input = st.chat_input("Ask about your data or enter a SQL query...")
        
        if user_input:
            self.handle_chat_message(user_input)
            st.rerun()

    def run(self):
        self.setup_sidebar()
        self.render_chat_interface()

def validate_sql_query(query: str) -> bool:
    query_upper = query.upper().strip()
    
    dangerous_patterns = [
        r'DROP\s+TABLE',
        r'DROP\s+DATABASE',
        r'TRUNCATE\s+TABLE',
        r'ALTER\s+TABLE.*DROP',
        r'CREATE\s+TABLE.*AS.*SELECT.*FROM',
        r'INSERT.*INTO.*SELECT.*FROM',
        r'UPDATE.*SET.*=.*SELECT'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query_upper, re.IGNORECASE):
            return False
    
    allowed_starters = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'SHOW', 'DESC', 'DESCRIBE']
    if not any(query_upper.startswith(starter) for starter in allowed_starters):
        return False
    
    return True

def sanitize_input(user_input: str) -> str:
    sanitized = re.sub(r'[\'\";]', '', user_input)
    return sanitized

if __name__ == "__main__":
    st.set_page_config(
        page_title="SQL Assistant with Gemini",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .chat-message.user {
        background-color: #1a1a1a;
        border-left-color: #4f8bf9;
    }
    .chat-message.assistant {
        background-color: #2d2d2d;
        border-left-color: #2e7d32;
    }
    .stButton button {
        width: 100%;
    }
    .stDownloadButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    app = StreamlitApp()
    app.run()

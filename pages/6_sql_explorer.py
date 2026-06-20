"""
SQL Query Explorer Page
Query the bank dataset using SQL directly in the app.
"""

import streamlit as st
import pandas as pd
import duckdb
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import RAW_DATA_PATH

st.set_page_config(page_title="SQL Explorer", page_icon="🗃️")

st.title("🗃️ SQL Query Explorer")

st.markdown("""
Query the Bank Marketing dataset using **SQL** directly in the browser.
Powered by [DuckDB](https://duckdb.org/) — a fast, in-process SQL engine.
""")

st.divider()

# Cache the DataFrame (pickle-serializable)
@st.cache_data
def load_dataframe():
    return pd.read_csv(RAW_DATA_PATH)

# Cache the DuckDB connection (resource, not data)
@st.cache_resource
def get_duckdb_connection():
    con = duckdb.connect(database=':memory:')
    df = load_dataframe()
    con.register('bank', df)
    return con

try:
    # Load data and connection
    df = load_dataframe()
    con = get_duckdb_connection()
    
    # Show schema
    with st.expander("📋 View Table Schema"):
        schema = con.execute("DESCRIBE bank").fetchdf()
        st.dataframe(schema)
        
        st.markdown("""
        **Sample columns:**
        - `age`, `job`, `marital`, `education`, `balance`
        - `housing`, `loan`, `contact`, `campaign`, `poutcome`
        - `deposit` (target: 'yes' or 'no')
        """)
    
    st.divider()
    
    # Pre-built queries
    st.subheader("🔍 Quick Queries")
    
    query_options = {
        "Select all records (limit 10)": "SELECT * FROM bank LIMIT 10",
        "Average balance by job": "SELECT job, AVG(balance) as avg_balance, COUNT(*) as count FROM bank GROUP BY job ORDER BY avg_balance DESC",
        "Subscription rate by education": "SELECT education, COUNT(*) as total, SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) as subscribers, ROUND(100.0 * SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) / COUNT(*), 2) as subscription_rate FROM bank GROUP BY education ORDER BY subscription_rate DESC",
        "High-balance customers (top 10)": "SELECT age, job, balance, deposit FROM bank ORDER BY balance DESC LIMIT 10",
        "Campaign effectiveness": "SELECT campaign, COUNT(*) as total, SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) as conversions, ROUND(100.0 * SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) / COUNT(*), 2) as conversion_rate FROM bank GROUP BY campaign ORDER BY campaign",
        "Age group analysis": "SELECT CASE WHEN age <= 30 THEN 'Young' WHEN age <= 45 THEN 'Adult' WHEN age <= 60 THEN 'Middle-Aged' ELSE 'Senior' END as age_group, COUNT(*) as count, AVG(balance) as avg_balance, SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) as subscribers FROM bank GROUP BY age_group ORDER BY age_group",
    }
    
    selected_query = st.selectbox("Choose a pre-built query", list(query_options.keys()))
    
    if st.button("Run Query", type="primary"):
        with st.spinner("Executing SQL..."):
            result = con.execute(query_options[selected_query]).fetchdf()
            st.success(f"✅ Returned {len(result)} rows")
            st.dataframe(result)
            
            # Download option
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="sql_query_results.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    # Custom SQL
    st.subheader("✏️ Custom SQL Query")
    
    custom_query = st.text_area(
        "Write your own SQL query",
        value="SELECT * FROM bank LIMIT 5",
        height=150
    )
    
    if st.button("Run Custom Query", type="secondary"):
        try:
            with st.spinner("Executing..."):
                custom_result = con.execute(custom_query).fetchdf()
                st.success(f"✅ Returned {len(custom_result)} rows")
                st.dataframe(custom_result)
        except Exception as e:
            st.error(f"❌ SQL Error: {e}")
    
    st.info("""
    **SQL Tips:**
    - Use `SELECT * FROM bank` to see all columns
    - Filter with `WHERE deposit = 'yes'`
    - Aggregate with `GROUP BY`
    - Sort with `ORDER BY column DESC`
    """)

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("Please ensure data/raw/bank.csv exists")
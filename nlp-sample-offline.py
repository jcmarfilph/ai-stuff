import os
import sqlalchemy
import pandas as pd
from langchain_community.llms import Ollama 
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate

def run_natural_language_query(user_prompt: str):
    # 1. Load the CSV files into DataFrames
    customers = pd.read_csv("customers.csv")
    orders = pd.read_csv("orders.csv")

    print("--- Loaded DataFrames ---")
    print(f"Customers columns: {list(customers.columns)}")
    print(f"Orders columns: {list(orders.columns)}\n")

    # Initialize a SQLAlchemy engine via duckdb_engine
    engine = sqlalchemy.create_engine("duckdb:///:memory:")

    # 2. Register the DataFrames to DuckDB via the SQLAlchemy engine
    with engine.connect() as connection:
        duckdb_conn = connection.connection
        duckdb_conn.register("customers", customers)
        duckdb_conn.register("orders", orders)

    # 3. Bind LangChain's SQL utility to our SQLAlchemy engine
    db = SQLDatabase(engine)

    # 4. Initialize LLM engine
    llm = Ollama(model="llama3", temperature=0) # Forces strict adherence to the schema

    # --- FIX: Hard-constrain the model to use ONLY your actual tables & columns ---
    custom_prompt_template = """You are a DuckDB SQL expert. Given an input question, create a syntactically correct DuckDB SQL query to run.
    CRITICAL: You are ONLY allowed to use the following two tables and their exact columns. Do NOT invent tables like 'transactions'.

    Available Tables & Columns:
    - Table name: customers
      Columns: {table_info_customers}
    
    - Table name: orders
      Columns: {table_info_orders}

    Formatting rules:
    - Output ONLY the executable SQL query string. 
    - Do NOT write conversational filler text or introductory commentary.
    - Do NOT include your thoughts.
    - Do NOT wrap the code in markdown code blocks like ```sql.

    Question: {input}
    SQLQuery:"""

    # Dynamically inject your actual DataFrame columns straight into the prompt template
    formatted_template = custom_prompt_template.format(
        table_info_customers=list(customers.columns),
        table_info_orders=list(orders.columns),
        input="{input}" # Keep this placeholder open for LangChain
    )

    PROMPT = PromptTemplate(
        input_variables=["input"], 
        template=formatted_template
    )

    # 5. Create the text-to-SQL execution chain passing the strict custom prompt
    db_chain = SQLDatabaseChain.from_llm(
        llm, 
        db, 
        prompt=PROMPT,
        verbose=True, 
        return_sql=True
    )

    print(f"--- Processing Request: '{user_prompt}' ---")
    try:
        # 6. Generate the SQL string using NLP
        generated_sql = db_chain.run(user_prompt)
        
        # Post-generation safety cleanup
        generated_sql = generated_sql.strip().replace("```sql", "").replace("```", "").strip()
        
        print("\n[GENERATED SQL CODE]:")
        print(generated_sql)
        
        # 7. Execute results back into a DataFrame using the engine
        with engine.connect() as connection:
            result_df = pd.read_sql(generated_sql, connection)
        
        print("\n[ACTUAL QUERY RESULTS]:")
        if result_df.empty:
            print("No records found matching the criteria.")
        else:
            print(result_df)
            
    except Exception as e:
        print(f"\nAn error occurred during translation or execution: {e}")

# Example Execution
if __name__ == "__main__":
    nl_query = "Show me the total amount spent by each customer name ordered from highest to lowest"
    run_natural_language_query(nl_query)

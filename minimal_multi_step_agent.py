from dotenv import load_dotenv
from llama_index.core import SQLDatabase
from llama_index.core.agent import FunctionAgent
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from config.app_context import db_manager

load_dotenv()


def fetch_data_from_db(llm_query: str) -> str:
    try:
        print("llm_query from fetch_data_from_db", llm_query)
        system_prompt = """
        Converts a natural language query into an SQL query for a SQLite database containing an 'emails' table.

        The table contains the following relevant columns with constrained values:
        - email_type: One of ['transactional', 'promotional', 'informational', 'educational', 'other']
        - category: One of ['purchase', 'travel', 'other']

        Guidelines for SQL generation:
        1. Always query from the 'emails' table.
        2. When filtering by 'vendor', use a case-insensitive LIKE clause:
        e.g., LOWER(vendor) LIKE '%<value>%'
        3. Use only valid SQLite SQL syntax.
        4. The currency is always INR.
        5. You must always return the matching rows along with the information requested in the query.
        6. Do not return id field in the response - as it is just a unique identifier.
        7. Filter out emails where amount is not present.
        """
        llm = OpenAI(model="gpt-4o-mini", system_prompt=system_prompt)

        import pandas as pd

        sql_database = SQLDatabase(db_manager.engine, include_tables=["emails"])
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=["emails"],
            llm=llm,
            synthesize_response=False,
        )
        # Get the SQL query string from the LLM
        response = query_engine.query(llm_query)
        sql_query = response.metadata["sql_query"]
        print("Generated SQL query:", sql_query)

        # Execute the SQL query directly using pandas
        with db_manager.engine.connect() as conn:
            df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        print("error in fetch_data_from_db", e)
        return f"Error in fetch_data_from_db: {str(e)}"


# print(fetch_data_from_db("all payments made to zerodha?"))

fetch_data_from_db_tool = FunctionTool.from_defaults(
    fn=fetch_data_from_db,
    name="fetch_data_from_db",
    description=(
        "Convert a natural language query into an SQL query for a SQLite database containing an 'emails' table, "
        "and return the result in table format. Use this tool when you need to answer questions about the emails table, "
        "such as filtering by vendor, email_type, category, or amount. The tool will return the matching rows as requested."
    ),
)


def execute_pandas_code_on_df(df, pandas_code: str):
    """
    Executes the given pandas code string using the provided DataFrame as 'df'.
    Returns the result of the last expression or any printed output.
    """
    import io
    import sys

    import pandas as pd

    local_vars = {"df": df, "pd": pd}
    stdout_capture = io.StringIO()
    try:
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        # Always use exec to execute the code
        exec(pandas_code, {}, local_vars)

        sys.stdout = old_stdout
        output = stdout_capture.getvalue()
        # If the code assigns a variable 'result', return it; else return printed output or success message
        result = local_vars.get("result", None)
        if result is not None:
            return result
        elif output:
            return output
        else:
            return "Code executed successfully."
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing pandas code: {str(e)}"


execute_pandas_code_on_df_tool = FunctionTool.from_defaults(
    fn=execute_pandas_code_on_df,
    name="execute_pandas_code_on_df",
    description=(
        "Execute pandas code on a given DataFrame. "
        "Takes a DataFrame (df) and a pandas code string as input. "
        "Use this tool to perform data analysis, transformations, or calculations on the provided DataFrame. "
        "The DataFrame is available as the variable 'df' in the code context. "
        "If you want to return a value, assign it to a variable named 'result'."
    ),
)


# Create the agent with the fetch_data_from_db_tool
async def main():
    agent = FunctionAgent(
        tools=[fetch_data_from_db_tool, execute_pandas_code_on_df_tool],
        llm=OpenAI(model="gpt-4o-mini"),
        verbose=True,
        system_prompt="""
        You are a helpful assistant that can answer questions about the emails table.
        You have access to the fetch_data_from_db tool to fetch data from the emails table.
        You have access to the execute_pandas_code_on_df tool to execute pandas code on a given DataFrame.
        Guidelines:
        1. fetch_data_from_db tool accepts a natural language query and returns the result in table format. It uses NLSQLTableQueryEngine internally.
        2. Dont send SQL query to the fetch_data_from_db tool, and just use the natural language query created by you to fetch the data that you will need in a single step.
        3. Use the execute_pandas_code_on_df tool to execute pandas code on a given DataFrame.
        4. If you need to perform calculations or transformations on the data, use the execute_pandas_code_on_df tool.
        """,
    )

    # Example usage:
    response = await agent.run(
        "give me month wise spend on uber, along with all uber spends?"
    )
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

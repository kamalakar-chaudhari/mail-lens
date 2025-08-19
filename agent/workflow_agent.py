import pandas as pd
from dotenv import load_dotenv
from llama_index.core import SQLDatabase
from llama_index.core.llms import ChatMessage
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from domain.vector_store_service import VectorStoreService
from infrastructure.database_manager import DatabaseManager
from infrastructure.llm_client import LLMClient

load_dotenv()


nlsql_system_prompt = """
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
7. Project all the fields that might be relevant to the query.
"""


class SQLQuery(Event):
    user_query: str


class RAGQuery(Event):
    user_query: str


class DataResult(Event):
    data_result: str
    user_query: str


class EmailQueryWorkflow(Workflow):
    """
    A workflow for processing email queries with classification and appropriate data fetching.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        openai_llm_client: LLMClient,
        timeout: float | None = 45.0,
    ):
        super().__init__(timeout=timeout)
        self.db_manager = db_manager
        self.openai_llm_client = openai_llm_client

        # Initialize the NLSQL LLM client with the system prompt
        self.openai_nlsql_llm_client = LLMClient(
            provider="openai", model="gpt-5-mini", system_prompt=nlsql_system_prompt
        )

    def _classify_query(self, user_query: str) -> str:
        """
        Classify whether the user query can be answered by simple SQL query or needs vector embeddings.
        Returns 'sql' for structured queries or 'vector' for semantic/contextual queries.
        """
        try:
            system_prompt = """
                You are a query classifier.

                You decide whether a user query can be answered from structured email table fields or needs semantic search.

                Email table fields available for SQL queries:
                - sender_name
                - sender_email
                - subject
                - date
                - email_type (transactional, promotional, informational, educational, other)
                - category (purchase, travel, other)
                - amount
                - vendor
                - body
                - item
                - summary
                - source
                - destination

                Guidelines:
                1. If the query can be answered entirely by filtering, aggregating, or matching these fields → return "sql".
                2. If the query needs full-text understanding of the email body or contextual/semantic meaning → return "vector".
                3. It is preferred to classify as vector instead of sql if the query will consist of search keyword heuristics in subject or body.
                4. Output exactly "sql" or "vector". No extra text.
            """

            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_query),
            ]
            response = self.openai_llm_client.chat_content(messages)
            return response.lower()
        except Exception as e:
            print(f"Error in classify_query: {e}")
            return "sql"  # Default to SQL for safety

    def _fetch_data_from_db(self, llm_query: str) -> str:
        """
        Convert a natural language query into an SQL query for a SQLite database containing an 'emails' table,
        and return the result in table format.
        """
        try:
            print("llm_query from fetch_data_from_db", llm_query)

            sql_database = SQLDatabase(
                self.db_manager.engine, include_tables=["emails"]
            )
            query_engine = NLSQLTableQueryEngine(
                sql_database=sql_database,
                tables=["emails"],
                llm=self.openai_nlsql_llm_client,
                synthesize_response=False,
            )
            # Get the SQL query string from the LLM
            response = query_engine.query(llm_query)
            sql_query = response.metadata["sql_query"]
            print("Generated SQL query:", sql_query)

            # Execute the SQL query directly using pandas
            with self.db_manager.engine.connect() as conn:
                df = pd.read_sql_query(sql_query, conn)
            return df.to_string()
        except Exception as e:
            print("error in fetch_data_from_db", e)
            return f"Error in fetch_data_from_db: {str(e)}"

    def _get_similar_embeddings(self, query: str, top_k: int = 5) -> str:
        """
        Find similar emails using vector embeddings based on semantic similarity.
        """
        try:
            # Initialize vector store service
            vector_service = VectorStoreService()

            # Search for similar emails
            results = vector_service.search_similar_emails(
                query=query, top_k=top_k, similarity_threshold=0.5
            )

            if not results:
                return "No similar emails found."

            # Format the response
            result = f"Found {len(results)} similar emails:\n\n"
            for i, result_item in enumerate(results, 1):
                metadata = result_item["metadata"]
                result += f"{i}. Score: {result_item['score']:.3f}\n"
                result += f"   Subject: {metadata.get('subject', 'N/A')}\n"
                result += f"   Content: {result_item['content'][:300]}...\n\n"
                result += f"   Metadata: {metadata}\n"

            return result
        except Exception as e:
            print(f"Error in get_similar_embeddings: {e}")
            return f"Error in get_similar_embeddings: {str(e)}"

    def _generate_final_response(self, user_query: str, data_result: str) -> str:
        """
        Generate a final response based on the user query and the data result from previous steps.
        """
        try:
            system_prompt = (
                "You are an expert assistant. Your task is to answer the user's query using ONLY the provided data results. "
                "Carefully analyze the user's question and the data. "
                "If the data contains relevant information, summarize the key findings with specific details (such as IDs, dates, or main fields) and explain their relevance. "
                "If no relevant data is found, clearly state that there are no matching results. "
                "If there are errors or inconsistencies in the data, acknowledge them and explain their impact. "
                "Keep your answer concise, accurate, and user-friendly. "
                "Do not make assumptions or add information not present in the data. "
                "Present your answer as a short paragraph, bullet points, or a compact table as appropriate."
                "Do not show ID in the response. It is just a unique identifier."
            )

            # Provide the user query and data result as separate, clearly labeled messages
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=f"User Query:\n{user_query}"),
                ChatMessage(role="user", content=f"Data Result:\n{data_result}"),
                ChatMessage(
                    role="user",
                    content=(
                        "Based on the above user query and data result, provide a clear, direct, and helpful answer. "
                        "If possible, highlight the most relevant data. If nothing matches, say so explicitly."
                    ),
                ),
            ]
            return self.openai_llm_client.chat_content(messages)
        except Exception as e:
            print(f"Error in generate_final_response: {e}")
            return f"Error generating final response: {str(e)}"

    @step
    def decide_sql_or_rag(self, ev: StartEvent) -> SQLQuery | RAGQuery:
        """
        Step 1: Classify whether the user query can be answered by SQL or needs vector embeddings.
        """
        user_query = ev.input
        query_type = self._classify_query(user_query)
        print(f"Query classified as: {query_type}")
        return (
            SQLQuery(user_query=user_query)
            if query_type == "sql"
            else RAGQuery(user_query=user_query)
        )

    @step
    def execute_sql_query(self, ev: SQLQuery) -> DataResult:
        """
        Step 2: Fetch data using SQL query.
        """
        data_result = self._fetch_data_from_db(ev.user_query)
        print(f"Data result: {data_result[:200]}...")
        return DataResult(data_result=data_result, user_query=ev.user_query)

    @step
    def get_similar_embeddings(self, ev: RAGQuery) -> DataResult:
        """
        Step 2: Fetch data using vector embeddings.
        """
        data_result = self._get_similar_embeddings(ev.user_query)
        return DataResult(data_result=data_result, user_query=ev.user_query)

    @step
    def synthesize_response(self, ev: DataResult) -> StopEvent:
        """
        Step 3: Generate final response based on user query and data result.
        """
        final_response = self._generate_final_response(ev.user_query, ev.data_result)
        return StopEvent(result=final_response)

    def _manual_execution(self, user_query: str) -> str:
        """Manual execution as fallback"""
        print("Using manual execution fallback")
        query_type = self._classify_query(user_query)
        if query_type == "sql":
            data_result = self._fetch_data_from_db(user_query)
        else:
            data_result = self._get_similar_embeddings(user_query)
        return self._generate_final_response(user_query, data_result)


# Create a global instance of the workflow
# email_workflow = EmailQueryWorkflow() # This line is removed as per the edit hint


# Example usage
async def main():
    # Test the workflow with different types of queries
    test_queries = [
        # "Show me all emails from Gmail",
        # "Find emails about travel expenses",
        # "What's my total spend on Uber this month?",
        # "Show me emails discussing project deadlines",
        # "show me emails related to cancellations of train tickets",
        "emails from my daughter's school"
    ]

    # Create workflow instance
    # Note: In a real application, these dependencies would be injected
    # For testing purposes, we'll create them here
    from config.constants import EMAILS_DB_PATH
    from infrastructure.database_manager import DatabaseManager
    from infrastructure.llm_client import LLMClient

    db_manager = DatabaseManager(EMAILS_DB_PATH)
    openai_llm_client = LLMClient(provider="openai", model="gpt-5-mini")
    workflow = EmailQueryWorkflow(
        db_manager=db_manager, openai_llm_client=openai_llm_client, timeout=60
    )

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")

        try:
            result = await workflow.run(input=query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

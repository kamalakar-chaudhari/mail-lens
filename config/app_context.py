from agent.workflow_agent import EmailQueryWorkflow
from config.constants import EMAILS_DB_PATH
from domain.email_repository import EmailRepository
from infrastructure.database_manager import DatabaseManager
from infrastructure.llm_client import LLMClient

db_manager = DatabaseManager(EMAILS_DB_PATH)
email_repository = EmailRepository(db_manager)
openai_llm_client = LLMClient(provider="openai", model="gpt-5-mini")
workflow_agent = EmailQueryWorkflow(
    db_manager=db_manager, openai_llm_client=openai_llm_client, timeout=60
)

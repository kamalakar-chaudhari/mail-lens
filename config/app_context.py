from config.constants import EMAILS_DB_PATH
from domain.email_repository import EmailRepository
from infrastructure.database_manager import DatabaseManager
from infrastructure.llm_client import LLMClient

# Initialize database and repository
db_manager = DatabaseManager(EMAILS_DB_PATH)
email_repository = EmailRepository(db_manager)
openai_llm_client = LLMClient(provider="openai", model="gpt-5-mini")

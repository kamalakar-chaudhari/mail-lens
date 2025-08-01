from infrastructure.database_manager import DatabaseManager
from domain.email_repository import EmailRepository
from config.constants import EMAILS_DB_PATH

# Initialize database and repository
db_manager = DatabaseManager(EMAILS_DB_PATH)
email_repository = EmailRepository(db_manager)

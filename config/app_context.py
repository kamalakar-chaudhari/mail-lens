from config.constants import EMAILS_DB_PATH
from domain.email_repository import EmailRepository
from infrastructure.database_manager import DatabaseManager

# Initialize database and repository
db_manager = DatabaseManager(EMAILS_DB_PATH)
email_repository = EmailRepository(db_manager)

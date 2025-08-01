from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Optional


class DatabaseManager:
    """Database connection manager."""

    _instance: Optional["DatabaseManager"] = None
    _initialized = False

    def __new__(cls, db_path: str = "emails.db"):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "emails.db"):
        if not self._initialized:
            self.db_path = db_path
            self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
            self.Session = sessionmaker(bind=self.engine)
            self._initialized = True

    def get_session(self):
        return self.Session()

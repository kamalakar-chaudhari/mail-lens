from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from infrastructure.database_manager import DatabaseManager

Base = declarative_base()


class Email(Base):
    """SQLAlchemy model for emails table."""

    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(500), nullable=False)
    sender_name = Column(String(255), nullable=False)
    sender_email = Column(String(255), nullable=False)
    date = Column(String(100))
    body = Column(Text)
    email_type = Column(String(50))
    category = Column(String(50))
    vendor = Column(String(255))
    item = Column(String(255))
    source = Column(String(100))
    destination = Column(String(100))
    amount = Column(String(50))
    summary = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    # Create indexes for better performance
    __table_args__ = (
        Index("idx_emails_created_at", "created_at"),
        Index("idx_emails_sender_email", "sender_email"),
        Index("idx_emails_subject", "subject"),
        Index("idx_emails_email_type", "email_type"),
        Index("idx_emails_category", "category"),
        Index("idx_emails_vendor", "vendor"),
    )

    @classmethod
    def create_schema(cls, engine):
        """Create the emails table schema if it doesn't exist."""
        Base.metadata.create_all(engine, tables=[cls.__table__])


class EmailRepository:
    """SQLAlchemy implementation of EmailRepository."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def init_db(self):
        """Initialize the database and create tables."""
        Email.create_schema(self.db_manager.engine)

    def update_emails(self, emails: list[dict[str, Any]]) -> list[int]:
        """
        Update emails with annotated fields using bulk operations.

        Args:
            emails: List of email dictionaries with id and fields to update

        Returns:
            List of updated email IDs
        """
        session = self.db_manager.get_session()
        try:
            # Use bulk update with executemany for better performance
            session.execute(
                text(
                    """
                    UPDATE emails SET
                        summary = :summary,
                        amount = :amount,
                        email_type = :email_type,
                        category = :category,
                        vendor = :vendor,
                        item = :item,
                        source = :source,
                        destination = :destination,
                        updated_at = datetime('now')
                    WHERE id = :id
                    """
                ),
                emails,
            )
            session.commit()
            return [email.get("id", "") for email in emails]
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def store_emails(self, emails: list[dict[str, Any]]) -> list[int]:
        """
        Store a list of emails using batch insertion.

        Args:
            emails: List of email dictionaries

        Returns:
            List of inserted email IDs
        """
        session = self.db_manager.get_session()
        try:
            # Get the current max ID to calculate new IDs
            max_id_result = session.execute(
                text("SELECT COALESCE(MAX(id), 0) FROM emails")
            )
            max_id = max_id_result.scalar() or 0

            # Prepare batch data
            batch_data = []
            for email in emails:
                batch_data.append(
                    {
                        "subject": email.get("subject", ""),
                        "sender_name": email.get("sender_name", ""),
                        "sender_email": email.get("sender_email", ""),
                        "date": email.get("date", ""),
                        "body": email.get("body", ""),
                        "email_type": email.get("type", ""),
                        "category": email.get("category", ""),
                        "vendor": email.get("vendor", ""),
                        "item": email.get("item", ""),
                        "source": email.get("source", ""),
                        "destination": email.get("destination", ""),
                        "amount": email.get("amount", ""),
                        "summary": email.get("summary", ""),
                    }
                )

            # Use raw SQL for batch insert
            session.execute(
                text(
                    """
                INSERT INTO emails (
                    subject, sender_name, sender_email, date, body, email_type, category,
                    vendor, item, source, destination,
                    amount, summary
                ) VALUES (
                    :subject, :sender_name, :sender_email, :date, :body, :email_type, :category,
                    :vendor, :item, :source, :destination,
                    :amount, :summary
                )
                """
                ),
                batch_data,
            )

            session.commit()

            # Calculate the IDs of inserted rows
            email_ids = list(range(max_id + 1, max_id + 1 + len(emails)))

            return email_ids

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_emails(
        self, start: int = 0, limit: int = 100, projected_fields: list[str] = None
    ) -> list[dict[str, Any]]:
        """
        Get emails with pagination.

        Args:
            start: Starting offset
            limit: Maximum number of emails to return
            projected_fields: List of fields to include in each email dictionary

        Returns:
            List of email dictionaries
        """
        # Default projected fields if not provided
        if projected_fields is None:
            projected_fields = [
                "id",
                "subject",
                "sender_name",
                "sender_email",
                "date",
                "body",
                "email_type",
                "category",
                "vendor",
                "item",
                "source",
                "destination",
                "amount",
                "summary",
                "created_at",
            ]

        session = self.db_manager.get_session()
        try:
            emails = (
                session.query(Email)
                .order_by(Email.created_at.desc())
                .offset(start)
                .limit(limit)
                .all()
            )

            # Convert to dictionaries using the projected fields
            result = []
            for email in emails:
                email_dict = {}
                for field in projected_fields:
                    value = getattr(email, field, None)
                    if field in ["created_at", "updated_at"]:
                        email_dict[field] = value.isoformat() if value else None
                    else:
                        email_dict[field] = value
                result.append(email_dict)

            return result

        finally:
            session.close()

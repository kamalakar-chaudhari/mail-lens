from email import policy
import mailbox
import itertools
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from bs4 import BeautifulSoup
from email.utils import parseaddr
from domain.email_repository import EmailRepository


class EmailIngestionService:
    """
    Ingests emails from mbox files into a database with batch processing.

    Parses email content (subject, sender, body) and stores in database
    using configurable batch sizes for memory efficiency.
    """

    def __init__(self, mbox_path: str, email_repository: EmailRepository):
        self.mbox_path = mbox_path
        self.db = email_repository
        self.db.init_db()

    def _parse_body(self, msg):
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    body += part.get_content()
                elif ctype == "text/html" and not body:
                    html_content = part.get_content()
                    soup = BeautifulSoup(html_content, "html.parser")
                    body = soup.get_text(separator="\n", strip=True)
        else:
            ctype = msg.get_content_type()
            if ctype == "text/plain":
                body = msg.get_content()
            elif ctype == "text/html":
                html_content = msg.get_content()
                soup = BeautifulSoup(html_content, "html.parser")
                body = soup.get_text(separator="\n", strip=True)
        return body

    def _parse_email(self, email):
        raw_bytes = email.as_bytes()
        msg = BytesParser(policy=policy.default).parse(BytesIO(raw_bytes))

        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        sender_name, sender_email = parseaddr(sender)
        date = msg.get("date", "")
        body = self._parse_body(msg)[:4000]

        return {
            "subject": subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "date": date,
            "body": body,
        }

    def ingest(self, start: int = 0, limit: int = None, batch_size: int = 100) -> int:
        """
        Ingest emails from mbox file in batches.

        Args:
            start: Starting index in the mbox file
            limit: Maximum number of emails to ingest (None for all)
            batch_size: Number of emails to process in each batch

        Returns:
            Number of emails ingested
        """
        mbox = mailbox.mbox(self.mbox_path)
        total_ingested = 0

        try:
            # Calculate end index
            end_index = start + limit if limit else None

            # Process emails in batches
            current_index = start
            while True:
                # Determine batch end
                batch_end = current_index + batch_size
                if end_index and batch_end > end_index:
                    batch_end = end_index

                # Get batch of emails
                emails = itertools.islice(mbox, current_index, batch_end)
                parsed_emails = []

                # Parse emails in batch
                for email in emails:
                    email_dict = self._parse_email(email)
                    parsed_emails.append(email_dict)

                # If no emails in batch, we're done
                if not parsed_emails:
                    break

                # Store batch in database
                self.db.store_emails(parsed_emails)
                total_ingested += len(parsed_emails)

                # Move to next batch
                current_index = batch_end

                # Check if we've reached the limit
                if end_index and current_index >= end_index:
                    break

        finally:
            mbox.close()

        return total_ingested

from typing import Any, Literal

from groq import Groq
from pydantic import BaseModel, Field

from domain.email_repository import EmailRepository


class EmailParseResponse(BaseModel):
    id: int = Field(..., description="Unique identifier for the email")
    email_type: Literal[
        "transactional", "promotional", "informational", "educational", "other"
    ] = Field(..., description="Email type")
    category: Literal["purchase", "travel", "other"] = Field(
        ..., description="Category of the email"
    )
    vendor: str = Field("", description="Company or sender name")
    item: str = Field("", description="Product or service booked or purchased")
    date: str = Field("", description="Date in the email or booking date")
    source: str = Field("", description="Origin city name")
    destination: str = Field("", description="Final destination city name")
    amount: str = Field("", description="Numeric amount without currency symbols")
    summary: str = Field(
        "", description="1-2 sentence human-readable summary of the email content"
    )


class EmailParseResponseBulk(BaseModel):
    emails: list[EmailParseResponse] = Field(..., description="List of parsed emails")


class EmailAnnotationService:
    """
    Service to annotate emails using LLM and update database with results.
    """

    def __init__(self, groq_llama_client: Groq, email_repository: EmailRepository):
        self.llm_client = groq_llama_client
        self.email_repository = email_repository

    def annotate_emails(self, start=0, limit=None, batch_size=10):
        """
        Annotate emails in batches starting from the given offset.

        Args:
            start: Starting offset for email processing
            limit: Maximum number of emails to process (None for all)
            batch_size: Number of emails to process in each batch

        Returns:
            Number of emails processed
        """
        total_processed = 0
        current_offset = start

        while True:
            # Calculate batch limit
            batch_limit = batch_size
            if limit and (current_offset + batch_size) > limit:
                batch_limit = limit - current_offset
                if batch_limit <= 0:
                    break

            # Get batch of emails from database
            emails = self.email_repository.get_emails(
                start=current_offset,
                limit=batch_limit,
                projected_fields=[
                    "id",
                    "subject",
                    "sender_name",
                    "sender_email",
                    "date",
                    "body",
                ],
            )

            if not emails:
                print(f"No more emails found at offset {current_offset}")
                break

            print(
                f"Processing batch: {len(emails)} emails starting from offset {current_offset}"
            )

            # Send emails to LLM for annotation
            annotated_emails = self._send_emails_to_llm(emails)

            if annotated_emails:
                # Update database with annotated fields
                self._update_inferred_fields_in_db(annotated_emails)
                total_processed += len(annotated_emails)
                print(f"Successfully processed {len(annotated_emails)} emails")
            else:
                print(
                    f"Failed to annotate emails in batch starting at offset {current_offset}"
                )

            # Move to next batch
            current_offset += len(emails)

            # Check if we've reached the limit
            if limit and current_offset >= limit:
                break

        return total_processed

    def _send_emails_to_llm(self, emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Send emails to LLM for annotation and return structured results.

        Args:
            emails: List of email dictionaries with id, subject, sender_name, sender_email, date, body

        Returns:
            List of email dictionaries with annotated fields added
        """
        if not emails:
            return []

        system_message = {
            "role": "system",
            "content": """
                You are an Email Parser. Your job is to analyze email content and output structured JSON.

                ## Task:
                Extract the following fields from the parsed email text:
                {
                "email_type": "transactional | promotional | informational | educational | other",
                "category": "purchase | travel | other",
                "vendor": "<company or sender name>",
                "item": "<product/service booked or purchased>",
                "date": "<date in email or booking date>",
                "source": "<origin city name>",
                "destination": "<final destination city name>",
                "amount": "<numeric amount without currency or symbols>",
                "summary": "<1-2 sentence human-readable summary of the email content>"
                }

                ## Parsing Rules:

                1. Output:
                - Return ONLY valid JSON.
                - Do NOT include explanations or extra text outside the JSON.

                2. Unknown / Unclear Fields:
                - If the email does NOT clearly provide a value for a field, output an empty string "".

                3. Email Type (email_type):
                - "transactional" ONLY IF the email CONFIRMS a purchase, payment, or booking.
                    * Must clearly mention an order number, invoice, or successful payment/booking.
                - "informational" for delivery notifications, shipment tracking, OTPs, account alerts, or updates
                    that do NOT involve new purchases.
                - "promotional" for marketing, offers, ads, newsletters, or ANY uncertain case.
                - IF unsure, ALWAYS choose "promotional".

                4. Category (category):
                - "purchase" ONLY IF an item/service was BOUGHT AND PAID for, with clear evidence
                    such as an order, invoice, or receipt.
                - "travel" ONLY IF the email is about flight, train, bus, or hotel bookings.
                - "other" for all other cases.
                - DO NOT infer "purchase" if no transaction is explicitly confirmed.

                5. Amount (amount):
                - Extract the first clearly indicated TOTAL amount.
                - Return digits ONLY (no currency symbols, commas, or extra text).
                - If no clear total amount is present, return "".

                6. Source & Destination:
                - Use full official city names ONLY (e.g., "Mumbai" not "BOM").
                - Destination = final destination ONLY, exclude layovers.

                7. Summary (summary):
                - Provide 1-2 sentences summarizing the MAIN purpose of the email.
                - Avoid speculation or assumptions.

                8. Strict Classification Reminder:
                - NEVER classify as "transactional" or "purchase" unless there is unambiguous evidence
                    of a completed order or booking.
                - In case of doubt, choose "informational" or "promotional".
            """,
        }

        user_messages = [
            {"role": "user", "content": str(email_data)} for email_data in emails
        ]

        try:
            completion = self.llm_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[system_message, *user_messages],
                temperature=0,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "EmailParseResponseBulk",
                        "schema": EmailParseResponseBulk.model_json_schema(),
                    },
                },
            )

            # Parse the LLM response
            response_content = completion.choices[0].message.content
            parsed_response = EmailParseResponseBulk.model_validate_json(
                response_content
            )

            # Combine original emails with LLM annotations
            annotated_emails = []
            for i, email in enumerate(emails):
                if i < len(parsed_response.emails):
                    annotated_email = email.copy()
                    llm_result = parsed_response.emails[i]

                    # Add LLM annotations to email dict
                    annotated_email.update(
                        {
                            "email_type": llm_result.email_type,
                            "category": llm_result.category,
                            "vendor": llm_result.vendor,
                            "item": llm_result.item,
                            "date": llm_result.date,
                            "source": llm_result.source,
                            "destination": llm_result.destination,
                            "amount": llm_result.amount,
                            "summary": llm_result.summary,
                        }
                    )
                    annotated_emails.append(annotated_email)
                else:
                    # If LLM didn't return enough results, keep original email
                    annotated_emails.append(email)

            return annotated_emails

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return []

    def _update_inferred_fields_in_db(self, emails: list[dict[str, Any]]) -> list[int]:
        """
        Update emails in database with annotated fields.

        Args:
            emails: List of email dictionaries with id and annotated fields

        Returns:
            List of updated email IDs
        """
        if not emails:
            return []

        try:
            # Prepare emails for database update
            update_data = []
            for email in emails:
                update_data.append(
                    {
                        "id": email["id"],
                        "summary": email.get("summary", ""),
                        "amount": email.get("amount", ""),
                        "email_type": email.get("email_type", ""),
                        "category": email.get("category", ""),
                        "vendor": email.get("vendor", ""),
                        "item": email.get("item", ""),
                        "source": email.get("source", ""),
                        "destination": email.get("destination", ""),
                    }
                )

            # Update emails in database
            updated_ids = self.email_repository.update_emails(update_data)
            return updated_ids

        except Exception as e:
            print(f"Error updating emails in database: {e}")
            return []

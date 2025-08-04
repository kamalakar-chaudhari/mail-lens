#!/usr/bin/env python3
"""
Email annotation script that demonstrates how to use EmailAnnotationService
to read emails from database, send them to LLM for annotation, and update the database.
"""

import argparse
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq

from config.app_context import email_repository
from domain.email_annotation_service import EmailAnnotationService


def annotate_batch(start=0, limit=None, batch_size=10):
    """Main function to demonstrate EmailAnnotationService usage."""

    # Initialize the database (create tables if they don't exist)
    email_repository.init_db()
    print("Database initialized successfully!")

    # Initialize Groq client
    try:
        groq_client = Groq()
        print("Groq client initialized successfully!")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        print("Make sure you have set the GROQ_API_KEY environment variable")
        return

    print("Starting email annotation")
    print(f"Parameters: start={start}, limit={limit}, batch_size={batch_size}")

    # Create EmailAnnotationService instance
    annotation_service = EmailAnnotationService(groq_client, email_repository)

    try:
        # Annotate emails using the service
        total_processed = annotation_service.annotate_emails(
            start=start, limit=limit, batch_size=batch_size
        )
        print(f"Successfully processed {total_processed} emails!")

    except Exception as e:
        print(f"Error during email annotation: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate emails from database using LLM"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Starting offset in database (default: 0)"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of emails to process (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )

    args = parser.parse_args()

    annotate_batch(
        start=args.start,
        limit=200,
        batch_size=args.batch_size,
        # start=50,
        # limit=60,
        # batch_size=args.batch_size,
    )

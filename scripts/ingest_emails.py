#!/usr/bin/env python3
"""
Email ingestion script that demonstrates how to access EmailRepository
from the app context and use EmailIngestionService for mbox file processing.
"""

import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_context import email_repository
from domain.email_ingestion_service import EmailIngestionService


def ingest_batch(mbox, start=0, limit=None, batch_size=100):
    """Main function to demonstrate EmailIngestionService usage."""

    # Initialize the database (create tables if they don't exist)
    email_repository.init_db()
    print("Database initialized successfully!")

    # If mbox file is provided, use EmailIngestionService
    if mbox:
        if not os.path.exists(mbox):
            print(f"Error: Mbox file '{mbox}' not found!")
            return

        print(f"Starting email ingestion from: {mbox}")
        print(f"Parameters: start={start}, limit={limit}, batch_size={batch_size}")

        # Create EmailIngestionService instance with the repository from app context
        ingestion_service = EmailIngestionService(mbox, email_repository)

        try:
            # Ingest emails using the service
            total_ingested = ingestion_service.ingest(
                start=start, limit=limit, batch_size=batch_size
            )
            print(f"Successfully ingested {total_ingested} emails!")

        except Exception as e:
            print(f"Error during email ingestion: {e}")
            return

    else:
        print(
            "No mbox file specified. Please provide the --mbox argument to ingest emails."
        )
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest emails from mbox file")
    parser.add_argument("--mbox", type=str, help="Path to mbox file to ingest")
    parser.add_argument(
        "--start", type=int, default=0, help="Starting index in mbox file (default: 0)"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of emails to ingest (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )

    # main(
    #     mbox=parser.parse_args().mbox,
    #     start=parser.parse_args().start,
    #     limit=parser.parse_args().limit,
    #     batch_size=parser.parse_args().batch_size,
    # )

    ingest_batch(
        mbox="/home/kamal/work/ai/alt-oracle/data/gmail.mbox",
        start=0,
        limit=1000,
        batch_size=100,
    )

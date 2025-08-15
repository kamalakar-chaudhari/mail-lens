#!/usr/bin/env python3
"""
Script to build vector embeddings index from emails in the database.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from config.app_context import email_repository
from domain.vector_store_service import VectorStoreService

load_dotenv()

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_index(
    start=0, limit=None, batch_size=100, clear=False, test=False, test_query=None
):
    """Build vector index from emails."""

    # Initialize the database
    email_repository.init_db()
    logger.info("Database initialized successfully!")

    # Initialize vector store service
    try:
        vector_service = VectorStoreService()
        logger.info("Vector store service initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing vector store service: {e}")
        logger.error("Make sure you have set the OPENAI_API_KEY environment variable")
        return

    # Build index
    try:
        total_processed = vector_service.build_index_from_emails(
            email_repository=email_repository,
            start=start,
            limit=limit,
            batch_size=batch_size,
            clear_existing=clear,
        )
        logger.info(f"Successfully processed {total_processed} emails!")

        # Get index statistics
        stats = vector_service.get_index_stats()
        logger.info(f"Index statistics: {stats}")

        # Test search if requested
        if test:
            if test_query:
                query = test_query
            else:
                query = "travel expenses"

            logger.info(f"Testing search with query: '{query}'")
            results = vector_service.search_similar_emails(query, top_k=3)

            if results:
                logger.info("Search results:")
                for i, result in enumerate(results, 1):
                    logger.info(f"{i}. Score: {result['score']:.3f}")
                    logger.info(
                        f"   Subject: {result['metadata'].get('subject', 'N/A')}"
                    )
                    logger.info(f"   Content: {result['content'][:200]}...")
                    logger.info("---")
            else:
                logger.info("No search results found.")

    except Exception as e:
        logger.error(f"Error during index building: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build vector embeddings index from emails"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Starting offset for email processing"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of emails to process"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Batch size for processing"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing index before building"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test search after building index"
    )
    parser.add_argument(
        "--test-query", type=str, help="Query to test search functionality"
    )

    args = parser.parse_args()

    build_index(
        start=args.start,
        limit=args.limit,
        batch_size=args.batch_size,
        clear=args.clear,
        test=args.test,
        test_query=args.test_query,
    )

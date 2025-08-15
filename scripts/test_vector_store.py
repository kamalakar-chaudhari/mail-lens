import pprint

from dotenv import load_dotenv

load_dotenv()

from domain.vector_store_service import VectorStoreService

vector_store_service = VectorStoreService()

results = vector_store_service.search_similar_emails(
    query="irctc cancelled tickets",
    top_k=5,
    similarity_threshold=0.5,
)
pprint.pprint(results)

"""
Benchmark script for testing ACL filtering performance in Qdrant.
Inserts many chunks with the same ACL, then measures latency of filtering by that ACL.
"""

import time

from fastembed import SparseTextEmbedding
from fastembed import TextEmbedding
from qdrant_client.models import Distance
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
from qdrant_client.models import SparseVectorParams
from qdrant_client.models import VectorParams

from scratch.qdrant.client import QdrantClient
from scratch.qdrant.fake_chunk_helpers import generate_fake_qdrant_chunks
from scratch.qdrant.schemas.collection_name import CollectionName
from scratch.qdrant.service import QdrantService


def main():
    collection_name = CollectionName.TEST_COLLECTION
    dense_model_name = "nomic-ai/nomic-embed-text-v1"
    sparse_model_name = "prithivida/Splade_PP_en_v1"

    # Initialize client and service
    dense_embedding_model = TextEmbedding(model_name=dense_model_name)
    sparse_embedding_model = SparseTextEmbedding(model_name=sparse_model_name)
    service = QdrantService(client=QdrantClient())

    # Get embedding size for dense vectors
    dense_embedding_size = service.client.get_embedding_size(dense_model_name)

    # Delete and recreate collection
    print(f"Setting up collection: {collection_name}")
    service.client.delete_collection(collection_name=collection_name)
    service.client.create_collection(
        collection_name=collection_name,
        dense_vectors_config={
            "dense": VectorParams(size=dense_embedding_size, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    print(f"Collection {collection_name} created\n")

    # Define a common ACL that will be shared by many documents
    common_acl = ["user_email:test@example.com", "group:Engineering", "PUBLIC"]
    print("Common ACL for filtering:", common_acl, "\n")

    # Generate fake chunks - assign the common ACL to 70% of them
    # Use same content for all to isolate filtering performance from ranking
    num_fake_chunks = 1000
    same_content = (
        "This is the same content for all chunks to test filtering performance."
    )
    print(f"Generating {num_fake_chunks} fake chunks with identical content...")
    fake_chunks = []
    for chunk in generate_fake_qdrant_chunks(num_fake_chunks, content=same_content):
        # 70% of chunks will have the common ACL
        if len(fake_chunks) < num_fake_chunks * 0.7:
            chunk.access_control_list = common_acl
        fake_chunks.append(chunk)

    print(f"Generated {len(fake_chunks)} chunks")
    print(f"  - {int(num_fake_chunks * 0.7)} chunks with common ACL")
    print(f"  - {int(num_fake_chunks * 0.3)} chunks with random ACLs\n")

    # Write to Qdrant using service
    print("Writing chunks to Qdrant...")
    start_time = time.time()
    result = service.embed_and_upsert_chunks(
        fake_chunks, dense_embedding_model, sparse_embedding_model, collection_name
    )
    elapsed_time = time.time() - start_time
    print(f"Upsert result: {result}")
    print(f"Time taken: {elapsed_time:.2f} seconds\n")

    # Benchmark: Filter by the common ACL
    print("=" * 60)
    print("BENCHMARK: Filtering by ACL with Hybrid Search")
    print("=" * 60)

    # Test filtering by one ACL entry that should match ~700 documents
    test_acl_entry = common_acl[0]  # "user_email:test@example.com"
    print(f"\nFiltering by ACL entry: '{test_acl_entry}'")
    print(f"Expected matches: ~{int(num_fake_chunks * 0.7)} documents\n")

    # Create ACL filter
    acl_filter = Filter(
        must=[
            FieldCondition(
                key="access_control_list", match=MatchValue(value=test_acl_entry)
            )
        ]
    )

    query_text = (
        "this is the same content for all chunks to test filtering performance."
    )

    # Measure filtering latency (single run)
    start = time.time()

    # Dense search with ACL filter (simpler than hybrid for benchmarking)
    search_result = service.dense_search(
        query_text=query_text,
        dense_embedding_model=dense_embedding_model,
        collection_name=collection_name,
        limit=1000,  # Set high to see all matching results
        query_filter=acl_filter,
    )

    latency = time.time() - start

    print("Search completed:")
    print(f"  Found: {len(search_result.points)} results")
    print(f"  Latency: {latency * 1000:.2f}ms")

    # Show sample results
    print("\nSample Filtered Results (first 3):")
    for idx, point in enumerate(search_result.points[:3], 1):
        print(f"\n{idx}. Score: {point.score:.4f}")
        print(f"   ID: {point.id}")
        print(f"   Document ID: {point.payload.get('document_id')}")
        print(f"   Source: {point.payload.get('source_type')}")
        print(f"   ACL: {point.payload.get('access_control_list')}")
        print(f"   Content: {point.payload.get('content')[:50]}...")


if __name__ == "__main__":
    main()

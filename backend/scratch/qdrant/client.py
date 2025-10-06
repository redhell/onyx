from qdrant_client import QdrantClient as ThirdPartyQdrantClient
from qdrant_client.models import CollectionInfo
from qdrant_client.models import Filter
from qdrant_client.models import FusionQuery
from qdrant_client.models import PointStruct
from qdrant_client.models import Prefetch
from qdrant_client.models import QueryResponse
from qdrant_client.models import SparseVectorParams
from qdrant_client.models import UpdateResult
from qdrant_client.models import VectorParams

from scratch.qdrant.config import QdrantConfig
from scratch.qdrant.schemas.collection_name import CollectionName
from scratch.qdrant.schemas.collection_operations import CreateCollectionResult
from scratch.qdrant.schemas.collection_operations import DeleteCollectionResult


class QdrantClient:
    def __init__(self):
        self.client = ThirdPartyQdrantClient(url=QdrantConfig.url)

    def create_collection(
        self,
        collection_name: CollectionName,
        dense_vectors_config: VectorParams | dict[str, VectorParams] | None,
        sparse_vectors_config: dict[str, SparseVectorParams] | None,
    ) -> CreateCollectionResult:
        is_successful = self.client.create_collection(
            collection_name=collection_name,
            vectors_config=dense_vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

        return CreateCollectionResult(success=is_successful)

    def delete_collection(
        self, collection_name: CollectionName
    ) -> DeleteCollectionResult:
        is_successful = self.client.delete_collection(collection_name=collection_name)

        return DeleteCollectionResult(success=is_successful)

    def get_collection(self, collection_name: CollectionName) -> CollectionInfo:
        result = self.client.get_collection(collection_name=collection_name)
        return result

    def override_points(
        self, points: list[PointStruct], collection_name: CollectionName
    ) -> UpdateResult:
        result = self.client.upsert(points=points, collection_name=collection_name)
        return result

    def get_embedding_size(self, model_name: str) -> int:
        """Get the embedding size for a dense model."""
        return self.client.get_embedding_size(model_name)

    def query_points(
        self,
        collection_name: CollectionName,
        query: list[float] | None = None,
        prefetch: list[Prefetch] | None = None,
        query_filter: Filter | None = None,
        fusion_query: FusionQuery | None = None,
        using: str | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        limit: int = 10,
    ) -> QueryResponse:
        """Query points from a collection with optional prefetch and fusion."""
        return self.client.query_points(
            collection_name=collection_name,
            query=query if fusion_query is None else fusion_query,
            prefetch=prefetch,
            query_filter=query_filter,
            using=using,
            with_payload=with_payload,
            with_vectors=with_vectors,
            limit=limit,
        )

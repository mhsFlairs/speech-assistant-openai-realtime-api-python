"""
Qdrant RAG client for retrieving knowledge from vector database using LiteLLM embeddings.
"""
import asyncio
from typing import List, Optional

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    RAG_TOP_K,
    OPENAI_API_KEY,
    LITELLM_API_KEY,
)


class QdrantRAGClient:
    """
    Retrieval Augmented Generation (RAG) client using Qdrant vector database.
    
    Uses LiteLLM to generate embeddings with text-embedding-3-large model.
    """

    def __init__(self):
        """Initialize Qdrant client and embedding configuration."""
        # Import here to avoid dependency issues before installation
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
        
        # Use LiteLLM API key if set, otherwise fall back to OpenAI API key
        self.api_key = LITELLM_API_KEY or OPENAI_API_KEY
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimension = 3072  # text-embedding-3-large dimension
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False
        )
        self.collection_name = QDRANT_COLLECTION

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text using LiteLLM.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            from litellm import embedding
        except ImportError:
            raise ImportError("litellm not installed. Run: pip install litellm")
        
        try:
            # Use asyncio to run blocking LiteLLM call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: embedding(
                    model="text-embedding-3-large",
                    input=text,
                    api_key=self.api_key
                )
            )
            # Extract embedding vector from response
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    async def search(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """
        Search for relevant documents in Qdrant collection.
        
        Args:
            query: User's question or search text
            top_k: Number of documents to retrieve (uses RAG_TOP_K if not specified)
            
        Returns:
            List of documents with metadata and content
        """
        if top_k is None:
            top_k = RAG_TOP_K
            
        try:
            # Generate embedding for the query
            query_embedding = await self.embed(query)
            
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Search in Qdrant
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    with_payload=True
                )
            )
            
            # Extract and format results
            documents = []
            for point in search_results:
                if point.payload:
                    documents.append({
                        'page_content': point.payload.get('page_content', ''),
                        'metadata': point.payload.get('metadata', {}),
                        'score': point.score
                    })
            
            return documents
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []

    def format_context(self, results: List[dict]) -> str:
        """
        Format search results into a context string for the AI.
        
        Args:
            results: List of documents from search()
            
        Returns:
            Formatted context string with document content
        """
        if not results:
            return ""
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            content = doc.get('page_content', '').strip()
            if content:
                context_parts.append(f"Source {i}:\n{content}")
        
        if not context_parts:
            return ""
        
        return "\n\n".join(context_parts)

    async def get_relevant_context(self, query: str) -> str:
        """
        Retrieve and format relevant context for a query.
        
        Args:
            query: User's question or search text
            
        Returns:
            Formatted context string ready for injection into AI instructions
        """
        results = await self.search(query)
        return self.format_context(results)

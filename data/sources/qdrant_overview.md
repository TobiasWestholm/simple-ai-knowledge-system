Qdrant is a vector database designed for semantic search and retrieval-augmented
generation workloads.

Dense embeddings generated from BGE-M3 are stored in Qdrant so the system can
retrieve semantically similar chunks even when the user does not use the exact
keywords found in the source documents.

Qdrant complements lexical search rather than replacing it. Dense retrieval is
good at semantic similarity, while lexical retrieval is strong when exact terms
matter.

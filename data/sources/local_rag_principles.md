Local RAG keeps document retrieval, embeddings, and generation inside the local
development environment.

The ingestion pipeline should be deterministic. Given the same sources, chunking
rules, and embedding model, the generated chunk identities and index artifacts
should stay stable across repeated runs.

Hybrid retrieval combines semantic vectors with lexical ranking such as BM25.
That combination often improves coverage because some questions depend on meaning
while others depend on exact vocabulary.

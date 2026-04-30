import json
from pathlib import Path
from zipfile import ZipFile

from ai_ks.config import Settings
from ai_ks.embeddings import RemoteBgeM3Embedder
from ai_ks.ingestion import (
    ChunkRecord,
    IngestionService,
    QdrantVectorStore,
    RawDocument,
    align_chunk_start,
    build_qdrant_point,
    choose_split_end,
    chunk_documents,
    estimate_point_payload_bytes,
    fetch_documents,
    load_sources,
)


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(len(text.split()))] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.vector_size: int | None = None
        self.upserted_count = 0

    def prepare_collection(self, vector_size: int, recreate: bool) -> None:
        self.vector_size = vector_size

    def upsert_chunks(self, chunks, vectors) -> None:  # type: ignore[no-untyped-def]
        self.upserted_count = len(chunks)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.upsert_batches: list[list[object]] = []

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upsert_batches.append(list(points))


def test_chunk_documents_is_deterministic() -> None:
    document = RawDocument(
        id="doc-1",
        title="Deterministic",
        source_uri="memory://doc-1",
        text=(
            "Hybrid retrieval improves recall. "
            "Deterministic chunking keeps repeated ingests stable. "
            "This makes tests and re-indexing much easier to reason about."
        ),
        tags=("rag",),
        metadata={},
    )

    first = chunk_documents([document], chunk_size=60, chunk_overlap=10)
    second = chunk_documents([document], chunk_size=60, chunk_overlap=10)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_choose_split_end_prefers_paragraph_breaks_over_later_sentence_or_space() -> None:
    text = (
        "Intro block with enough text to move beyond the lower bound.\n\n"
        "Second paragraph continues with extra words and a sentence ending. "
        "Trailing words push the target window forward."
    )

    target_end = text.index("Trailing")
    split_end = choose_split_end(text, start=0, target_end=target_end)

    assert text[:split_end].endswith("\n\n")


def test_choose_split_end_prefers_sentence_breaks_over_later_spaces() -> None:
    text = (
        "This opening sentence is long enough to pass the lower bound. "
        "Then more trailing words continue without another sentence ending nearby"
    )

    target_end = len(text) - 5
    split_end = choose_split_end(text, start=0, target_end=target_end)

    assert text[:split_end].endswith(".")


def test_align_chunk_start_rewinds_to_word_boundary_inside_overlap() -> None:
    text = "legislation in Poland but also fundamentalistic islamic terrorism."
    cursor = text.index("islation")

    aligned_cursor = align_chunk_start(text, cursor)

    assert aligned_cursor == text.index("legislation")


def test_chunk_documents_does_not_start_chunks_mid_word() -> None:
    repeated_sentence = (
        "legislation in Poland but also fundamentalistic islamic terrorism. " * 4
    ).strip()
    document = RawDocument(
        id="doc-2",
        title="Word Boundaries",
        source_uri="memory://doc-2",
        text=repeated_sentence,
        tags=("chunking",),
        metadata={},
    )

    chunks = chunk_documents([document], chunk_size=90, chunk_overlap=25)

    assert len(chunks) > 1
    assert all(
        not chunk.text.startswith(("egislation", "gislation", "islation"))
        for chunk in chunks
    )
    for chunk in chunks[1:]:
        start = int(chunk.metadata["char_start"])
        assert start > 0
        assert not repeated_sentence[start - 1].isalnum()


def test_ingestion_writes_manifest_and_bm25_artifacts(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    source_path = knowledge_dir / "intro.md"
    source_path.write_text(
        "FastAPI exposes APIs. Qdrant stores dense vectors. BM25 helps exact matching.",
        encoding="utf-8",
    )

    sources_config = tmp_path / "sources.yaml"
    sources_config.write_text(
        "\n".join(
            [
                "sources:",
                "  - id: knowledge",
                "    title: Knowledge",
                "    kind: directory",
                "    path: knowledge",
                "    include_extensions:",
                "      - .md",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(
        sources_path=sources_config,
        data_dir=tmp_path,
        index_dir=tmp_path / "index",
        sqlite_path=tmp_path / "logs" / "telemetry.db",
        log_jsonl_path=tmp_path / "logs" / "requests.jsonl",
    )
    store = FakeVectorStore()
    result = IngestionService(
        settings=settings,
        embedder=FakeEmbedder(),
        vector_store=store,
    ).run()

    manifest = json.loads((tmp_path / "index" / "index_manifest.json").read_text(encoding="utf-8"))
    bm25_docs = json.loads((tmp_path / "index" / "bm25_documents.json").read_text(encoding="utf-8"))

    assert result.document_count == 1
    assert result.chunk_count >= 1
    assert store.vector_size == 2
    assert store.upserted_count == result.chunk_count
    assert manifest["embed_model_id"] == "BAAI/bge-m3"
    assert bm25_docs[0]["tokens"]


def test_ingestion_service_defaults_to_remote_embedder() -> None:
    settings = Settings()
    service = IngestionService(settings=settings, vector_store=FakeVectorStore())

    assert isinstance(service.embedder, RemoteBgeM3Embedder)


def test_fetch_documents_reads_supported_directory_sources(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    note_path = knowledge_dir / "lecture_notes.md"
    note_path.write_text(
        "\n".join(
            [
                "FastAPI validates incoming data.",
                "Qdrant stores dense vectors for semantic search.",
            ]
        ),
        encoding="utf-8",
    )
    docx_path = knowledge_dir / "lesson_notes.docx"
    create_minimal_docx(
        docx_path,
        [
            "Sentence transformers create embeddings.",
            "Qdrant stores the resulting vectors.",
        ],
    )

    sources_config = tmp_path / "sources.yaml"
    sources_config.write_text(
        "\n".join(
            [
                "sources:",
                "  - id: kb",
                "    title: Knowledge",
                "    kind: directory",
                "    path: knowledge",
                "    include_extensions:",
                "      - .docx",
                "      - .md",
            ]
        ),
        encoding="utf-8",
    )

    documents = fetch_documents(
        sources=load_sources(sources_config),
        base_dir=sources_config.parent,
    )

    assert len(documents) == 2
    document_by_id = {document.id: document for document in documents}
    assert "FastAPI validates incoming data." in document_by_id["kb/lecture_notes.md"].text
    assert "Qdrant stores dense vectors" in document_by_id["kb/lecture_notes.md"].text
    assert "Sentence transformers create embeddings." in document_by_id["kb/lesson_notes.docx"].text
    assert "Qdrant stores the resulting vectors." in document_by_id["kb/lesson_notes.docx"].text


def test_load_sources_rejects_non_directory_entries(tmp_path: Path) -> None:
    sources_config = tmp_path / "sources.yaml"
    sources_config.write_text(
        "\n".join(
            [
                "sources:",
                "  - id: intro",
                "    title: Intro",
                "    kind: file",
                "    path: knowledge/intro.md",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_sources(sources_config)
    except ValueError as exc:
        assert str(exc) == "Only directory sources are supported."
    else:
        raise AssertionError("Expected load_sources to reject non-directory sources.")


def test_qdrant_vector_store_batches_upserts_by_payload_size() -> None:
    client = FakeQdrantClient()
    payload_limit = 700
    store = QdrantVectorStore(
        url="http://qdrant:6333",
        collection_name="knowledge_chunks",
        client=client,  # type: ignore[arg-type]
        max_upsert_payload_bytes=payload_limit,
        max_upsert_points=10,
    )
    chunks = [
        ChunkRecord(
            id=f"chunk-{index}",
            document_id="doc-1",
            title="FastAPI Notes",
            source_uri="knowledge/fastapi.md",
            chunk_index=index,
            text="FastAPI makes API development fast and explicit." * 2,
            tags=("fastapi",),
            metadata={"kind": "note", "index": index},
        )
        for index in range(3)
    ]
    vectors = [[0.1, 0.2, 0.3] for _ in chunks]

    store.upsert_chunks(chunks=chunks, vectors=vectors)

    assert len(client.upsert_batches) > 1
    assert sum(len(batch) for batch in client.upsert_batches) == len(chunks)
    assert all(
        sum(estimate_point_payload_bytes(point) for point in batch) <= payload_limit
        for batch in client.upsert_batches
    )


def test_estimate_point_payload_bytes_grows_with_payload_content() -> None:
    chunk = ChunkRecord(
        id="chunk-1",
        document_id="doc-1",
        title="Short",
        source_uri="knowledge/short.md",
        chunk_index=0,
        text="short text",
        tags=(),
        metadata={},
    )
    longer_chunk = ChunkRecord(
        id="chunk-2",
        document_id="doc-1",
        title="Long",
        source_uri="knowledge/long.md",
        chunk_index=1,
        text="longer text " * 50,
        tags=("tag",),
        metadata={"kind": "long"},
    )
    short_point_size = estimate_point_payload_bytes(build_qdrant_point(chunk, [0.1, 0.2]))
    long_point_size = estimate_point_payload_bytes(build_qdrant_point(longer_chunk, [0.1, 0.2]))

    assert long_point_size > short_point_size


def create_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    paragraph_xml = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraph_xml}</w:body>"
        "</w:document>"
    )

    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

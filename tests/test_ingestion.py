import json
from pathlib import Path
from zipfile import ZipFile

from ai_ks.config import Settings
from ai_ks.ingestion import (
    IngestionService,
    RawDocument,
    chunk_documents,
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


def test_ingestion_writes_manifest_and_bm25_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    source_path = source_dir / "intro.md"
    source_path.write_text(
        "FastAPI exposes APIs. Qdrant stores dense vectors. BM25 helps exact matching.",
        encoding="utf-8",
    )

    sources_config = tmp_path / "sources.yaml"
    sources_config.write_text(
        "\n".join(
            [
                "sources:",
                "  - id: intro",
                "    title: Intro",
                "    kind: file",
                "    path: docs/intro.md",
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


def test_fetch_documents_reads_docx_directory_source(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    docx_path = knowledge_dir / "lecture_notes.docx"
    create_minimal_docx(
        docx_path,
        [
            "FastAPI validates incoming data.",
            "Qdrant stores dense vectors for semantic search.",
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
            ]
        ),
        encoding="utf-8",
    )

    documents = fetch_documents(
        sources=load_sources(sources_config),
        base_dir=sources_config.parent,
    )

    assert len(documents) == 1
    assert documents[0].id == "kb/lecture_notes.docx"
    assert "FastAPI validates incoming data." in documents[0].text
    assert "Qdrant stores dense vectors" in documents[0].text


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

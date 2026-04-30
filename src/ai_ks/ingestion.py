from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict
from xml.etree import ElementTree
from zipfile import ZipFile

import yaml  # type: ignore[import-untyped]
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from ai_ks.config import Settings
from ai_ks.embeddings import RemoteBgeM3Embedder


@dataclass(frozen=True)
class SourceDefinition:
    id: str
    title: str
    kind: Literal["directory"]
    location: str
    tags: tuple[str, ...]
    include_extensions: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RawDocument:
    id: str
    title: str
    source_uri: str
    text: str
    tags: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    title: str
    source_uri: str
    chunk_index: int
    text: str
    tags: tuple[str, ...]
    metadata: dict[str, Any]


class SplitChunk(TypedDict):
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class IngestionResult:
    source_count: int
    document_count: int
    chunk_count: int
    vector_size: int
    collection_name: str
    bm25_artifact_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    def prepare_collection(self, vector_size: int, recreate: bool) -> None:
        ...

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        ...


class BgeM3Embedder:
    REQUIRED_MODEL_PATTERNS = [
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "1_Pooling/config.json",
        "sparse_linear.pt",
        "colbert_linear.pt",
    ]

    def __init__(
        self,
        model_id: str,
        cache_dir: Path,
        *,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self._model: SentenceTransformer | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            self._model = SentenceTransformer(
                str(self._ensure_local_model()),
                device=self.device,
            )

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def resolved_device(self) -> str:
        if self._model is None:
            self.embed_texts(["device probe"])
        assert self._model is not None
        return str(self._model.device)

    def _ensure_local_model(self) -> Path:
        model_dir = self.cache_dir / self.model_id.replace("/", "--")
        if self._is_complete_model_dir(model_dir):
            return model_dir

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        hub_home = self.cache_dir / "hf_home"
        hub_cache = hub_home / "hub"
        xet_cache = hub_home / "xet"
        os.environ.setdefault("HF_HOME", str(hub_home))
        os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
        os.environ.setdefault("HF_XET_CACHE", str(xet_cache))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        snapshot_download(
            repo_id=self.model_id,
            cache_dir=str(hub_cache),
            local_dir=str(model_dir),
            allow_patterns=self.REQUIRED_MODEL_PATTERNS,
            force_download=True,
            max_workers=1,
        )
        return model_dir

    def _is_complete_model_dir(self, model_dir: Path) -> bool:
        if not model_dir.exists():
            return False

        required_paths = [
            model_dir / "modules.json",
            model_dir / "config_sentence_transformers.json",
        ]
        if not all(path.exists() for path in required_paths):
            return False

        weight_candidates = [
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
            model_dir / "0_Transformer" / "model.safetensors",
            model_dir / "0_Transformer" / "pytorch_model.bin",
        ]
        return any(path.exists() for path in weight_candidates)


ALLOWED_SOURCE_EXTENSIONS = (".docx", ".md", ".txt")


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        *,
        client: QdrantClient | None = None,
        max_upsert_payload_bytes: int = 24 * 1024 * 1024,
        max_upsert_points: int = 128,
    ) -> None:
        self.client = client or build_qdrant_client(url)
        self.collection_name = collection_name
        self.max_upsert_payload_bytes = max_upsert_payload_bytes
        self.max_upsert_points = max_upsert_points

    def prepare_collection(self, vector_size: int, recreate: bool) -> None:
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        batch: list[models.PointStruct] = []
        batch_size_bytes = 0

        for chunk, vector in zip(chunks, vectors, strict=True):
            point = build_qdrant_point(chunk, vector)
            point_size_bytes = estimate_point_payload_bytes(point)

            if (
                batch
                and (
                    len(batch) >= self.max_upsert_points
                    or batch_size_bytes + point_size_bytes > self.max_upsert_payload_bytes
                )
            ):
                self.client.upsert(collection_name=self.collection_name, points=batch)
                batch = []
                batch_size_bytes = 0

            batch.append(point)
            batch_size_bytes += point_size_bytes

        if batch:
            self.client.upsert(collection_name=self.collection_name, points=batch)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or RemoteBgeM3Embedder(
            base_url=settings.embedding_url,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        self.vector_store = vector_store or QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            max_upsert_payload_bytes=settings.qdrant_upsert_max_payload_bytes,
            max_upsert_points=settings.qdrant_upsert_max_points,
        )

    def run(self, recreate_collection: bool = True) -> IngestionResult:
        sources = load_sources(self.settings.sources_path)
        documents = fetch_documents(
            sources,
            base_dir=self.settings.sources_path.parent,
        )
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks were generated from the configured sources.")

        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match chunk count.")

        vector_size = len(vectors[0])
        self.vector_store.prepare_collection(vector_size=vector_size, recreate=recreate_collection)
        self.vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

        bm25_artifact_path = self.settings.index_dir / "bm25_documents.json"
        manifest_path = self.settings.index_dir / "index_manifest.json"
        bm25_artifact = build_bm25_artifact(chunks)

        write_json(
            bm25_artifact_path,
            bm25_artifact,
        )
        write_json(
            manifest_path,
            build_manifest(
                settings=self.settings,
                sources=sources,
                documents=documents,
                chunks=chunks,
                vector_size=vector_size,
            ),
        )

        return IngestionResult(
            source_count=len(sources),
            document_count=len(documents),
            chunk_count=len(chunks),
            vector_size=vector_size,
            collection_name=self.settings.qdrant_collection,
            bm25_artifact_path=str(bm25_artifact_path),
            manifest_path=str(manifest_path),
        )


def load_sources(path: Path) -> list[SourceDefinition]:
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_sources = raw_config.get("sources", [])
    sources: list[SourceDefinition] = []

    for entry in raw_sources:
        kind = entry["kind"]
        if kind != "directory":
            raise ValueError("Only directory sources are supported.")

        sources.append(
            SourceDefinition(
                id=entry["id"],
                title=entry.get("title", entry["id"]),
                kind=kind,
                location=entry["path"],
                tags=tuple(entry.get("tags", [])),
                include_extensions=tuple(
                    extension.lower() for extension in entry.get("include_extensions", [])
                ),
                metadata=dict(entry.get("metadata", {})),
            )
        )

    return sorted(sources, key=lambda source: source.id)


def fetch_documents(sources: list[SourceDefinition], base_dir: Path) -> list[RawDocument]:
    documents: list[RawDocument] = []

    for source in sources:
        documents.extend(fetch_directory_documents(source, base_dir))

    return documents


def clean_source_text(raw_text: str) -> str:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]

    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(line)
        previous_blank = False

    return "\n".join(compact_lines).strip()


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []

    for document in documents:
        for index, piece in enumerate(split_text(document.text, chunk_size, chunk_overlap)):
            chunk_id = stable_chunk_id(document.id, index, piece["text"])
            chunk_metadata = {
                **document.metadata,
                "document_id": document.id,
                "chunk_index": index,
                "char_start": piece["start"],
                "char_end": piece["end"],
            }
            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document.id,
                    title=document.title,
                    source_uri=document.source_uri,
                    chunk_index=index,
                    text=piece["text"],
                    tags=document.tags,
                    metadata=chunk_metadata,
                )
            )

    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[SplitChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[SplitChunk] = []
    cursor = 0
    normalized_text = text.strip()

    while cursor < len(normalized_text):
        target_end = min(len(normalized_text), cursor + chunk_size)
        end = choose_split_end(normalized_text, cursor, target_end)
        chunk_text = normalized_text[cursor:end].strip()

        if chunk_text:
            chunks.append({"start": cursor, "end": end, "text": chunk_text})

        if end >= len(normalized_text):
            break

        next_cursor = max(end - chunk_overlap, cursor + 1)
        next_cursor = align_chunk_start(normalized_text, next_cursor)
        if next_cursor <= cursor:
            next_cursor = end
        cursor = next_cursor

    return chunks


def choose_split_end(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    lower_bound = start + max(1, (target_end - start) // 2)
    prioritized_breakpoints = (
        ("\n\n", 2),
        (". ", 1),
        (" ", 1),
    )
    search_ranges = (
        (lower_bound, target_end),
        (start, target_end),
    )

    for marker, offset in prioritized_breakpoints:
        for range_start, range_end in search_ranges:
            split_at = text.rfind(marker, range_start, range_end)
            if split_at != -1:
                return split_at + offset

    return target_end


def align_chunk_start(text: str, cursor: int) -> int:
    if cursor <= 0:
        return 0
    if cursor >= len(text):
        return len(text)

    if text[cursor].isspace():
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    if text[cursor - 1].isalnum() and text[cursor].isalnum():
        while cursor > 0 and text[cursor - 1].isalnum():
            cursor -= 1
        return cursor

    return cursor


def build_bm25_artifact(chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "source_uri": chunk.source_uri,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "tokens": tokenize_for_bm25(chunk.text),
            "tags": list(chunk.tags),
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def build_manifest(
    settings: Settings,
    sources: list[SourceDefinition],
    documents: list[RawDocument],
    chunks: list[ChunkRecord],
    vector_size: int,
) -> dict[str, Any]:
    return {
        "embed_model_id": settings.embed_model_id,
        "qdrant_collection": settings.qdrant_collection,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "source_count": len(sources),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "vector_size": vector_size,
        "source_ids": [source.id for source in sources],
    }


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = sha256(f"{document_id}:{chunk_index}:{text}".encode()).hexdigest()
    return str(uuid.UUID(hex=digest[:32]))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def estimate_point_payload_bytes(point: models.PointStruct) -> int:
    payload = {
        "id": str(point.id),
        "vector": point.vector,
        "payload": point.payload,
    }
    return len(json.dumps(payload, ensure_ascii=True).encode("utf-8"))


def build_qdrant_point(chunk: ChunkRecord, vector: list[float]) -> models.PointStruct:
    return models.PointStruct(
        id=chunk.id,
        vector=vector,
        payload={
            "document_id": chunk.document_id,
            "title": chunk.title,
            "source_uri": chunk.source_uri,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "tags": list(chunk.tags),
            "metadata": chunk.metadata,
        },
    )


def build_qdrant_client(location: str) -> QdrantClient:
    return QdrantClient(url=location)


def fetch_directory_documents(source: SourceDefinition, base_dir: Path) -> list[RawDocument]:
    directory_path = (base_dir / source.location).resolve()
    include_extensions = source.include_extensions or ALLOWED_SOURCE_EXTENSIONS
    unsupported_extensions = set(include_extensions) - set(ALLOWED_SOURCE_EXTENSIONS)
    if unsupported_extensions:
        raise ValueError(
            "Only text file extensions are supported: "
            + ", ".join(sorted(ALLOWED_SOURCE_EXTENSIONS))
        )

    documents: list[RawDocument] = []

    for file_path in sorted(path for path in directory_path.rglob("*") if path.is_file()):
        if file_path.suffix.lower() not in include_extensions:
            continue

        relative_path = file_path.relative_to(directory_path)
        cleaned_text = clean_source_text(read_text_file(file_path))
        document_id = f"{source.id}/{relative_path.as_posix()}"
        document_title = file_path.stem.replace("_", " ")
        document_metadata = {
            **source.metadata,
            "directory_source_id": source.id,
            "relative_path": relative_path.as_posix(),
        }
        documents.append(
            RawDocument(
                id=document_id,
                title=document_title,
                source_uri=str(file_path),
                text=cleaned_text,
                tags=source.tags,
                metadata=document_metadata,
            )
        )

    return documents


def read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return extract_docx_text(path)
    return path.read_text(encoding="utf-8")


WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        text_parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{WORD_NAMESPACE['w']}}}t":
                text_parts.append(node.text or "")
            elif node.tag == f"{{{WORD_NAMESPACE['w']}}}tab":
                text_parts.append("\t")
            elif node.tag in {
                f"{{{WORD_NAMESPACE['w']}}}br",
                f"{{{WORD_NAMESPACE['w']}}}cr",
            }:
                text_parts.append("\n")

        paragraph_text = "".join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n\n".join(paragraphs)

import pathlib
from typing import cast

import chromadb
from chromadb.utils import embedding_functions

from src.schema import AetherisSchema, TableSchema


class SchemaRetriever:
    """Handles indexing and searching of table schemas using ChromaDB."""

    def __init__(
        self,
        schema: AetherisSchema,
        persist_directory: str = "data/chroma",
        collection_name: str = "home_schema",
    ):
        self.schema = schema
        pathlib.Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=cast("chromadb.EmbeddingFunction", self.emb_fn),
        )

        if self.collection.count() == 0:
            self._index_tables()

    def _index_tables(self) -> None:
        """Indexes all tables from the Pydantic schema into ChromaDB."""
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for table in self.schema.tables:
            columns_str = ", ".join([c.name for c in table.columns])
            doc = f"Table: {table.name}. Description: {table.description}. Columns: {columns_str}"

            documents.append(doc)
            metadatas.append({"name": table.name})
            ids.append(table.name)

        self.collection.add(
            documents=documents,
            metadatas=metadatas,  # pyright: ignore[reportArgumentType]
            ids=ids,
        )

    def get_relevant_tables(self, query: str, top_k: int = 2) -> list[TableSchema]:
        """Returns the most relevant table schemas for a given NL query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        relevant_tables: list[TableSchema] = []

        if results["metadatas"]:
            for metadata_list in results["metadatas"]:
                for metadata in metadata_list:
                    table_name = str(metadata.get("name", ""))
                    table = self.schema.get_table(table_name)
                    if table:
                        relevant_tables.append(table)

        return relevant_tables

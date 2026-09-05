import os
import logging
from typing import List
from embedding.embedder import embed_texts

logger = logging.getLogger(__name__)

def store_node_embeddings(nodes: List[dict], source_hash: str = "") -> None:
    """Persist embeddings for a list of graph nodes into the PGVECTOR table.

    Each node dictionary is expected to contain at least an 'id' field and
    optionally a 'type' field. The function embeds the node 'id' and upserts
    the vectors into 'node_embeddings'.
    """
    if not nodes:
        return

    # 1️⃣ Build the list of texts to embed – currently we embed the node id.
    texts = [node["id"] for node in nodes]
    embeddings = embed_texts(texts)  # → List[List[float]]

    # 2️⃣ Prepare rows for bulk insert.
    records = []
    for node, emb in zip(nodes, embeddings):
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        entity_type = node.get("type", "Entity")
        records.append((node_id, entity_type, emb, source_hash))

    if not records:
        return

    # 3️⃣ Connect via tools.init_db._connect_db or fallback
    close_after = False
    try:
        from tools.init_db import _connect_db
        conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
        close_after = True
    except Exception:
        from .db import pg_connection
        conn = pg_connection()
        close_after = False

    # 4️⃣ Upsert into PGVECTOR.
    sql = """
        INSERT INTO node_embeddings (node_id, entity_type, embedding, source_hash)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (node_id) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            entity_type = EXCLUDED.entity_type,
            source_hash = EXCLUDED.source_hash;
    """
    try:
        with conn.cursor() as cur:
            for rec in records:
                cur.execute(sql, rec)
        conn.commit()
        logger.info("Persisted %d node embeddings to PGVECTOR", len(records))
    except Exception as exc:
        logger.error("Failed to store embeddings in PGVECTOR: %s", exc)
        raise
    finally:
        if close_after and conn:
            conn.close()

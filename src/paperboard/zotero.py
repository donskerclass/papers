"""Read the local Zotero library (read-only) into plain records.

Runs on the laptop only. Nothing here is committed: the export lands in
``local/`` and only models derived from it are published.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

ZOTERO_DIR = Path.home() / "Zotero"
SKIP_TYPES = ("attachment", "note", "annotation")
ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)\s*([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})", re.I)
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>{}]+")


def _connect(db: Path) -> sqlite3.Connection:
    """Open a private snapshot, so a running Zotero is never touched or blocked."""
    tmp = Path(tempfile.mkdtemp(prefix="zotero-snap-"))
    for suffix in ("", "-wal"):
        src = Path(str(db) + suffix)
        if src.exists():
            shutil.copy2(src, tmp / (db.name + suffix))
    con = sqlite3.connect(tmp / db.name)
    con.row_factory = sqlite3.Row
    return con


def _first_page_text(key: str, limit: int = 4000) -> str:
    cache = ZOTERO_DIR / "storage" / key / ".zotero-ft-cache"
    if not cache.exists():
        return ""
    try:
        return cache.read_text(errors="ignore")[:limit]
    except OSError:
        return ""


def abstract_from_fulltext(text: str) -> str:
    """Best-effort abstract from PDF text: what follows 'Abstract', else the opening."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    m = re.search(r"\babstract\b[\s.:—-]*", text, re.I)
    start = m.end() if m and m.start() < 2500 else 0
    chunk = text[start:start + 1500]
    stop = re.search(r"\b(1\.?\s+Introduction|Introduction\b|Keywords?\b|JEL\b)", chunk)
    if stop and stop.start() > 300:
        chunk = chunk[: stop.start()]
    return chunk.strip()


def export(exclude_collections: tuple[str, ...] = ("To read at some point, maybe",)) -> list[dict]:
    """Every non-deleted top-level item except those filed only under excluded collections."""
    con = _connect(ZOTERO_DIR / "zotero.sqlite")
    q = lambda sql, *a: con.execute(sql, a).fetchall()  # noqa: E731

    items = q(
        f"""select i.itemID, i.key, t.typeName, i.dateAdded from items i join itemTypes t using(itemTypeID)
            where t.typeName not in ({','.join('?' * len(SKIP_TYPES))})
            and i.itemID not in (select itemID from deletedItems)""",
        *SKIP_TYPES,
    )
    fields: dict[int, dict[str, str]] = {}
    for r in q(
        """select d.itemID, f.fieldName, v.value from itemData d join fields f using(fieldID)
           join itemDataValues v using(valueID)"""
    ):
        fields.setdefault(r["itemID"], {})[r["fieldName"]] = r["value"]
    colls: dict[int, list[str]] = {}
    for r in q("select ci.itemID, c.collectionName from collectionItems ci join collections c using(collectionID)"):
        colls.setdefault(r["itemID"], []).append(r["collectionName"])
    attachments: dict[int, list[str]] = {}
    for r in q(
        """select a.parentItemID, i.key from itemAttachments a join items i using(itemID)
           where a.parentItemID is not null and a.contentType='application/pdf'"""
    ):
        attachments.setdefault(r["parentItemID"], []).append(r["key"])

    out = []
    for it in items:
        c = colls.get(it["itemID"], [])
        if c and all(x in exclude_collections for x in c):
            continue
        f = fields.get(it["itemID"], {})
        title = (f.get("title") or "").strip()
        if not title:
            continue
        blob = " ".join(f.get(k, "") for k in ("url", "extra", "archiveID", "DOI"))
        arxiv = ARXIV_RE.search(blob)
        doi = (f.get("DOI") or "").strip() or (m.group(0) if (m := DOI_RE.search(f.get("url", "") + " " + f.get("extra", ""))) else "")
        year = m.group(0) if (m := re.search(r"(19|20)\d\d", f.get("date", ""))) else ""
        fulltext = ""
        for key in attachments.get(it["itemID"], []):
            if fulltext := _first_page_text(key):
                break
        out.append(
            {
                "key": it["key"],
                "type": it["typeName"],
                "title": title,
                "abstract": (f.get("abstractNote") or "").strip(),
                "abstract_source": "zotero" if f.get("abstractNote") else "",
                "doi": doi.rstrip(".").lower(),
                "arxiv": arxiv.group(1) if arxiv else "",
                "year": year,
                "venue": f.get("publicationTitle", ""),
                "collections": sorted(c),
                "date_added": it["dateAdded"],
                "fulltext_head": fulltext,
            }
        )
    return out

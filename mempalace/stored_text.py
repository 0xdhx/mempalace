"""One reversible stored-text policy for every backend.

Postgres cannot store NUL in text, and a lone surrogate cannot be UTF-8
encoded. Chroma's FTS index is also corrupted by a raw NUL. The palace
stores those characters as a UTF-8-safe escape, flagged in metadata, and
returns the original characters on read. Ids are never rewritten: an id
that cannot be stored is rejected.

The escape is a prefix plus a backslash encoding. A second pass sees no
raw NUL or surrogate, so it does not escape again. A read decodes only
strings that still carry the prefix, so it does not decode twice.
"""

import re

FLAG = "_mempalace_escaped"
_PREFIX = "\x01mpesc:"
_SURROGATE = re.compile(r"[\ud800-\udfff]")


class StoredTextError(ValueError):
    """An id cannot be stored without changing it."""


def _bad(text: str) -> bool:
    return "\x00" in text or _SURROGATE.search(text) is not None


def _escape_body(text: str) -> str:
    parts = []
    for char in text:
        code = ord(char)
        if char == "\\":
            parts.append("\\\\")
        elif char == "\x00":
            parts.append("\\0")
        elif 0xD800 <= code <= 0xDFFF:
            parts.append(f"\\u{code:04X}")
        else:
            parts.append(char)
    return "".join(parts)


def _unescape_body(text: str) -> str:
    out = []
    index = 0
    size = len(text)
    while index < size:
        if text[index] != "\\":
            out.append(text[index])
            index += 1
            continue
        if index + 1 >= size:
            out.append("\\")
            break
        nxt = text[index + 1]
        if nxt == "\\":
            out.append("\\")
            index += 2
        elif nxt == "0":
            out.append("\x00")
            index += 2
        elif nxt == "u" and index + 6 <= size:
            try:
                code = int(text[index + 2 : index + 6], 16)
            except ValueError:
                out.append("\\")
                index += 1
                continue
            if 0xD800 <= code <= 0xDFFF:
                out.append(chr(code))
                index += 6
            else:
                out.append("\\")
                index += 1
        else:
            out.append("\\")
            index += 1
    return "".join(out)


def encode_text(text: str) -> str:
    """Escape NUL and lone surrogates. Other strings are unchanged."""
    if not isinstance(text, str) or not _bad(text):
        return text
    return _PREFIX + _escape_body(text)


def decode_text(text: str) -> str:
    """Reverse :func:`encode_text`. Strings without the prefix are unchanged."""
    if not isinstance(text, str) or not text.startswith(_PREFIX):
        return text
    return _unescape_body(text[len(_PREFIX) :])


def _walk(value, fn):
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {
            (fn(key) if isinstance(key, str) else key): _walk(item, fn)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_walk(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, fn) for item in value)
    return value


def reject_stored_id(doc_id: str) -> str:
    """Raise if ``doc_id`` contains a NUL or a lone surrogate."""
    if not isinstance(doc_id, str) or not doc_id:
        raise StoredTextError("id must be a non-empty string")
    if _bad(doc_id):
        raise StoredTextError(
            f"id {doc_id!r} cannot be stored: NUL and lone surrogates are rejected, not rewritten"
        )
    return doc_id


def prepare_stored_write(documents, ids, metadatas=None):
    """Escape drawer text and reject ids that cannot be stored.

    A bare string document stays a string (Chroma's one-or-many shape).
    Metadata is returned only when the caller passed it, with
    ``_mempalace_escaped`` set to 1 on any record that was escaped.
    """
    single = isinstance(documents, str)
    docs = [documents] if single else list(documents)
    id_list = list(ids)
    if len(docs) != len(id_list):
        raise StoredTextError("documents and ids must be the same length")
    supplied = metadatas is not None
    metas = list(metadatas) if supplied else [{} for _ in id_list]
    if len(metas) != len(id_list):
        raise StoredTextError("metadatas and ids must be the same length")
    out_docs = []
    out_metas = []
    for doc_id, doc, meta in zip(id_list, docs, metas):
        reject_stored_id(doc_id)
        meta = dict(meta or {})
        encoded_doc = encode_text(doc) if isinstance(doc, str) else doc
        encoded_meta = _walk(meta, encode_text)
        if encoded_doc != doc or encoded_meta != meta:
            encoded_meta[FLAG] = 1
        else:
            encoded_meta.pop(FLAG, None)
        out_docs.append(encoded_doc)
        out_metas.append(encoded_meta)
    if single:
        out_docs = out_docs[0]
    escaped = any(isinstance(meta, dict) and meta.get(FLAG) == 1 for meta in out_metas)
    if not supplied and not escaped:
        return out_docs, id_list, None
    return out_docs, id_list, out_metas


def restore_documents(documents):
    """Decode escaped documents. A bare string stays a string.

    Query results are a list of lists; each inner document is decoded.
    """
    if isinstance(documents, str):
        return decode_text(documents)
    if not documents:
        return documents
    return [
        decode_text(doc) if isinstance(doc, str) else restore_documents(doc) for doc in documents
    ]


def restore_metadatas(metadatas):
    """Decode escaped metadata strings. The flag is left in place.

    Query results are a list of lists of metadata dicts.
    """
    if not metadatas:
        return metadatas
    restored = []
    for meta in metadatas:
        if isinstance(meta, dict):
            restored.append(_walk(meta, decode_text))
        elif isinstance(meta, list):
            restored.append(restore_metadatas(meta))
        else:
            restored.append(meta)
    return restored

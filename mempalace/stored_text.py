"""One reversible stored-text policy for every backend.

Postgres cannot store NUL in text, a lone surrogate cannot be UTF-8 encoded,
and a raw NUL corrupts Chroma's FTS index. A string holding either, or one
that already starts with the escape prefix, is stored as the prefix plus a
backslash escape, and the record carries ``FLAG`` in its metadata. A read
decodes only flagged records, and in them only prefixed strings, so text
that merely looks escaped is returned as written. The flag never reaches
callers. Ids and metadata keys that cannot be stored are rejected, never
rewritten.

``FLAG`` is reserved. A record whose metadata carries ``FLAG: 1`` is taken
as stored form and written unchanged, so copying raw rows between palaces
(repair, migrate) stays verbatim; any other value is refused.

Where filters compare against the stored form: a filter value holding NUL or
a lone surrogate does not match the escaped value.
"""

import re

FLAG = "_mempalace_escaped"
_PREFIX = "\x01mpesc:"
_BAD = re.compile(r"[\x00\ud800-\udfff]")
_ESCAPE = re.compile(r"[\\\x00\ud800-\udfff]")
_UNESCAPE = re.compile(r"\\(\\|0|u[0-9A-F]{4})")


class StoredTextError(ValueError):
    """An id or metadata key cannot be stored without changing it."""


def _needs_escape(text: str) -> bool:
    return text.startswith(_PREFIX) or _BAD.search(text) is not None


def _escape_char(match) -> str:
    char = match.group()
    if char == "\\":
        return "\\\\"
    if char == "\x00":
        return "\\0"
    return f"\\u{ord(char):04X}"


def _unescape_seq(match) -> str:
    seq = match.group(1)
    if seq == "\\":
        return "\\"
    if seq == "0":
        return "\x00"
    return chr(int(seq[1:], 16))


def _encode(value, changed: list):
    if isinstance(value, str):
        if not _needs_escape(value):
            return value
        changed.append(True)
        return _PREFIX + _ESCAPE.sub(_escape_char, value)
    if isinstance(value, list):
        return [_encode(item, changed) for item in value]
    return value


def _decode(value):
    if isinstance(value, str):
        if not value.startswith(_PREFIX):
            return value
        return _UNESCAPE.sub(_unescape_seq, value[len(_PREFIX) :])
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _check_key(key) -> None:
    if isinstance(key, str) and _BAD.search(key):
        raise StoredTextError(
            f"{key!r} cannot be stored: NUL and lone surrogates are rejected, not rewritten"
        )


def encode_record(document, metadata):
    """Return ``(document, metadata)`` in stored form.

    ``metadata`` is ``None`` only when it was ``None`` and nothing needed
    escaping.
    """
    if metadata and FLAG in metadata:
        if metadata[FLAG] != 1:
            raise StoredTextError(f"metadata key {FLAG!r} is reserved")
        return document, metadata
    changed: list = []
    document = _encode(document, changed)
    encoded = {}
    for key, value in (metadata or {}).items():
        _check_key(key)
        encoded[key] = _encode(value, changed)
    if changed:
        encoded[FLAG] = 1
    elif metadata is None:
        return document, None
    return document, encoded


def decode_record(document, metadata):
    """Reverse :func:`encode_record` and drop the flag."""
    if not metadata or FLAG not in metadata:
        return document, metadata
    metadata = dict(metadata)
    if not metadata.pop(FLAG):
        return document, metadata
    return _decode(document), {key: _decode(value) for key, value in metadata.items()}


def prepare_stored_write(documents, ids, metadatas=None):
    """Encode a write batch and reject ids that cannot be stored.

    ``documents`` may be ``None`` (a metadata-only update) or a bare string
    (Chroma's one-or-many shape); either comes back in the same shape.
    Metadata comes back as ``None`` only when none was passed and nothing was
    escaped. Otherwise every record gets a dict, and a record with nothing of
    its own gets ``{FLAG: 1}``: a spare flag is harmless, since only prefixed
    strings are decoded, and merging backends keep older escaped values
    readable.
    """
    single = isinstance(documents, str)
    ids = [ids] if isinstance(ids, str) else list(ids)
    docs = [documents] if single else documents
    for doc_id in ids:
        if not isinstance(doc_id, str) or not doc_id:
            raise StoredTextError("id must be a non-empty string")
        _check_key(doc_id)
    metas = metadatas if metadatas is not None else [None] * len(ids)
    out_docs, out_metas = [], []
    for doc, meta in zip(docs if docs is not None else [None] * len(ids), metas):
        doc, meta = encode_record(doc, meta)
        out_docs.append(doc)
        out_metas.append(meta)
    if metadatas is None:
        if all(meta is None for meta in out_metas):
            out_metas = None
        else:
            out_metas = [meta if meta is not None else {FLAG: 1} for meta in out_metas]
    if documents is None:
        out_docs = None
    elif single:
        out_docs = out_docs[0]
    return out_docs, ids, out_metas


def restore_rows(documents, metadatas):
    """Decode parallel ``documents``/``metadatas`` lists, nested one level for
    query results. Either list may be empty when it was not requested."""
    if metadatas and isinstance(metadatas[0], list):
        pairs = [
            restore_rows(docs, metas)
            for docs, metas in zip(documents or [[]] * len(metadatas), metadatas)
        ]
        return (
            [docs for docs, _ in pairs] if documents else documents,
            [metas for _, metas in pairs],
        )
    if not metadatas:
        return documents, metadatas
    docs = documents if documents else [None] * len(metadatas)
    pairs = [decode_record(doc, meta) for doc, meta in zip(docs, metadatas)]
    return (
        [doc for doc, _ in pairs] if documents else documents,
        [meta for _, meta in pairs],
    )

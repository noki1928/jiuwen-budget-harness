"""
Knowledge base of markdown files as a graph: entity -> {field: value}.

  "# Sergei Garcia"           -> entity "Sergei Garcia"
  "Employer: Fjord Data"      -> field Employer; the value links to the entity
  "- **Born:** 1985"            "Fjord Data" if such a title exists

The model never reads files. It either writes a path query
("Sergei Garcia > Employer > Headquarters") that the wrapper follows, or it
gets a small subgraph around the entities named in the question.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

FIELD_RE = re.compile(r"^[-*+\s]*([A-Za-z][\w /()-]{0,40}?)\s*:\s*(.+?)\s*$")
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:[ -][A-Z][a-z]+)+)\b")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class KB:
    def __init__(self, ents: dict[str, dict[str, str]], kinds: dict[str, str]):
        self.ents = ents
        self.kinds = kinds
        self.index = {norm(t): t for t in ents}

    # ---------------------------------------------------------------- lookup
    def entity(self, text: str, fuzzy: bool = True) -> str | None:
        n = norm(text)
        if n in self.index:
            return self.index[n]
        for k in sorted(self.index, key=len, reverse=True):  # "Sergei Garcia's employer"
            if re.search(rf"\b{re.escape(k)}\b", n):
                return self.index[k]
        if fuzzy:
            m = difflib.get_close_matches(n, list(self.index), n=1, cutoff=0.85)
            if m:
                return self.index[m[0]]
        return None

    def link(self, value: str) -> str | None:
        return self.index.get(norm(value))

    def field(self, ent: str, name: str) -> str | None:
        fields = self.ents[ent]
        by_norm = {norm(k): k for k in fields}
        n = norm(name)
        if n in by_norm:
            return by_norm[n]
        m = difflib.get_close_matches(n, list(by_norm), n=1, cutoff=0.7)
        if m:
            return by_norm[m[0]]
        for k in by_norm:  # "headquarters city" -> "headquarters"
            if k in n or n in k:
                return by_norm[k]
        return None

    # ---------------------------------------------------------------- question
    def mentioned(self, question: str) -> list[str]:
        n = norm(question)
        found, taken = [], n
        for k in sorted(self.index, key=len, reverse=True):
            if re.search(rf"\b{re.escape(k)}\b", taken):
                found.append(self.index[k])
                taken = re.sub(rf"\b{re.escape(k)}\b", " ", taken)
        return found

    def unknown_names(self, question: str) -> list[str]:
        """Capitalized multi-word names in the question that match no entity."""
        out = []
        for name in NAME_RE.findall(question):
            words = name.split()
            if words[0] in ("The", "In", "Which", "What", "Who", "Where", "When", "How"):
                name = " ".join(words[1:])
                if len(words) < 3:
                    continue
            if self.entity(name, fuzzy=False) is None:
                out.append(name)
        return out

    # ---------------------------------------------------------------- views
    def schema(self) -> str:
        """Field names per entity kind; '->' marks fields whose values are entities."""
        per_kind: dict[str, dict[str, bool]] = {}
        for e, fields in self.ents.items():
            d = per_kind.setdefault(self.kinds.get(e, "entity"), {})
            for k, v in fields.items():
                d[k] = d.get(k, False) or self.link(v) is not None
        return "\n".join(f"{kind}: " + ", ".join(f"{k}{' ->' if link else ''}" for k, link in fs.items())
                         for kind, fs in per_kind.items())

    def render(self, names) -> str:
        return "\n".join(f"{n}: " + "; ".join(f"{k}={v}" for k, v in self.ents[n].items()) for n in names)

    def subgraph(self, question: str, depth: int = 2, limit: int = 12) -> str:
        seen, frontier = [], self.mentioned(question)
        for _ in range(depth + 1):
            nxt = []
            for e in frontier:
                if e in seen or len(seen) >= limit:
                    continue
                seen.append(e)
                nxt += [l for v in self.ents[e].values() if (l := self.link(v)) and l not in seen]
            frontier = nxt
        return self.render(seen)

    def whole(self) -> str:
        return self.render(self.ents)

    # ---------------------------------------------------------------- path query
    def follow(self, reply: str, start: str | None = None) -> str:
        """Evaluate 'Entity > Field > Field'. Raises LookupError with a reason.

        `start` (the entity named in the question) is used when the model's path
        does not begin with a known entity, e.g. "person: Employer > Headquarters".
        """
        text = re.sub(r"```\w*", "", reply).strip()
        line = next((l for l in text.splitlines() if re.search(r">|->|→", l)), text.splitlines()[0] if text else "")
        line = re.sub(r"^(path|answer|query)\s*:\s*", "", line.strip(), flags=re.I)
        parts = [p.strip(" `'\"*.") for p in re.split(r"\s*(?:->|→|>)\s*", line) if p.strip(" `'\"*.")]
        parts = [re.sub(r"^(person|company|entity|\w+)\s*:\s*(?=\S)", "", p) if ":" in p else p for p in parts]
        if not parts:
            raise LookupError("empty path")
        if parts[0].upper() == "IMPOSSIBLE":
            raise LookupError("IMPOSSIBLE")
        cur = self.entity(parts[0])
        if cur is None and start is not None:
            cur, parts = start, [start] + parts
        if cur is None:
            raise LookupError(f"no entity named {parts[0]!r}")
        for i, f in enumerate(parts[1:]):
            k = self.field(cur, f)
            if k is None:
                raise LookupError(f"{cur} has no field {f!r}; fields: {', '.join(self.ents[cur])}")
            val = self.ents[cur][k]
            if i == len(parts) - 2:
                return val
            nxt = self.link(val)
            if nxt is None:
                raise LookupError(f"{cur}.{k} = {val!r} is not an entity")
            cur = nxt
        return cur  # a bare entity name was asked for


def _fields(lines: list[str]) -> dict[str, str]:
    fields = {}
    for l in lines:
        m = FIELD_RE.match(l.replace("*", ""))
        if m and not l.startswith("#"):
            fields[m.group(1).strip()] = m.group(2).strip()
    return fields


def _sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """'## Name' sections of a page that holds many entities (a wiki / directory page)."""
    out, title, body = [], None, []
    for l in lines:
        m = re.match(r"^#{2,4}\s+(.+?)\s*#*$", l)
        if m:
            if title is not None:
                out.append((title, body))
            title, body = m.group(1).strip(), []
        elif title is not None:
            body.append(l)
    if title is not None:
        out.append((title, body))
    return out


def load(ws: Path) -> KB | None:
    """Parse markdown/text files; None when the files do not look like a field-based kb.

    One file per entity ("# Title" + fields), or several entities per file as
    "## Title" sections with their own fields.
    """
    files = sorted(p for p in ws.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt", ".markdown"))
    if not files:
        return None
    ents: dict[str, dict[str, str]] = {}
    kinds: dict[str, str] = {}
    structured = 0
    for p in files:
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        kind = p.stem.split("_", 1)[0] if "_" in p.stem else "entity"
        secs = [(t, _fields(b)) for t, b in _sections(lines)]
        secs = [(t, f) for t, f in secs if f]
        if len(secs) >= 2:
            structured += 1
            for t, f in secs:
                ents[t] = f
                kinds[t] = kind
            continue
        title = next((l.lstrip("#").strip() for l in lines if l.startswith("#")), "")
        if not title:
            stem = p.stem.split("_", 1)[-1] if "_" in p.stem else p.stem
            title = stem.replace("_", " ").title()
        fields = _fields(lines)
        if fields:
            structured += 1
        ents[title] = fields
        kinds[title] = kind
    if structured < 0.6 * len(files):
        return None
    return KB(ents, kinds)

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.models import GuidanceCitation, ScenarioType


GUIDANCE_DIR = Path(__file__).resolve().parent.parent / "data" / "guidance"
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GuidanceDoc:
    title: str
    source: str
    scenario: str
    body: str


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if len(token) > 2}


def load_guidance_docs(guidance_dir: Path = GUIDANCE_DIR) -> list[GuidanceDoc]:
    docs: list[GuidanceDoc] = []
    for path in sorted(guidance_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ").title()
        scenario = "combined"
        body_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.lower().startswith("scenario:"):
                scenario = line.split(":", 1)[1].strip().lower()
            else:
                body_lines.append(line)
        docs.append(
            GuidanceDoc(
                title=title,
                source=str(path.relative_to(guidance_dir.parent.parent)),
                scenario=scenario,
                body="\n".join(body_lines).strip(),
            )
        )
    return docs


def retrieve_guidance(
    query: str,
    scenario_type: ScenarioType,
    limit: int = 3,
    docs: list[GuidanceDoc] | None = None,
) -> list[GuidanceCitation]:
    docs = docs if docs is not None else load_guidance_docs()
    query_tokens = _tokens(f"{scenario_type} {query}")
    scored: list[tuple[float, GuidanceDoc]] = []
    for doc in docs:
        doc_tokens = _tokens(f"{doc.title} {doc.scenario} {doc.body}")
        overlap = len(query_tokens & doc_tokens)
        scenario_bonus = 2 if doc.scenario in {scenario_type, "combined"} else 0
        score = overlap + scenario_bonus
        if score > 0:
            scored.append((float(score), doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        GuidanceCitation(
            title=doc.title,
            source=doc.source,
            snippet=_best_snippet(doc.body, query_tokens),
            score=score,
        )
        for score, doc in scored[:limit]
    ]


def _best_snippet(body: str, query_tokens: set[str], max_len: int = 260) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        return ""
    best = max(paragraphs, key=lambda p: len(_tokens(p) & query_tokens))
    if len(best) <= max_len:
        return best
    return best[: max_len - 3].rstrip() + "..."


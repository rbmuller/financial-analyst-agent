"""Turn an agent result into Slack Block Kit blocks.

Slack does not render Markdown. Its own dialect ("mrkdwn") uses *single
asterisks* for bold and <url|label> for links, has no headings and no tables,
and needs &, < and > escaped. Messages are a list of blocks, each with its own
size limit, so a long answer is split across several section blocks.
"""

import re
from typing import Any

from agent.loop import AgentResult
from agent.sources import Source
from agent.tracing import RunStatus, Step, StepType

SECTION_TEXT_LIMIT = 2900  # Slack rejects section text over 3000 characters
MAX_BLOCKS = 50
FALLBACK_TEXT_LIMIT = 3000
SOURCE_LABEL_LIMIT = 220

TEXT_THINKING = ":hourglass_flowing_sand: Pensando…"
TEXT_SOURCES_HEADER = "*Fontes*"
TEXT_UNKNOWN_CITATIONS = (
    ":warning: A resposta cita fontes que nenhuma consulta retornou: {ids}. "
    "Trate essas afirmações como não verificadas."
)
TEXT_ERROR = (
    ":x: Não consegui responder por um erro interno. O erro foi registrado em `agent_runs`."
)
PROGRESS_BY_TOOL = {
    "search_filings": ":mag: Buscando nos relatórios da SEC…",
    "query_financials": ":bar_chart: Consultando os números…",
    "calculator": ":abacus: Calculando…",
}
PROGRESS_WRITING = ":writing_hand: Analisando o que encontrei…"
STATUS_LABELS = {
    RunStatus.STEP_LIMIT: ":warning: limite de passos atingido",
    RunStatus.TRUNCATED: ":warning: resposta cortada",
    RunStatus.REFUSED: ":warning: o modelo recusou a pergunta",
}

BOLD = re.compile(r"\*\*(.+?)\*\*")
HEADING = re.compile(r"^#{1,6}\s+(.*)$")
BULLET = re.compile(r"^(\s*)[-*]\s+")
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
TABLE_SEPARATOR = re.compile(r"^\|[\s:|-]+\|$")
CODE_FENCE = "```"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _convert_line(line: str) -> str:
    line = escape(line)
    heading = HEADING.match(line)
    if heading:
        return f"*{BOLD.sub(r'\1', heading.group(1))}*"
    line = BULLET.sub(r"\1• ", line)
    line = BOLD.sub(r"*\1*", line)
    return LINK.sub(r"<\2|\1>", line)


def to_mrkdwn(markdown: str) -> str:
    """Convert the Markdown the model writes into Slack mrkdwn.

    Tables cannot be rendered, so a table becomes a monospace block that at
    least keeps its rows readable.
    """
    converted: list[str] = []
    table: list[str] = []

    def flush_table() -> None:
        if table:
            converted.extend([CODE_FENCE, *table, CODE_FENCE])
            table.clear()

    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not TABLE_SEPARATOR.match(stripped):
                table.append(BOLD.sub(r"\1", stripped))
            continue
        flush_table()
        converted.append(_convert_line(line))
    flush_table()
    return "\n".join(converted)


def split_text(text: str, limit: int = SECTION_TEXT_LIMIT) -> list[str]:
    """Split on paragraph, then line boundaries so every piece fits one section block."""
    pieces: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single line longer than a block
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        pieces.append(current)
    return pieces


def _section(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _context(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def _source_line(source: Source) -> str:
    label = escape(source.label[:SOURCE_LABEL_LIMIT])
    linked = f"<{source.url}|{label}>" if source.url else label
    return f"• `{source.id}` {linked}"


def _footer(result: AgentResult) -> str:
    parts = [f"{result.latency_ms / 1000:.0f}s", f"{result.steps} passos"]
    if result.cost_usd is not None:
        parts.append(f"US$ {result.cost_usd:.2f}")
    if result.status in STATUS_LABELS:
        parts.append(STATUS_LABELS[result.status])
    parts.append(f"run `{result.run_id[:8]}`")
    return " · ".join(parts)


def result_blocks(result: AgentResult) -> list[dict[str, Any]]:
    blocks = [_section(piece) for piece in split_text(to_mrkdwn(result.answer))]
    if result.unknown_citations:
        blocks.append(
            _context(TEXT_UNKNOWN_CITATIONS.format(ids=", ".join(result.unknown_citations)))
        )
    if result.sources:
        source_lines = "\n".join(_source_line(source) for source in result.sources)
        blocks.append({"type": "divider"})
        blocks.extend(
            _section(piece) for piece in split_text(f"{TEXT_SOURCES_HEADER}\n{source_lines}")
        )
    blocks.append(_context(_footer(result)))
    return blocks[:MAX_BLOCKS]


def fallback_text(result: AgentResult) -> str:
    """Plain text for notifications and clients that do not render blocks."""
    return result.answer[:FALLBACK_TEXT_LIMIT]


def progress_text(step: Step) -> str:
    """One-line status shown while the agent works, based on the step just finished."""
    if step.type is StepType.TOOL and step.tool_name in PROGRESS_BY_TOOL:
        return PROGRESS_BY_TOOL[step.tool_name]
    return PROGRESS_WRITING

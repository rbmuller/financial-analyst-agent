"""Turn a 10-K / 10-Q HTML document into plain-text sections, one per Item.

Sections matter for retrieval: "what did management say about margins" lives in
the MD&A, and each chunk will carry its section as citation metadata.
"""

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

FORM_10K = "10-K"
FORM_10Q = "10-Q"
FULL_DOCUMENT = "Full document"

ITEM_TITLES_10K: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity and Related Stockholder Matters",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# 10-Q item numbers repeat across its two parts, so the part is in the key.
PART_I = "I"
PART_II = "II"
ITEM_TITLES_10Q: dict[tuple[str, str], str] = {
    (PART_I, "1"): "Financial Statements",
    (PART_I, "2"): "Management's Discussion and Analysis",
    (PART_I, "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    (PART_I, "4"): "Controls and Procedures",
    (PART_II, "1"): "Legal Proceedings",
    (PART_II, "1A"): "Risk Factors",
    (PART_II, "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    (PART_II, "3"): "Defaults Upon Senior Securities",
    (PART_II, "4"): "Mine Safety Disclosures",
    (PART_II, "5"): "Other Information",
    (PART_II, "6"): "Exhibits",
}

# Sections the agent depends on most; ingestion warns when one is missing.
KEY_ITEMS: dict[str, tuple[str, ...]] = {
    FORM_10K: ("Item 1A", "Item 7"),
    FORM_10Q: ("Part I, Item 2",),
}

BLOCK_TAGS = ["p", "div", "tr", "table", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
CELL_TAGS = ["td", "th"]
REMOVED_TAGS = ["script", "style", "head", "ix:header"]
HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)

# Page furniture repeated on every printed page: bare page numbers and the
# "Table of Contents" back-link. It would otherwise land in the middle of chunks.
PAGE_FURNITURE = re.compile(r"^(?:(?:page\s+)?\d{1,3}|table of contents)$", re.IGNORECASE)

PART_HEADING = re.compile(r"^PART\s+(I{1,3}|IV)\b", re.IGNORECASE)
ITEM_HEADING = re.compile(r"^ITEM\s+(\d{1,2}[A-C]?)\b", re.IGNORECASE)
MAX_HEADING_LENGTH = 160
# Table-of-contents entries match the same headings but carry almost no text.
MIN_SECTION_LENGTH = 400


@dataclass(frozen=True)
class Section:
    position: int
    item: str
    title: str
    content: str


def html_to_text(html: str) -> str:
    """Visible text of the document, one line per block element."""
    # Filings are inline XBRL (XHTML with an XML declaration). The lenient HTML
    # parser is used on purpose: it tolerates malformed markup in older filings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(REMOVED_TAGS):
        tag.decompose()
    for tag in soup.find_all(style=HIDDEN_STYLE):
        if not tag.decomposed:
            tag.decompose()

    for line_break in soup.find_all("br"):
        line_break.replace_with("\n")
    for cell in soup.find_all(CELL_TAGS):
        cell.append(" ")
    for block in soup.find_all(BLOCK_TAGS):
        block.insert_before("\n")
        block.append("\n")

    lines = (" ".join(line.split()) for line in soup.get_text().replace("\xa0", " ").split("\n"))
    return "\n".join(line for line in lines if line and not PAGE_FURNITURE.match(line))


def _heading_key(line: str, form: str, current_part: str | None) -> tuple[str, str] | None:
    """(item label, title) if the line is a known Item heading for this form."""
    if len(line) > MAX_HEADING_LENGTH:
        return None
    match = ITEM_HEADING.match(line)
    if match is None:
        return None
    number = match.group(1).upper()

    if form == FORM_10Q:
        if current_part is None or (current_part, number) not in ITEM_TITLES_10Q:
            return None
        return f"Part {current_part}, Item {number}", ITEM_TITLES_10Q[(current_part, number)]
    if number not in ITEM_TITLES_10K:
        return None
    return f"Item {number}", ITEM_TITLES_10K[number]


def split_sections(text: str, form: str) -> list[Section]:
    """Split document text into Item sections.

    Each heading opens a segment that runs until the next heading. The table of
    contents produces short duplicate segments, which are dropped in favour of
    the real ones. A document with no recognizable headings becomes one section.
    """
    segments: dict[tuple[str, str], list[list[str]]] = {}
    current_part: str | None = None
    current_lines: list[str] | None = None

    for line in text.split("\n"):
        part = PART_HEADING.match(line)
        if part is not None and len(line) <= MAX_HEADING_LENGTH:
            current_part = part.group(1).upper()
            current_lines = None
            continue
        key = _heading_key(line, form, current_part)
        if key is not None:
            current_lines = [line]
            segments.setdefault(key, []).append(current_lines)
        elif current_lines is not None:
            current_lines.append(line)

    sections: list[Section] = []
    for (item, title), item_segments in segments.items():
        texts = ["\n".join(lines) for lines in item_segments]
        substantial = [body for body in texts if len(body) >= MIN_SECTION_LENGTH]
        content = "\n".join(substantial) if substantial else max(texts, key=len)
        sections.append(Section(position=len(sections), item=item, title=title, content=content))

    if not sections:
        return [Section(position=0, item=FULL_DOCUMENT, title=form, content=text)]
    return sections


def missing_key_items(sections: list[Section], form: str) -> list[str]:
    found = {section.item for section in sections}
    return [item for item in KEY_ITEMS.get(form, ()) if item not in found]

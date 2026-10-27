"""Project 3 portable copy of the ORIGINAL Project 2 Extension JATS parser.

Unmodified parsing implementation extracted from
  02-query expansion models ext/src/e1_02_local_jats_corpus.py
Only the standalone parsing constants and helpers are included.
No runtime dependency on frozen Project 2 or its config; Project 2 untouched.

Original methods and return-document structure are preserved verbatim.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


EXCLUDED_ABSTRACT_TYPE_MARKERS = (
    "author-highlights",
    "highlight",
    "highlights",
    "precis",
    "graphical",
    "teaser",
    "plain-language",
    "plain language",
    "key-points",
    "key points",
    "publisher",
    "summary-for",
    "supplement",
)

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _text(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _article_id(article_meta, id_type):
    if article_meta is None:
        return ""

    wanted = str(id_type).casefold()
    for node in article_meta.findall("./{*}article-id"):
        if (node.attrib.get("pub-id-type") or "").casefold() == wanted:
            return _text(node)
    return ""


def _authors(article_meta):
    output = []
    if article_meta is None:
        return output

    for contrib in article_meta.findall(".//{*}contrib[@contrib-type='author']"):
        name = contrib.find("./{*}name")
        if name is None:
            collab = _text(contrib.find("./{*}collab"))
            if collab:
                output.append(collab)
            continue

        surname = _text(name.find("./{*}surname"))
        given = _text(name.find("./{*}given-names"))
        full = " ".join(part for part in (given, surname) if part).strip()
        if full:
            output.append(full)

    return output


def _affiliations(article_meta):
    if article_meta is None:
        return []

    seen = set()
    output = []
    for node in article_meta.findall(".//{*}aff"):
        value = _text(node)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _keywords(article_meta):
    if article_meta is None:
        return []

    output = []
    seen = set()
    for node in article_meta.findall(".//{*}kwd-group/{*}kwd"):
        value = _text(node)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _normalized_abstract_type(node):
    return (
        node.attrib.get("abstract-type")
        or node.attrib.get("content-type")
        or ""
    ).strip().casefold()


def _is_excluded_abstract(node):
    abstract_type = _normalized_abstract_type(node)
    return any(marker in abstract_type for marker in EXCLUDED_ABSTRACT_TYPE_MARKERS)


def _select_main_abstract(article_meta):
    if article_meta is None:
        return None, {
            "abstract_candidates": 0,
            "selected_abstract_type": "",
            "selection_reason": "no article-meta",
        }

    candidates = list(article_meta.findall("./{*}abstract"))
    usable = [node for node in candidates if not _is_excluded_abstract(node)]

    untyped = [node for node in usable if not _normalized_abstract_type(node)]

    if untyped:
        selected = untyped[0]
        reason = "preferred untyped main abstract"
    elif usable:
        selected = usable[0]
        reason = "fallback first non-excluded abstract"
    else:
        selected = None
        reason = "no non-excluded abstract"

    return selected, {
        "abstract_candidates": len(candidates),
        "usable_abstract_candidates": len(usable),
        "selected_abstract_type": (
            _normalized_abstract_type(selected)
            if selected is not None
            else ""
        ),
        "selection_reason": reason,
    }


def _extract_abstract_content(abstract_node):
    if abstract_node is None:
        return [], [], []

    section_titles = []
    sections = []
    paragraphs = []

    # Structured abstracts commonly use <sec><title>...<p>...</p></sec>.
    # Keep headings for display/audit but exclude headings from the term corpus.
    sec_nodes = list(abstract_node.findall("./{*}sec"))

    if sec_nodes:
        for sec in sec_nodes:
            label = _text(sec.find("./{*}title"))
            sec_paragraphs = [
                _text(p)
                for p in sec.findall(".//{*}p")
                if _text(p)
            ]

            if not sec_paragraphs:
                # Rare fallback: section text not wrapped in <p>.
                value = _text(sec)
                if label and value.startswith(label):
                    value = value[len(label):].strip()
                if value:
                    sec_paragraphs = [value]

            if label:
                section_titles.append(label)

            for value in sec_paragraphs:
                paragraphs.append(value)
                sections.append(
                    {
                        "label": label,
                        "raw_label": label,
                        "nlm_category": "",
                        "text": value,
                    }
                )
    else:
        direct_paragraphs = [
            _text(p)
            for p in abstract_node.findall(".//{*}p")
            if _text(p)
        ]

        if not direct_paragraphs:
            value = _text(abstract_node)
            if value:
                direct_paragraphs = [value]

        for value in direct_paragraphs:
            paragraphs.append(value)
            sections.append(
                {
                    "label": "",
                    "raw_label": "",
                    "nlm_category": "",
                    "text": value,
                }
            )

    return section_titles, paragraphs, sections


def parse_jats_abstract_only(xml_path):
    """Parse one PMC JATS file into the Project 2 abstract-only document contract."""
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    article = root
    if not str(article.tag).endswith("article"):
        article = root.find(".//{*}article")

    if article is None:
        raise ValueError("No <article> element found.")

    front = article.find("./{*}front")
    if front is None:
        raise ValueError("JATS article has no <front>.")

    article_meta = front.find("./{*}article-meta")
    if article_meta is None:
        raise ValueError("JATS article has no <article-meta>.")

    pmcid = _article_id(article_meta, "pmcid") or _article_id(article_meta, "pmc")
    pmid = _article_id(article_meta, "pmid")
    doi = _article_id(article_meta, "doi")

    title = _text(article_meta.find("./{*}title-group/{*}article-title"))
    journal = _text(
        front.find("./{*}journal-meta/{*}journal-title-group/{*}journal-title")
    )

    language = (
        article.attrib.get(XML_LANG)
        or article.attrib.get("xml:lang")
        or ""
    ).strip()

    abstract_node, selection = _select_main_abstract(article_meta)
    section_titles, paragraphs, sections = _extract_abstract_content(abstract_node)
    abstract_text = " ".join(paragraphs).strip()

    return {
        "filename": xml_path.name,
        "source_format": "JATS",
        "corpus_source": "PubMed-selected -> direct PMC JATS XML (pubmed_pmc)",
        "search_scope": "abstract_only",
        "analysis_scope": "main_abstract_body_only",
        "article_type": article.attrib.get("article-type", ""),
        "language": language,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "journal": journal,
        "title": title,
        "authors": _authors(article_meta),
        "affiliations": _affiliations(article_meta),
        "keywords": _keywords(article_meta),
        "abstract_section_titles": section_titles,
        "abstract_sections": sections,
        "abstract_paragraphs": paragraphs,
        "abstract_paragraphs_clean": list(paragraphs),
        "abstract": abstract_text,
        "abstract_clean": abstract_text,
        # Full text intentionally excluded from the Project 2 corpus.
        "section_titles": [],
        "body_paragraphs": [],
        "body_paragraphs_clean": [],
        "body": "",
        "body_clean": "",
        "definitions": [],
        "figures": [],
        "tables": [],
        "acknowledgments": [],
        "references": [],
        # M02 Document Statistics is also abstract-only for this corpus.
        "statistics_front_text": "",
        "statistics_body_text": abstract_text,
        "statistics_back_text": "",
        "statistics_text": abstract_text,
        "statistics_scope": (
            "Project 2 main JATS abstract body text only "
            "(structured section headings excluded)"
        ),
        # JATS <word-count> generally refers to the article, not this abstract,
        # so it is deliberately not used for the Project 2 abstract corpus.
        "reported_word_count": None,
        "reported_figure_count": None,
        "reported_table_count": None,
        "reported_ref_count": None,
        "reported_page_count": None,
        "abstract_selection": selection,
    }



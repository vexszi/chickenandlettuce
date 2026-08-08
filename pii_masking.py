from presidio_analyzer import AnalyzerEngine
from state import HospitalState

analyzer_engine = AnalyzerEngine()

def _mask(text: str) -> tuple[str, dict]:
    if not text:
        return "", {}

    results = analyzer_engine.analyze(text=text, language="en")
    results_in_order = sorted(results, key=lambda r: r.start)

    counts = {}
    pii_map = {}
    spans = []

    for r in results_in_order:
        entity_type = r.entity_type
        counts[entity_type] = counts.get(entity_type, 0) + 1
        placeholder = f"[{entity_type}_{counts[entity_type]}]"

        original_value = text[r.start:r.end]
        pii_map[placeholder] = original_value
        spans.append((r.start, r.end, placeholder))

    masked_text = text
    for start, end, placeholder in sorted(spans, key=lambda s: s[0], reverse=True):
        masked_text = masked_text[:start] + placeholder + masked_text[end:]

    return masked_text, pii_map

def pii_masking_node(state: HospitalState) -> dict:
    updates = {}

    if state.get("translated_text"):
        masked, pii_map = _mask(state["translated_text"])
        updates["masked_text"] = masked
        updates["pii_map"] = pii_map

    if state.get("translated_document_text"):
        masked_doc, doc_pii_map = _mask(state["translated_document_text"])
        updates["masked_document_text"] = masked_doc
        updates["pii_map"] = {**updates.get("pii_map", {}), **doc_pii_map}

    return updates
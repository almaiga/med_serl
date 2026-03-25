"""
Prompt loader for the Agentic UMLS Judge.

Loads all prompts from configs/prompts/agentic_judge_prompts.json.
No hardcoded prompt strings — edit the JSON config to modify behavior.

The judge follows a forced 3-step pipeline:
  Step 1 (Extract): Extract medical entities from the changed sentence.
  Step 2 (Retrieve): Query UMLS/RxNorm for each entity (handled in umls_async.py).
  Step 3 (Adjudicate): Given evidence, produce a verdict with CUI citations.
"""

import json
from pathlib import Path
from typing import Optional

# =============================================================================
# Load config
# =============================================================================

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "prompts" / "agentic_judge_prompts.json"
_config: Optional[dict] = None


def _load_config() -> dict:
    """Load and cache the judge prompts config."""
    global _config
    if _config is None:
        with open(_CONFIG_PATH, "r") as f:
            _config = json.load(f)
    return _config


def get_config() -> dict:
    """Public accessor for the full config dict."""
    return _load_config()


# =============================================================================
# Prompt accessors
# =============================================================================

def get_extraction_system_prompt() -> str:
    """System prompt for Step 1: entity extraction."""
    return _load_config()["extraction"]["system_prompt"]


def get_extraction_user_template() -> str:
    """User template for Step 1. Has placeholder: {sentence}"""
    return _load_config()["extraction"]["user_template"]


def get_adjudication_system_prompt() -> str:
    """System prompt for Step 3: adjudication."""
    return _load_config()["adjudication"]["system_prompt"]


def get_adjudication_user_template() -> str:
    """User template for Step 3. Placeholders: {original_sentence}, {modified_sentence}, {assessor_prediction}, {evidence_json}"""
    return _load_config()["adjudication"]["user_template"]


def get_model_params(step: str) -> dict:
    """Get default generation parameters for a judge step.
    
    Args:
        step: "extraction" or "adjudication"
    """
    return _load_config()["model_params"].get(step, {})


def get_evidence_format_config() -> dict:
    """Get the evidence formatting configuration."""
    return _load_config()["evidence_format"]


# =============================================================================
# Helper: format evidence for the adjudication prompt
# =============================================================================

def format_evidence_for_prompt(evidence_list: list) -> str:
    """Format a list of UMLS evidence dicts into a readable string for the judge.
    
    Uses templates from configs/prompts/agentic_judge_prompts.json.
    Edit the JSON config to change formatting.
    
    Args:
        evidence_list: List of dicts with keys like:
            entity_name, cui, semantic_type, synonyms, relations, source
    
    Returns:
        Formatted string for insertion into the adjudication prompt.
    """
    fmt = get_evidence_format_config()
    max_synonyms = fmt.get("max_synonyms_per_entity", 8)
    max_relations = fmt.get("max_relations_per_entity", 5)
    
    if not evidence_list:
        return fmt.get("no_evidence_message", "No UMLS evidence available.")
    
    lines = []
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        name = ev.get("entity_name", "unknown")
        cui = ev.get("cui", "N/A")
        sem_type = ev.get("semantic_type", "N/A")
        synonyms = ev.get("synonyms", [])
        relations = ev.get("relations", [])
        source = ev.get("source", "UMLS")

        # Entity header from config template
        line = fmt.get("entity_template", "Entity: {entity_name}").format(
            entity_name=name, cui=cui, semantic_type=sem_type, source=source
        )

        if synonyms:
            syn_str = ", ".join(synonyms[:max_synonyms])
            line += "\n" + fmt.get("synonyms_template", "  Synonyms: {synonyms}").format(
                synonyms=syn_str
            )

        if relations:
            rel_strs = []
            for rel in relations[:max_relations]:
                if not isinstance(rel, dict):
                    continue
                rel_strs.append(
                    f"{rel.get('relation', '?')} → {rel.get('related_name', '?')} "
                    f"(CUI: {rel.get('related_cui', '?')})"
                )
            line += "\n" + fmt.get("relations_template", "  Relations: {relations}").format(
                relations="; ".join(rel_strs)
            )

        lines.append(line)
    
    return "\n".join(lines)

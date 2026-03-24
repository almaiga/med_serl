"""
Async UMLS and RxNorm client for the Agentic Judge.

Provides non-blocking HTTP lookups against:
  - UMLS REST API (requires UMLS_API_KEY)
  - RxNorm REST API (no key required)

Ported from scripts/injection/medical_knowledge_base.py (sync requests)
to aiohttp for use inside verl's async RewardLoopWorker.

All results are cached in a module-level dict to avoid redundant API calls
within a training batch.
"""

import os
import json
import hashlib
import logging
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import aiohttp

logger = logging.getLogger(__name__)

# Load .env if available
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class UMLSEvidence:
    """Evidence retrieved from UMLS for a single medical entity."""
    entity_name: str
    cui: Optional[str] = None
    semantic_type: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    relations: List[Dict[str, str]] = field(default_factory=list)
    source: str = "UMLS"
    found: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Module-level cache
# =============================================================================

_CACHE: Dict[str, Any] = {}


def _cache_key(prefix: str, **kwargs) -> str:
    data = json.dumps(kwargs, sort_keys=True)
    return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"


# =============================================================================
# UMLS Async Client
# =============================================================================

UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
TRUSTED_SOURCES = ["SNOMEDCT_US", "NCI", "MSH", "MTH", "CHV", "MDR"]
TRUSTED_TTYS = ["PT", "SY", "ET", "LLT"]
UMLS_MAX_CONCURRENCY = int(os.getenv("UMLS_MAX_CONCURRENCY", "4"))
UMLS_ENTITY_TIMEOUT = float(os.getenv("UMLS_ENTITY_TIMEOUT", "20"))

# Semantic types we consider clinically relevant (from medical_knowledge_base.py)
CLINICAL_SEMANTIC_TYPES = {
    "Disease or Syndrome", "Neoplastic Process", "Sign or Symptom",
    "Pathologic Function", "Mental or Behavioral Dysfunction",
    "Injury or Poisoning", "Congenital Abnormality",
    "Pharmacologic Substance", "Clinical Drug", "Antibiotic",
    "Organic Chemical", "Amino Acid, Peptide, or Protein",
    "Therapeutic or Preventive Procedure", "Diagnostic Procedure",
    "Laboratory Procedure", "Laboratory or Test Result",
    "Body Part, Organ, or Organ Component", "Body System",
    "Body Location or Region", "Tissue",
    "Finding", "Cell or Molecular Dysfunction",
    "Anatomical Abnormality", "Acquired Abnormality",
    "Virus", "Bacterium", "Fungus",
    "Biomedical or Dental Material", "Medical Device",
    "Hormone", "Enzyme", "Vitamin", "Immunologic Factor",
    "Receptor", "Biologically Active Substance",
    "Temporal Concept", "Quantitative Concept",
    "Functional Concept", "Spatial Concept",
    "Molecular Function", "Cell Function",
    "Organ or Tissue Function", "Physiologic Function",
}


def _get_api_key() -> str:
    """Get UMLS API key from environment."""
    key = os.getenv("UMLS_API_KEY", "")
    if not key:
        logger.warning("UMLS_API_KEY not set — UMLS lookups will fail")
    return key


async def _async_get(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, str],
    timeout: float = 15.0,
) -> Optional[dict]:
    """Make a cached async GET request."""
    key = _cache_key("umls", url=url, params={k: v for k, v in params.items() if k != "apiKey"})
    if key in _CACHE:
        return _CACHE[key]

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                data = await resp.json()
                _CACHE[key] = data
                return data
            else:
                logger.debug(f"UMLS API returned {resp.status} for {url}")
                return None
    except Exception as e:
        logger.debug(f"UMLS API error for {url}: {e}")
        return None


async def async_get_cui(
    session: aiohttp.ClientSession,
    term: str,
    api_key: str = "",
) -> Optional[str]:
    """Resolve a medical term to its UMLS CUI."""
    api_key = api_key or _get_api_key()
    if not api_key:
        return None

    url = f"{UMLS_BASE_URL}/search/current"
    params = {
        "apiKey": api_key,
        "string": term,
        "searchType": "exact",
        "sabs": ",".join(TRUSTED_SOURCES),
        "pageSize": "5",
    }
    data = await _async_get(session, url, params)
    if data:
        results = data.get("result", {}).get("results", [])
        if results:
            return results[0].get("ui")
    
    # Fallback: try approximate search
    params["searchType"] = "words"
    data = await _async_get(session, url, params)
    if data:
        results = data.get("result", {}).get("results", [])
        if results:
            return results[0].get("ui")
    
    return None


async def async_get_synonyms(
    session: aiohttp.ClientSession,
    cui: str,
    api_key: str = "",
) -> List[str]:
    """Get synonym names for a CUI."""
    api_key = api_key or _get_api_key()
    if not api_key or not cui:
        return []

    url = f"{UMLS_BASE_URL}/content/current/CUI/{cui}/atoms"
    params = {
        "apiKey": api_key,
        "language": "ENG",
        "sabs": ",".join(TRUSTED_SOURCES),
        "ttys": ",".join(TRUSTED_TTYS),
        "pageSize": "50",
    }
    data = await _async_get(session, url, params)
    if not data:
        return []

    atoms = data.get("result", [])
    seen = set()
    synonyms = []
    for atom in atoms:
        name = atom.get("name", "").strip()
        name_lower = name.lower()
        if not name or name_lower in seen:
            continue
        # Skip pure codes
        import re
        if re.match(r"^[A-Z0-9\-\.]+$", name):
            continue
        seen.add(name_lower)
        synonyms.append(name)

    return synonyms


async def async_get_semantic_type(
    session: aiohttp.ClientSession,
    cui: str,
    api_key: str = "",
) -> Optional[str]:
    """Get the primary semantic type for a CUI."""
    api_key = api_key or _get_api_key()
    if not api_key or not cui:
        return None

    url = f"{UMLS_BASE_URL}/content/current/CUI/{cui}"
    params = {"apiKey": api_key}
    data = await _async_get(session, url, params)
    if data and "result" in data:
        types = data["result"].get("semanticTypes", [])
        if types:
            return types[0].get("name")
    return None


async def async_get_relations(
    session: aiohttp.ClientSession,
    cui: str,
    api_key: str = "",
    max_results: int = 10,
) -> List[Dict[str, str]]:
    """Get UMLS relations for a CUI (parent/child, related, broader/narrower).
    
    This is NEW — not in the existing sync medical_knowledge_base.py.
    Enables the judge to verify clinical relationships (e.g., drug-disease,
    contraindications, broader/narrower categorization).
    
    Returns:
        List of dicts: {relation, related_name, related_cui, source}
    """
    api_key = api_key or _get_api_key()
    if not api_key or not cui:
        return []

    url = f"{UMLS_BASE_URL}/content/current/CUI/{cui}/relations"
    params = {
        "apiKey": api_key,
        "pageSize": str(max_results),
    }
    data = await _async_get(session, url, params)
    if not data:
        return []

    raw_relations = data.get("result", [])
    relations = []
    for rel in raw_relations:
        related_label = rel.get("relatedIdName", "")
        related_id = rel.get("relatedId", "")
        relation_label = rel.get("relationLabel", rel.get("additionalRelationLabel", ""))

        # Extract CUI from URL if needed
        related_cui = ""
        if isinstance(related_id, str) and "/CUI/" in related_id:
            related_cui = related_id.split("/CUI/")[-1]
        elif isinstance(related_id, str) and related_id.startswith("C"):
            related_cui = related_id

        if related_label or related_cui:
            relations.append({
                "relation": relation_label,
                "related_name": related_label,
                "related_cui": related_cui,
                "source": rel.get("rootSource", ""),
            })

    return relations


# =============================================================================
# RxNorm Async Client (no API key required)
# =============================================================================

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


async def async_get_rxcui(
    session: aiohttp.ClientSession,
    drug_name: str,
) -> Optional[str]:
    """Resolve a drug name to its RxNorm RXCUI."""
    key = _cache_key("rxnorm_cui", drug=drug_name)
    if key in _CACHE:
        return _CACHE[key]

    url = f"{RXNORM_BASE_URL}/rxcui.json"
    params = {"name": drug_name, "search": "1"}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                ids = data.get("idGroup", {}).get("rxnormId", [])
                if ids:
                    _CACHE[key] = ids[0]
                    return ids[0]
    except Exception as e:
        logger.debug(f"RxNorm API error: {e}")
    return None


async def async_get_drug_class(
    session: aiohttp.ClientSession,
    rxcui: str,
) -> Optional[str]:
    """Get the Established Pharmacologic Class (EPC) for a drug RXCUI."""
    key = _cache_key("rxnorm_class", rxcui=rxcui)
    if key in _CACHE:
        return _CACHE[key]

    url = f"{RXNORM_BASE_URL}/rxclass/class/byDrugName.json"
    params = {"rxcui": rxcui, "relaSource": "FDASPL", "rela": "has_EPC"}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                classes = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
                if classes:
                    class_name = classes[0].get("rxclassMinConceptItem", {}).get("className", "")
                    if class_name:
                        _CACHE[key] = class_name
                        return class_name
    except Exception as e:
        logger.debug(f"RxNorm class API error: {e}")
    return None


# =============================================================================
# High-level: gather full evidence for an entity
# =============================================================================

async def gather_entity_evidence(
    session: aiohttp.ClientSession,
    entity_name: str,
    entity_type: str = "",
    api_key: str = "",
) -> UMLSEvidence:
    """Gather complete UMLS evidence for a single medical entity.
    
    Runs CUI lookup → synonyms + semantic type + relations in parallel.
    For medications, also queries RxNorm.
    
    Args:
        session: aiohttp session
        entity_name: The medical term to look up
        entity_type: Optional hint (medication, diagnosis, etc.)
        api_key: UMLS API key
        
    Returns:
        UMLSEvidence dataclass with all retrieved information.
    """
    api_key = api_key or _get_api_key()
    evidence = UMLSEvidence(entity_name=entity_name)
    
    # Step 1: Resolve CUI
    cui = await async_get_cui(session, entity_name, api_key)
    if not cui:
        # Try without possessives, plurals, etc.
        import re
        clean_name = re.sub(r"'s$", "", entity_name)
        clean_name = re.sub(r"s$", "", clean_name) if len(clean_name) > 4 else clean_name
        if clean_name != entity_name:
            cui = await async_get_cui(session, clean_name, api_key)
    
    if not cui:
        # Try RxNorm for medications
        if entity_type.lower() in ("medication", "drug", "pharmacologic_substance", ""):
            rxcui = await async_get_rxcui(session, entity_name)
            if rxcui:
                drug_class = await async_get_drug_class(session, rxcui)
                evidence.source = "RxNorm"
                evidence.found = True
                evidence.cui = f"RXCUI:{rxcui}"
                if drug_class:
                    evidence.synonyms = [drug_class]
                    evidence.semantic_type = "Pharmacologic Substance"
                return evidence
        return evidence  # found=False
    
    evidence.cui = cui
    evidence.found = True
    
    # Step 2: Parallel fetch synonyms, semantic type, relations
    syn_task = asyncio.create_task(async_get_synonyms(session, cui, api_key))
    sem_task = asyncio.create_task(async_get_semantic_type(session, cui, api_key))
    rel_task = asyncio.create_task(async_get_relations(session, cui, api_key))

    synonyms, sem_type, relations = await asyncio.gather(
        syn_task, sem_task, rel_task, return_exceptions=True
    )

    if isinstance(synonyms, Exception):
        logger.debug(f"UMLS synonyms lookup failed for {entity_name}: {synonyms}")
        synonyms = []
    if isinstance(sem_type, Exception):
        logger.debug(f"UMLS semantic type lookup failed for {entity_name}: {sem_type}")
        sem_type = ""
    if isinstance(relations, Exception):
        logger.debug(f"UMLS relations lookup failed for {entity_name}: {relations}")
        relations = []
    
    evidence.synonyms = synonyms[:10]  # Cap at 10
    evidence.semantic_type = sem_type or ""
    evidence.relations = relations[:5]  # Cap at 5
    evidence.source = "UMLS"
    
    return evidence


async def gather_evidence_batch(
    session: aiohttp.ClientSession,
    entities: List[Dict[str, str]],
    api_key: str = "",
) -> List[UMLSEvidence]:
    """Gather UMLS evidence for a batch of entities in parallel.
    
    Args:
        session: aiohttp session
        entities: List of {"name": str, "type": str} dicts
        api_key: UMLS API key
        
    Returns:
        List of UMLSEvidence, one per entity.
    """
    semaphore = asyncio.Semaphore(max(1, UMLS_MAX_CONCURRENCY))

    async def _run_one(ent: Dict[str, str]) -> UMLSEvidence:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    gather_entity_evidence(session, ent["name"], ent.get("type", ""), api_key),
                    timeout=UMLS_ENTITY_TIMEOUT,
                )
            except Exception as e:
                logger.debug(f"Evidence lookup failed for {ent.get('name', '')}: {e}")
                return UMLSEvidence(entity_name=ent.get("name", ""))

    tasks = [_run_one(ent) for ent in entities]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    evidence_list: List[UMLSEvidence] = []
    for ent, result in zip(entities, results):
        if isinstance(result, Exception):
            logger.debug(f"Evidence batch task failed for {ent.get('name', '')}: {result}")
            evidence_list.append(UMLSEvidence(entity_name=ent.get("name", "")))
        else:
            evidence_list.append(result)
    return evidence_list


def clear_cache():
    """Clear the module-level cache. Call between training epochs if desired."""
    global _CACHE
    _CACHE.clear()

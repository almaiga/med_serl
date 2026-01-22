import requests
from typing import Optional, Dict, List
import re
import os

# ============================================
# UMLS API (Requires API Key)
# https://documentation.uts.nlm.nih.gov/rest/home.html
# ============================================

class UMLSClient:
    """UMLS Terminology Services API - Requires API Key."""
    
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        """Initialize with API key from environment or parameter."""
        self.api_key = api_key or os.getenv("UMLS_API_KEY")
        self.verbose = verbose
        if not self.api_key:
            raise ValueError("UMLS API key required. Set UMLS_API_KEY environment variable or pass api_key parameter.")
        
        if self.verbose:
            print(f"[DEBUG] UMLS API Key loaded: {self.api_key[:10]}...")
    
    def _get_ticket(self) -> Optional[str]:
        """Get service ticket for authenticated requests."""
        try:
            # Step 1: Get TGT (Ticket Granting Ticket)
            tgt_url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
            response = requests.post(tgt_url, data={"apikey": self.api_key}, timeout=10)
            
            if self.verbose:
                print(f"[DEBUG] TGT request status: {response.status_code}")
            
            if response.status_code == 201:
                # Extract TGT endpoint from HTML form action
                match = re.search(r'action="([^"]+)"', response.text)
                if match:
                    tgt_endpoint = match.group(1)
                    
                    if self.verbose:
                        print(f"[DEBUG] TGT endpoint: {tgt_endpoint}")
                    
                    # Step 2: Request service ticket using TGT
                    ticket_response = requests.post(
                        tgt_endpoint,
                        data={"service": "http://umlsks.nlm.nih.gov"},
                        timeout=10
                    )
                    
                    if ticket_response.status_code == 200:
                        ticket = ticket_response.text.strip()
                        
                        if self.verbose:
                            print(f"[DEBUG] Got ticket: {ticket[:30]}...")
                        
                        return ticket
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Ticket error: {e}")
        return None
    
    def search_concept(self, term: str) -> Optional[Dict]:
        """Search for a concept in UMLS."""
        ticket = self._get_ticket()
        if not ticket:
            if self.verbose:
                print(f"[DEBUG] No ticket available for search: {term}")
            return None
        
        try:
            url = f"{self.BASE_URL}/search/current"
            params = {
                "string": term,
                "ticket": ticket,
                "pageSize": 10
            }
            
            if self.verbose:
                print(f"[DEBUG] Searching UMLS for: {term}")
            
            response = requests.get(url, params=params, timeout=10)
            
            if self.verbose:
                print(f"[DEBUG] Search status: {response.status_code}")
            
            if response.ok:
                data = response.json()
                results = data.get("result", {}).get("results", [])
                if results:
                    if self.verbose:
                        print(f"[DEBUG] Found {len(results)} results")
                    return {
                        "term": term,
                        "concepts": results,
                        "source": "UMLS"
                    }
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Search error: {e}")
        return None
    
    def get_concept_details(self, cui: str) -> Optional[Dict]:
        """Get detailed information about a concept by CUI."""
        ticket = self._get_ticket()
        if not ticket:
            return None
        
        try:
            url = f"{self.BASE_URL}/content/current/CUI/{cui}"
            params = {"ticket": ticket}
            response = requests.get(url, params=params, timeout=10)
            if response.ok:
                return response.json()
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Details error: {e}")
        return None
    
    def get_synonyms(self, cui: str) -> Optional[List[str]]:
        """Get all synonyms (atoms) for a concept - fetch ALL pages."""
        ticket = self._get_ticket()
        if not ticket:
            return None
        
        try:
            all_synonyms = set()
            page = 1
            page_size = 100  # Increased page size
            
            while True:
                url = f"{self.BASE_URL}/content/current/CUI/{cui}/atoms"
                params = {
                    "ticket": ticket,
                    "language": "ENG",
                    "pageSize": page_size,
                    "pageNumber": page
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if not response.ok:
                    break
                
                data = response.json()
                atoms = data.get("result", [])
                
                if not atoms:
                    break
                
                # Extract names from atoms
                for atom in atoms:
                    name = atom.get("name")
                    if name:
                        # Filter out technical codes and keep readable names
                        if not re.match(r'^[A-Z0-9\-]+$', name):  # Skip pure codes like "C0025859"
                            all_synonyms.add(name)
                
                # Check if there are more pages
                page_count = data.get("pageCount", 1)
                if page >= page_count:
                    break
                
                page += 1
            
            if self.verbose:
                print(f"[DEBUG] Total unique synonyms: {len(all_synonyms)}")
            
            return list(all_synonyms) if all_synonyms else None
            
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Synonyms error: {e}")
        return None
    
    def get_related_concepts(self, cui: str) -> Optional[List[Dict]]:
        """Get related concepts (useful for finding drug classes, etc.)."""
        ticket = self._get_ticket()
        if not ticket:
            return None
        
        try:
            url = f"{self.BASE_URL}/content/current/CUI/{cui}/relations"
            params = {
                "ticket": ticket,
                "pageSize": 25
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.ok:
                data = response.json()
                relations = data.get("result", [])
                
                # Filter for useful relations
                related = []
                for rel in relations:
                    rel_type = rel.get("relationLabel", "")
                    related_cui = rel.get("relatedId")
                    related_name = rel.get("relatedIdName")
                    
                    if related_name and rel_type in ["isa", "has_tradename", "tradename_of"]:
                        related.append({
                            "type": rel_type,
                            "cui": related_cui,
                            "name": related_name
                        })
                
                return related if related else None
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Relations error: {e}")
        return None
    
    def get_semantic_types(self, cui: str) -> Optional[List[str]]:
        """Get semantic types for a concept."""
        details = self.get_concept_details(cui)
        if details and "result" in details:
            semantic_types = details["result"].get("semanticTypes", [])
            return [st.get("name") for st in semantic_types if st.get("name")]
        return None


# ============================================
# FALLBACK: RxNorm API (NIH - No API Key Required)
# Used as fallback if UMLS fails
# ============================================

class RxNormClient:
    """RxNorm API for drug information - NO API KEY NEEDED."""
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """Get RxCUI for a drug name."""
        try:
            url = f"{self.BASE_URL}/rxcui.json"
            response = requests.get(url, params={"name": drug_name}, timeout=10)
            if response.ok:
                data = response.json()
                rxcui_list = data.get("idGroup", {}).get("rxnormId", [])
                return rxcui_list[0] if rxcui_list else None
        except Exception:
            pass
        return None
    
    def get_drug_class(self, drug_name: str) -> Optional[dict]:
        """Get drug class for a medication."""
        rxcui = self.get_rxcui(drug_name)
        if not rxcui:
            return None
        
        try:
            url = f"{self.BASE_URL}/rxclass/class/byRxcui.json"
            response = requests.get(url, params={"rxcui": rxcui}, timeout=10)
            if response.ok:
                data = response.json()
                class_list = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
                if class_list:
                    # Only keep pharmacologic/chemical classes (EPC, CHEM)
                    pharm_classes = [
                        c.get("rxclassMinConceptItem", {}).get("className")
                        for c in class_list
                        if c.get("rxclassMinConceptItem", {}).get("className")
                        and c.get("rxclassMinConceptItem", {}).get("classType") in ("EPC", "CHEM")
                    ]
                    # Fallback: if none, use all classes as before
                    classes = pharm_classes if pharm_classes else [
                        c.get("rxclassMinConceptItem", {}).get("className")
                        for c in class_list
                        if c.get("rxclassMinConceptItem", {}).get("className")
                    ]
                    return {
                        "drug": drug_name,
                        "rxcui": rxcui,
                        "classes": list(set(classes))[:5],
                        "source": "RxNorm"
                    }
        except Exception:
            pass
        return None


# ============================================
# TEMPORAL CONVERSION (Rule-based, no API)
# ============================================

def get_temporal_synonym(term: str) -> dict:
    """Convert temporal expressions using mathematical equivalence."""
    
    term_lower = term.lower().strip()
    
    # Pattern: "X-unit" or "X unit"
    patterns = [
        (r'(\d+)-year', lambda m: f"{int(m.group(1)) * 12}-month"),
        (r'(\d+)-month', lambda m: f"{int(m.group(1)) * 4}-week" if int(m.group(1)) <= 3 else f"{int(m.group(1)) * 30}-day"),
        (r'(\d+)-week', lambda m: f"{int(m.group(1)) * 7}-day"),
        (r'(\d+)-day', lambda m: f"{int(m.group(1)) * 24}-hour"),
        (r'(\d+)\s+years?', lambda m: f"{int(m.group(1)) * 12} months"),
        (r'(\d+)\s+months?', lambda m: f"{int(m.group(1)) * 4} weeks" if int(m.group(1)) <= 3 else f"{int(m.group(1)) * 30} days"),
        (r'(\d+)\s+weeks?', lambda m: f"{int(m.group(1)) * 7} days"),
        (r'(\d+)\s+days?', lambda m: f"{int(m.group(1)) * 24} hours"),
    ]
    
    for pattern, replacer in patterns:
        match = re.search(pattern, term_lower, re.IGNORECASE)
        if match:
            try:
                replacement_part = replacer(match)
                new_term = re.sub(pattern, replacement_part, term_lower, count=1, flags=re.IGNORECASE)
                return {
                    "original": term,
                    "replacement": new_term,
                    "verified": True,
                    "source": "temporal_conversion"
                }
            except Exception:
                pass
    
    return {
        "original": term,
        "verified": False,
        "source": None
    }


# ============================================
# UNIFIED LOOKUP FUNCTIONS (UMLS Primary)
# ============================================

def get_verified_synonym(term: str, api_key: Optional[str] = None, verbose: bool = False) -> dict:
    """
    Get a verified medical synonym using UMLS API (primary) with RxNorm fallback.
    
    Args:
        term: Medical term to look up
        api_key: UMLS API key (optional if UMLS_API_KEY env var is set)
        verbose: Enable debug output
    """
    
    term_lower = term.lower().strip()
    
    # 1. Try UMLS first (most comprehensive)
    try:
        umls = UMLSClient(api_key, verbose=verbose)
        
        # Search for concept
        search_result = umls.search_concept(term)
        if search_result and search_result.get("concepts"):
            # Get first concept
            first_concept = search_result["concepts"][0]
            cui = first_concept.get("ui")
            concept_name = first_concept.get("name")
            
            if cui:
                # Get synonyms (all pages)
                synonyms = umls.get_synonyms(cui)
                
                # Get related concepts (trade names, etc.)
                related = umls.get_related_concepts(cui)
                
                if synonyms:
                    # Filter out the original term
                    filtered_synonyms = [
                        s for s in synonyms 
                        if s.lower() != term_lower
                    ]
                    
                    # Get semantic types
                    semantic_types = umls.get_semantic_types(cui)
                    
                    result = {
                        "original": term,
                        "concept": concept_name,
                        "cui": cui,
                        "synonyms": filtered_synonyms[:20],  # Top 20 synonyms
                        "semantic_types": semantic_types[:5] if semantic_types else [],
                        "verified": True,
                        "source": "UMLS"
                    }
                    
                    if related:
                        result["related"] = related[:5]
                    
                    return result
    except ValueError as e:
        if verbose:
            print(f"[DEBUG] UMLS not available: {e}")
    except Exception as e:
        if verbose:
            print(f"[DEBUG] UMLS error: {e}")
    
    # 2. Fallback to RxNorm for drugs (no key needed)
    rxnorm = RxNormClient()
    
    # Try drug class
    drug_info = rxnorm.get_drug_class(term)
    if drug_info and drug_info.get("classes"):
        return {
            "original": term,
            "drug_class": drug_info["classes"][0],
            "synonyms": drug_info["classes"],
            "verified": True,
            "source": "RxNorm (fallback)"
        }
    
    # Not found in any API
    return {
        "original": term,
        "verified": False,
        "source": None
    }


# ============================================
# QUICK TEST
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Medical Knowledge Base (UMLS API)")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("UMLS_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: UMLS_API_KEY not found in environment variables!")
        print("Please set it with: export UMLS_API_KEY='your_api_key'")
        print("Falling back to RxNorm only...\n")
        VERBOSE = False
    else:
        print(f"\n✓ UMLS API key found: {api_key[:10]}...\n")
        VERBOSE = True
    
    # Test drugs with full output
    print("\n1. Testing DRUG lookups:\n")
    drug_tests = ["metoprolol", "aspirin", "lisinopril"]
    
    for term in drug_tests:
        result = get_verified_synonym(term, verbose=VERBOSE)
        status = "✓" if result.get("verified") else "✗"
        print(f"\n{status} {term}")
        print(f"    Source: {result.get('source', 'not found')}")
        if result.get('cui'):
            print(f"    CUI: {result['cui']}")
        if result.get('concept'):
            print(f"    Concept: {result['concept']}")
        if result.get('semantic_types'):
            print(f"    Types: {', '.join(result['semantic_types'][:2])}")
        if result.get('synonyms'):
            print(f"    Synonyms ({len(result['synonyms'])}): {result['synonyms'][:8]}")
        if result.get('related'):
            print(f"    Related: {[r['name'] for r in result['related'][:3]]}")
    
    # Test diseases
    print("\n\n2. Testing DISEASE lookups:\n")
    disease_tests = ["diabetes", "hypertension", "pneumonia"]
    
    for term in disease_tests:
        result = get_verified_synonym(term, verbose=False)
        status = "✓" if result.get("verified") else "✗"
        print(f"\n{status} {term}")
        print(f"    Source: {result.get('source', 'not found')}")
        if result.get('cui'):
            print(f"    CUI: {result['cui']}")
        if result.get('synonyms'):
            print(f"    Synonyms ({len(result['synonyms'])}): {result['synonyms'][:8]}")
    
    # Test temporal
    print("\n\n3. Testing TEMPORAL conversions:\n")
    temporal_tests = ["1-year history", "2-week course"]
    
    for term in temporal_tests:
        result = get_temporal_synonym(term)
        status = "✓" if result.get("verified") else "✗"
        print(f"{status} {term} → {result.get('replacement', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✓ UMLS Integration Complete!")
    print("=" * 60)
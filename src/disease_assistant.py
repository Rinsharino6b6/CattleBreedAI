import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

from src.config import DISEASES_INFO_PATH

_CACHED_DISEASES_DATA = None

def load_disease_knowledge_base() -> Dict[str, Any]:
    """Load and cache the verified cattle & buffalo disease knowledge base."""
    global _CACHED_DISEASES_DATA
    if _CACHED_DISEASES_DATA is not None:
        return _CACHED_DISEASES_DATA

    if not DISEASES_INFO_PATH.exists():
        raise FileNotFoundError(f"Disease knowledge base not found at: {DISEASES_INFO_PATH}")

    with open(DISEASES_INFO_PATH, "r", encoding="utf-8") as f:
        _CACHED_DISEASES_DATA = json.load(f)

    return _CACHED_DISEASES_DATA

def get_symptom_categories() -> Dict[str, List[Dict[str, Any]]]:
    """Return categorized symptoms for visual checkbox selection in the UI."""
    data = load_disease_knowledge_base()
    return data.get("symptom_categories", {})

def get_emergency_symptom_ids() -> Set[str]:
    """Return all symptom IDs marked as potential veterinary emergencies."""
    data = load_disease_knowledge_base()
    categories = data.get("symptom_categories", {})
    emergency_ids = set()
    for category, symptoms in categories.items():
        for s in symptoms:
            if s.get("is_emergency", False):
                emergency_ids.add(s["id"])
    return emergency_ids

def check_emergency_triage(selected_symptom_ids: List[str]) -> Dict[str, Any]:
    """
    Evaluate selected symptoms against critical emergency criteria:
    - Acute Ruminal Bloat (Asphyxiation risk)
    - Hemorrhagic Septicemia throat edema & gasping
    - Milk Fever recumbency & cold extremities
    - Black Quarter acute gas gangrene
    - Red water / severe hemolysis
    """
    emergency_ids = get_emergency_symptom_ids()
    critical_matches = [sid for sid in selected_symptom_ids if sid in emergency_ids]

    is_emergency = len(critical_matches) > 0

    # Specific emergency alerts
    alerts = []
    if "swollen_left_flank" in critical_matches:
        alerts.append("CRITICAL BLOAT / AFARA: Risk of suffocation and cardiac failure. Keep animal on an incline and call veterinarian immediately.")
    if "breathing_difficulty" in critical_matches or "throat_swelling" in critical_matches:
        alerts.append("SEVERE RESPIRATORY DISTRESS / POSSIBLE HS: Throat swelling and labored breathing can be fatal within hours.")
    if "inability_to_stand" in critical_matches or "cold_ears_legs" in critical_matches:
        alerts.append("DOWNER ANIMAL / SUSPECTED MILK FEVER: Recumbency and hypothermia require emergency intravenous calcium / glucose. Do NOT force liquid into mouth.")
    if "sudden_lameness" in critical_matches and "muscle_swelling" in critical_matches:
        alerts.append("POSSIBLE BLACK QUARTER (BQ): Acute crepitating muscle swelling requires immediate emergency antibiotics.")
    if "red_coffee_urine" in critical_matches:
        alerts.append("SEVERE HEMOLYSIS (RED WATER / BABESIOSIS): Dark urine indicates rapid red blood cell breakdown.")

    return {
        "is_emergency": is_emergency,
        "critical_symptom_ids": critical_matches,
        "emergency_count": len(critical_matches),
        "alerts": alerts,
        "emergency_helpline": "1962",
        "action_instruction": "Contact your nearest Veterinary Dispensary or call the 1962 Mobile Veterinary Emergency Ambulance immediately."
    }

def analyze_symptoms(
    selected_symptom_ids: List[str],
    species: str = "Cattle"
) -> Dict[str, Any]:
    """
    Analyze farmer-reported symptoms, compute matching disease profiles,
    and return triaged healthcare recommendations.
    """
    if not selected_symptom_ids:
        return {
            "has_input": False,
            "matches": [],
            "triage": {"is_emergency": False, "alerts": []},
            "disclaimer": "Please select one or more observed symptoms to receive veterinary guidance."
        }

    data = load_disease_knowledge_base()
    all_diseases = data.get("diseases", [])
    triage = check_emergency_triage(selected_symptom_ids)

    selected_set = set(selected_symptom_ids)
    scored_diseases = []

    for d in all_diseases:
        # Check species match
        if species not in d.get("affected_species", ["Cattle", "Buffalo"]):
            continue

        disease_tags = set(d.get("symptom_tags", []))
        overlap = selected_set.intersection(disease_tags)
        
        if not overlap:
            continue

        # Weighted score: overlap relative to total symptoms and disease symptoms
        match_score = (len(overlap) / len(disease_tags)) * 100.0
        # Boost if user-reported key symptoms match
        relevance_score = (len(overlap) / len(selected_set)) * 100.0
        combined_score = round(0.6 * match_score + 0.4 * relevance_score, 1)

        missing_symptoms = [s for s in d.get("symptom_tags", []) if s not in selected_set]

        scored_diseases.append({
            "id": d["id"],
            "name": d["name"],
            "vernacular_name": d["vernacular_name"],
            "pathogen_type": d["pathogen_type"],
            "severity": d["severity"],
            "is_emergency": d["is_emergency"],
            "match_score": min(combined_score, 98.0),  # Never claim 100% certainty without lab confirmation
            "matched_symptoms_count": len(overlap),
            "matched_tags": list(overlap),
            "missing_tags": missing_symptoms,
            "clinical_signs": d["clinical_signs"],
            "first_aid": d["first_aid"],
            "preventive_measures": d["preventive_measures"],
            "disclaimer": d["disclaimer"]
        })

    # Sort descending by match score
    scored_diseases.sort(key=lambda x: (x["is_emergency"], x["match_score"]), reverse=True)

    return {
        "has_input": True,
        "species": species,
        "selected_symptoms_count": len(selected_symptom_ids),
        "triage": triage,
        "matches": scored_diseases[:4],  # Top 4 most relevant matches
        "all_match_count": len(scored_diseases),
        "mandatory_disclaimer": (
            "IMPORTANT VETERINARY NOTICE: This tool provides preliminary supportive guidance "
            "based on livestock health databases and DOES NOT constitute a confirmed medical diagnosis. "
            "Never administer prescription antibiotics without a registered veterinary practitioner's in-person examination."
        )
    }

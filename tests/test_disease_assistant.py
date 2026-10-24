import unittest
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.disease_assistant import (
    load_disease_knowledge_base, get_symptom_categories,
    check_emergency_triage, analyze_symptoms
)
from src.veterinary_support import (
    create_consultation_ticket, get_ticket_by_id, get_tickets_by_phone,
    get_national_helplines, get_state_directories, get_state_directory
)

class TestDiseaseAssistantAndVetSupport(unittest.TestCase):

    def test_01_knowledge_base_loaded(self):
        """Verify disease knowledge base structure and coverage."""
        data = load_disease_knowledge_base()
        diseases = data.get("diseases", [])
        self.assertGreaterEqual(len(diseases), 8)
        
        # Check required fields in each disease
        for d in diseases:
            self.assertIn("name", d)
            self.assertIn("severity", d)
            self.assertIn("symptom_tags", d)
            self.assertIn("first_aid", d)
            self.assertIn("preventive_measures", d)

    def test_02_emergency_triage_critical_symptoms(self):
        """Verify that life-threatening symptoms trigger an emergency alert."""
        # Test bloat emergency
        triage_bloat = check_emergency_triage(["swollen_left_flank", "abdominal_pain"])
        self.assertTrue(triage_bloat["is_emergency"])
        self.assertEqual(triage_bloat["emergency_helpline"], "1962")

        # Test milk fever recumbency emergency
        triage_downer = check_emergency_triage(["inability_to_stand", "cold_ears_legs"])
        self.assertTrue(triage_downer["is_emergency"])

        # Test hemorrhagic septicemia respiratory emergency
        triage_hs = check_emergency_triage(["throat_swelling", "breathing_difficulty"])
        self.assertTrue(triage_hs["is_emergency"])

    def test_03_non_emergency_guidance(self):
        """Verify non-emergency symptoms receive guidance without panic alert."""
        triage_mild = check_emergency_triage(["coughing", "appetite_loss"])
        self.assertFalse(triage_mild["is_emergency"])

        # Test analysis with mild symptoms
        res = analyze_symptoms(["coughing", "nasal_discharge"], species="Cattle")
        self.assertTrue(res["has_input"])
        self.assertFalse(res["triage"]["is_emergency"])
        self.assertGreater(len(res["matches"]), 0)

    def test_04_consultation_ticket_sqlite(self):
        """Verify creating, persisting, and querying consultation tickets in SQLite."""
        ticket_res = create_consultation_ticket(
            farmer_name="Ramesh Patel",
            phone_number="9876543210",
            state="Gujarat",
            district="Anand",
            animal_type="Cattle",
            breed="Gir",
            symptoms=["High fever", "mouth blisters", "drooling saliva"],
            severity_level="High",
            farmer_notes="Cow stopped feeding since yesterday."
        )

        self.assertTrue(ticket_res["success"])
        ticket_id = ticket_res["ticket_id"]
        self.assertTrue(ticket_id.startswith("VET-"))

        # Retrieve by Ticket ID
        fetched = get_ticket_by_id(ticket_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["farmer_name"], "Ramesh Patel")
        self.assertEqual(fetched["state"], "Gujarat")
        self.assertEqual(fetched["animal_type"], "Cattle")

        # Retrieve by Phone Number
        phone_results = get_tickets_by_phone("9876543210")
        self.assertGreaterEqual(len(phone_results), 1)

    def test_05_veterinary_directory(self):
        """Verify national helplines and state directory query."""
        nationals = get_national_helplines()
        self.assertGreaterEqual(len(nationals), 2)
        
        # Verify 1962 is present
        numbers = [nh["number"] for nh in nationals]
        self.assertIn("1962", numbers)

        states = get_state_directories()
        self.assertGreaterEqual(len(states), 10)

        # Check Gujarat & Punjab queries
        gujarat_info = get_state_directory("Gujarat")
        self.assertIsNotNone(gujarat_info)
        self.assertEqual(gujarat_info["primary_emergency"], "1962")

if __name__ == "__main__":
    unittest.main()

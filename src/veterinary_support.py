import json
import sqlite3
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import VET_DB_PATH, VET_DIRECTORY_PATH

def get_db_connection() -> sqlite3.Connection:
    """Create or connect to the SQLite veterinary support database."""
    VET_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(VET_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_veterinary_db() -> None:
    """Initialize consultation tickets table schema if not already present."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultation_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE NOT NULL,
                farmer_name TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                animal_type TEXT NOT NULL,
                breed TEXT,
                duration_days INTEGER,
                symptoms TEXT NOT NULL,
                severity_level TEXT NOT NULL,
                image_path TEXT,
                farmer_notes TEXT,
                status TEXT NOT NULL DEFAULT 'Submitted / Pending Doctor Review',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

# Initialize DB on module import
init_veterinary_db()

def generate_ticket_id() -> str:
    """Generate a unique human-readable Ticket ID, e.g. VET-2026-A8K2."""
    year = datetime.now().year
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"VET-{year}-{suffix}"

def create_consultation_ticket(
    farmer_name: str,
    phone_number: str,
    state: str,
    district: str,
    animal_type: str,
    symptoms: List[str],
    severity_level: str,
    breed: Optional[str] = None,
    duration_days: int = 1,
    image_path: Optional[str] = None,
    farmer_notes: Optional[str] = None
) -> Dict[str, Any]:
    """Create and persist a new veterinary consultation ticket."""
    ticket_id = generate_ticket_id()
    symptoms_str = ", ".join(symptoms) if isinstance(symptoms, list) else str(symptoms)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO consultation_tickets (
                ticket_id, farmer_name, phone_number, state, district,
                animal_type, breed, duration_days, symptoms, severity_level,
                image_path, farmer_notes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id, farmer_name.strip(), phone_number.strip(), state.strip(), district.strip(),
            animal_type, breed or "Unknown Breed", duration_days, symptoms_str, severity_level,
            image_path or "", farmer_notes or "", "Submitted / Pending Doctor Review"
        ))
        conn.commit()

    return {
        "success": True,
        "ticket_id": ticket_id,
        "farmer_name": farmer_name,
        "phone_number": phone_number,
        "severity_level": severity_level,
        "status": "Submitted / Pending Doctor Review",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_ticket_by_id(ticket_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve ticket details by Ticket ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM consultation_tickets WHERE ticket_id = ?", (ticket_id.strip(),))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None

def get_tickets_by_phone(phone_number: str) -> List[Dict[str, Any]]:
    """Retrieve all tickets registered under a farmer's phone number."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM consultation_tickets 
            WHERE phone_number LIKE ? 
            ORDER BY created_at DESC
        """, (f"%{phone_number.strip()}%",))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def list_recent_tickets(limit: int = 10) -> List[Dict[str, Any]]:
    """List recent consultation tickets."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM consultation_tickets ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

# ------------------------------------------------------------------------------
# VETERINARY DIRECTORY QUERIES
# ------------------------------------------------------------------------------
_CACHED_VET_DIRECTORY = None

def load_veterinary_directory() -> Dict[str, Any]:
    """Load verified national and state veterinary directories."""
    global _CACHED_VET_DIRECTORY
    if _CACHED_VET_DIRECTORY is not None:
        return _CACHED_VET_DIRECTORY

    if not VET_DIRECTORY_PATH.exists():
        return {"national_helplines": [], "state_directories": []}

    with open(VET_DIRECTORY_PATH, "r", encoding="utf-8") as f:
        _CACHED_VET_DIRECTORY = json.load(f)

    return _CACHED_VET_DIRECTORY

def get_national_helplines() -> List[Dict[str, Any]]:
    """Get verified government national animal disease helplines."""
    data = load_veterinary_directory()
    return data.get("national_helplines", [])

def get_state_directories() -> List[Dict[str, Any]]:
    """Get all state animal husbandry and hospital contact directories."""
    data = load_veterinary_directory()
    return data.get("state_directories", [])

def get_state_directory(state_name: str) -> Optional[Dict[str, Any]]:
    """Get veterinary directory for a specific state."""
    states = get_state_directories()
    for s in states:
        if s.get("state", "").lower() == state_name.lower():
            return s
    return None

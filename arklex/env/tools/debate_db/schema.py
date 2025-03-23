"""Common schema definitions for debate database tools."""

from typing import Dict, Any

# Table names
DEBATE_HISTORY_TABLE = "debate_history"
EFFECTIVENESS_SCORES_TABLE = "effectiveness_scores"

# Column names for debate_history table
DEBATE_COLUMNS = {
    "id": "id",
    "persuasion_technique": "persuasion_technique",
    "effectiveness_score": "effectiveness_score",
    "timestamp": "timestamp",
    "suggestion": "suggestion"
}

# Column names for effectiveness_scores table
EFFECTIVENESS_COLUMNS = {
    "id": "id",
    "persuasion_type": "persuasion_type",
    "score": "score",
    "updated_at": "updated_at"
}

# SQL for creating tables
CREATE_DEBATE_HISTORY_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {DEBATE_HISTORY_TABLE} (
        {DEBATE_COLUMNS['id']} INTEGER PRIMARY KEY AUTOINCREMENT,
        {DEBATE_COLUMNS['persuasion_technique']} TEXT NOT NULL,
        {DEBATE_COLUMNS['effectiveness_score']} REAL NOT NULL,
        {DEBATE_COLUMNS['timestamp']} DATETIME DEFAULT CURRENT_TIMESTAMP,
        {DEBATE_COLUMNS['suggestion']} TEXT
    )
"""

CREATE_EFFECTIVENESS_SCORES_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {EFFECTIVENESS_SCORES_TABLE} (
        {EFFECTIVENESS_COLUMNS['id']} INTEGER PRIMARY KEY AUTOINCREMENT,
        {EFFECTIVENESS_COLUMNS['persuasion_type']} TEXT NOT NULL UNIQUE,
        {EFFECTIVENESS_COLUMNS['score']} REAL NOT NULL,
        {EFFECTIVENESS_COLUMNS['updated_at']} TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

def format_debate_record(row: tuple) -> Dict[str, Any]:
    """Format a debate history record from database row.
    
    Args:
        row: Database row containing debate record fields
        
    Returns:
        Formatted dictionary with debate record data
    """
    return {
        DEBATE_COLUMNS['id']: row[0],
        DEBATE_COLUMNS['persuasion_technique']: row[1],
        DEBATE_COLUMNS['effectiveness_score']: row[2],
        DEBATE_COLUMNS['timestamp']: row[3],
        DEBATE_COLUMNS['suggestion']: row[4]
    }

def format_effectiveness_record(row: tuple) -> Dict[str, Any]:
    """Format an effectiveness score record from database row.
    
    Args:
        row: Database row containing effectiveness score fields
        
    Returns:
        Formatted dictionary with effectiveness score data
    """
    return {
        EFFECTIVENESS_COLUMNS['id']: row[0],
        EFFECTIVENESS_COLUMNS['persuasion_type']: row[1],
        EFFECTIVENESS_COLUMNS['score']: row[2],
        EFFECTIVENESS_COLUMNS['updated_at']: row[3]
    }

# Common SELECT statements
SELECT_DEBATE_RECORD = f"""
    SELECT 
        {DEBATE_COLUMNS['id']},
        {DEBATE_COLUMNS['persuasion_technique']},
        {DEBATE_COLUMNS['effectiveness_score']},
        {DEBATE_COLUMNS['timestamp']},
        {DEBATE_COLUMNS['suggestion']}
    FROM {DEBATE_HISTORY_TABLE}
"""

SELECT_EFFECTIVENESS_SCORE = f"""
    SELECT 
        {EFFECTIVENESS_COLUMNS['id']},
        {EFFECTIVENESS_COLUMNS['persuasion_type']},
        {EFFECTIVENESS_COLUMNS['score']},
        {EFFECTIVENESS_COLUMNS['updated_at']}
    FROM {EFFECTIVENESS_SCORES_TABLE}
""" 
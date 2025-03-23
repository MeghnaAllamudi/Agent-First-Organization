"""Tool for storing new debate records in the database."""

from ..tools import register_tool
from .connection import DebateDBConnection
from .schema import (
    DEBATE_COLUMNS,
    DEBATE_HISTORY_TABLE,
    format_debate_record
)
from . import SLOTS
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@register_tool(
    "Stores a new debate record in the database",
    [{**SLOTS['persuasion_technique']},
     {**SLOTS['effectiveness_score']},
     {**SLOTS['suggestion']}
    ],
    [{
        "name": "store_result",
        "type": "dict",
        "description": "Result of the store operation containing the new record ID and details"
    }]
)
def store_debate_record(persuasion_technique: str, effectiveness_score: float, 
                       suggestion: str = None) -> dict:
    """Store a new debate record in the database.
    
    Args:
        persuasion_technique: Type of persuasion used
        effectiveness_score: Score from 0 to 1
        suggestion: Suggestion for improvement (optional)
        
    Returns:
        Dict containing the new record details
    """
    db = DebateDBConnection()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Validate effectiveness score
            if not 0 <= effectiveness_score <= 1:
                raise ValueError("Effectiveness score must be between 0 and 1")
            
            # Insert the record
            cursor.execute(f"""
                INSERT INTO {DEBATE_HISTORY_TABLE} 
                ({DEBATE_COLUMNS['persuasion_technique']}, 
                 {DEBATE_COLUMNS['effectiveness_score']}, 
                 {DEBATE_COLUMNS['timestamp']}, 
                 {DEBATE_COLUMNS['suggestion']})
                VALUES (?, ?, ?, ?)
            """, (persuasion_technique, effectiveness_score, 
                 datetime.now().isoformat(), suggestion))
            
            # Get the ID of the new record
            new_id = cursor.lastrowid
            
            # Fetch the inserted record
            cursor.execute(f"""
                SELECT 
                    {DEBATE_COLUMNS['id']},
                    {DEBATE_COLUMNS['persuasion_technique']},
                    {DEBATE_COLUMNS['effectiveness_score']},
                    {DEBATE_COLUMNS['timestamp']},
                    {DEBATE_COLUMNS['suggestion']}
                FROM {DEBATE_HISTORY_TABLE}
                WHERE {DEBATE_COLUMNS['id']} = ?
            """, (new_id,))
            
            # Format the record using schema formatter
            new_record = format_debate_record(cursor.fetchone())
            
            conn.commit()
            logger.info(f"Successfully stored new debate record with ID {new_id}")
            
            return {
                "status": "success",
                "message": f"Successfully stored new debate record with ID {new_id}",
                "new_record": new_record
            }
            
    except Exception as e:
        logger.error(f"Error storing debate record: {str(e)}")
        raise 
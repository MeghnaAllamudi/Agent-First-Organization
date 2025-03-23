"""Tool for updating existing debate records in the database."""

from ..tools import register_tool
from .connection import DebateDBConnection
from . import SLOTS
import json
import logging

logger = logging.getLogger(__name__)

@register_tool(
    "Updates an existing debate record in the database",
    [{**SLOTS['record_id']},
     {**SLOTS['persuasion_technique']},
     {**SLOTS['effectiveness_score']},
     {**SLOTS['suggestion']}
    ],
    [{
        "name": "update_result",
        "type": "dict",
        "description": "Result of the update operation containing status and updated record"
    }]
)
def update_debate_record(record_id: int, persuasion_technique: str = None, 
                        effectiveness_score: float = None, suggestion: str = None) -> dict:
    """Update an existing debate record in the database.
    
    Args:
        record_id: ID of the record to update
        persuasion_technique: New persuasion technique (optional)
        effectiveness_score: New effectiveness score (optional)
        suggestion: New suggestion (optional)
        
    Returns:
        Dict containing update status and the updated record
    """
    db = DebateDBConnection()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # First check if record exists
            cursor.execute(
                "SELECT id FROM debate_history WHERE id = ?",
                (record_id,)
            )
            if not cursor.fetchone():
                raise ValueError(f"No debate record found with ID {record_id}")
            
            # Build update query dynamically based on provided fields
            update_fields = []
            params = []
            
            if persuasion_technique is not None:
                update_fields.append("persuasion_technique = ?")
                params.append(persuasion_technique)
            
            if effectiveness_score is not None:
                update_fields.append("effectiveness_score = ?")
                params.append(effectiveness_score)
            
            if suggestion is not None:
                update_fields.append("suggestion = ?")
                params.append(suggestion)
            
            if not update_fields:
                raise ValueError("No fields provided for update")
            
            # Add record_id to params
            params.append(record_id)
            
            # Execute update
            query = f"""
                UPDATE debate_history 
                SET {', '.join(update_fields)}
                WHERE id = ?
            """
            cursor.execute(query, params)
            
            # Get updated record
            cursor.execute("""
                SELECT 
                    id,
                    persuasion_technique,
                    effectiveness_score,
                    timestamp,
                    suggestion
                FROM debate_history
                WHERE id = ?
            """, (record_id,))
            
            row = cursor.fetchone()
            updated_record = {
                "id": row[0],
                "persuasion_technique": row[1],
                "effectiveness_score": row[2],
                "timestamp": row[3],
                "suggestion": row[4]
            }
            
            conn.commit()
            logger.info(f"Successfully updated debate record {record_id}")
            
            return {
                "status": "success",
                "message": f"Successfully updated debate record {record_id}",
                "updated_record": updated_record
            }
            
    except Exception as e:
        logger.error(f"Error updating debate record: {str(e)}")
        raise 
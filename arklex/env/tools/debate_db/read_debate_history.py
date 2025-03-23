"""Tool for reading debate history from the database."""

from ..tools import register_tool
from .connection import DebateDBConnection
from .schema import (
    SELECT_DEBATE_RECORD,
    format_debate_record,
    DEBATE_COLUMNS
)
from . import SLOTS
import logging

logger = logging.getLogger(__name__)

@register_tool(
    "Reads debate history records from the database",
    [{**SLOTS['limit']},
     {**SLOTS['persuasion_type']}
    ],
    [{
        "name": "debate_records",
        "type": "list",
        "description": "List of debate records matching the criteria"
    }]
)
def read_debate_history(limit: int = 100, persuasion_type: str = None) -> list:
    """Read debate history records from the database.
    
    Args:
        limit: Maximum number of records to return
        persuasion_type: Filter records by persuasion type
        
    Returns:
        List of debate records
    """
    db = DebateDBConnection()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Start with base query from schema
            query = SELECT_DEBATE_RECORD
            params = []
            
            # Add persuasion type filter if specified
            if persuasion_type:
                query += f" WHERE {DEBATE_COLUMNS['persuasion_technique']} = ?"
                params.append(persuasion_type)
            
            # Add ordering and limit
            query += f" ORDER BY {DEBATE_COLUMNS['timestamp']} DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            
            # Convert results using schema formatter
            records = [format_debate_record(row) for row in cursor.fetchall()]
            
            logger.info(f"Retrieved {len(records)} debate history records")
            return records
                
    except Exception as e:
        logger.error(f"Error reading debate history: {str(e)}")
        raise 
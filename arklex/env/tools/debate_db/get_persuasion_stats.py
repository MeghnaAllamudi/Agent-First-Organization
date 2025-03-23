"""Tool for retrieving persuasion technique statistics from the database."""

from ..tools import register_tool
from .connection import DebateDBConnection
from . import SLOTS
import logging

logger = logging.getLogger(__name__)

@register_tool(
    "Gets statistics for persuasion techniques from the database",
    [{**SLOTS['technique']}],
    [{
        "name": "stats",
        "type": "dict",
        "description": "Statistics for each persuasion technique including average score and usage count"
    }]
)
def get_persuasion_stats(technique: str = None) -> dict:
    """Get statistics for persuasion techniques.
    
    Args:
        technique: Specific technique to get stats for (optional)
        
    Returns:
        Dict containing statistics for each technique
    """
    db = DebateDBConnection()
    
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query based on whether a specific technique is requested
            query = """
                SELECT 
                    persuasion_technique,
                    ROUND(AVG(effectiveness_score), 2) as avg_score,
                    COUNT(*) as usage_count,
                    MIN(effectiveness_score) as min_score,
                    MAX(effectiveness_score) as max_score
                FROM debate_history
            """
            
            params = []
            if technique:
                query += " WHERE persuasion_technique = ?"
                params.append(technique)
            
            query += " GROUP BY persuasion_technique"
            
            # Execute query
            cursor.execute(query, params)
            
            # Convert results to dictionary
            stats = {}
            for row in cursor.fetchall():
                technique_name = row[0]
                stats[technique_name] = {
                    "average_score": row[1],
                    "usage_count": row[2],
                    "min_score": row[3],
                    "max_score": row[4]
                }
            
            logger.info(f"Retrieved stats for {len(stats)} persuasion techniques")
            
            return {
                "status": "success",
                "stats": stats,
                "total_debates": sum(s["usage_count"] for s in stats.values())
            }
            
    except Exception as e:
        logger.error(f"Error getting persuasion stats: {str(e)}")
        raise 
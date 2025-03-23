"""Debate database tools for managing debate records and statistics."""

from .read_debate_history import read_debate_history
from .store_debate_record import store_debate_record
from .update_debate_record import update_debate_record
from .get_persuasion_stats import get_persuasion_stats

# Common slot definitions for all debate database tools
SLOTS = {
    # Store and update slots
    'persuasion_technique': {
        'name': 'persuasion_technique',
        'type': 'str',
        'description': 'Type of persuasion used in the argument',
        'required': True
    },
    'effectiveness_score': {
        'name': 'effectiveness_score',
        'type': 'float',
        'description': 'Score from 0 to 1 indicating effectiveness',
        'required': True
    },
    'suggestion': {
        'name': 'suggestion',
        'type': 'str',
        'description': 'Suggestion for improving the debate/persuasion technique',
        'required': False
    },
    
    # Read slots
    'limit': {
        'name': 'limit',
        'type': 'int',
        'description': 'Maximum number of records to return',
        'required': False
    },
    'persuasion_type': {
        'name': 'persuasion_type',
        'type': 'str',
        'description': 'Filter records by persuasion type',
        'required': False
    },
    
    # Update slots
    'record_id': {
        'name': 'record_id',
        'type': 'int',
        'description': 'ID of the debate record to update',
        'required': True
    },
    
    # Stats slots
    'technique': {
        'name': 'technique',
        'type': 'str',
        'description': 'Specific persuasion technique to get stats for (optional)',
        'required': False
    }
}

__all__ = [
    'read_debate_history',
    'store_debate_record',
    'update_debate_record',
    'get_persuasion_stats',
    'SLOTS'
] 
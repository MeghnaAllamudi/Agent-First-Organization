import logging
from typing import Dict, Any
from arklex.env.tools.tools import register_tool

logger = logging.getLogger(__name__)

@register_tool(
    desc="Tool for creating standardized error responses",
    slots=[{
        "name": "persuasion_type",
        "type": "string",
        "description": "Type of persuasion (pathos, logos, ethos)",
        "prompt": "Please provide the persuasion type",
        "required": True,
        "enum": ["pathos", "logos", "ethos"]
    }],
    outputs=[{
        "name": "error_response",
        "type": "object",
        "description": "Standardized error response",
        "required": True
    }]
)
class ErrorHandlingTool:
    """Tool for creating standardized error responses."""
    
    def create_error_response(self, persuasion_type: str) -> Dict[str, Any]:
        """Creates a standardized error response for failed operations."""
        return {
            "counter_argument": f"Failed to generate {persuasion_type} counter-argument",
            "techniques_used": [],
            f"{persuasion_type}_focus": "none",
            "effectiveness_score": 0.0
        } 
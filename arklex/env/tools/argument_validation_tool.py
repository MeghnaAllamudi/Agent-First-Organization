import logging
from typing import Dict, Any, List
from arklex.env.tools.tools import register_tool

logger = logging.getLogger(__name__)

@register_tool(
    desc="Tool for validating argument structure and content",
    slots=[{
        "name": "argument",
        "type": "object",
        "description": "Argument to validate",
        "prompt": "Please provide the argument to validate",
        "required": True
    }],
    outputs=[{
        "name": "is_valid",
        "type": "boolean",
        "description": "Whether the argument is valid",
        "required": True
    }]
)
class ArgumentValidationTool:
    """Tool for validating argument structure and content."""
    
    def __init__(self, required_fields: List[str], valid_techniques: List[str]):
        self.required_fields = required_fields
        self.valid_techniques = valid_techniques
    
    def validate_argument(self, argument: Dict[str, Any]) -> bool:
        """Validates the argument output format."""
        # Check required fields
        if not all(field in argument for field in self.required_fields):
            logger.error(f"Missing required fields in argument")
            return False
            
        # Validate effectiveness score
        if not 0 <= argument["effectiveness_score"] <= 1:
            logger.error("Invalid effectiveness score in argument")
            return False
            
        # Validate techniques used
        if not all(technique in self.valid_techniques for technique in argument["techniques_used"]):
            logger.error(f"Invalid techniques in argument")
            return False
            
        return True 
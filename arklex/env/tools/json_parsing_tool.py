import json
import logging
from typing import Dict, Any, Optional
from arklex.env.tools.tools import register_tool

logger = logging.getLogger(__name__)

@register_tool(
    desc="Tool for parsing JSON from text responses",
    slots=[{
        "name": "text",
        "type": "string",
        "description": "Text containing JSON to parse",
        "prompt": "Please provide the text containing JSON to parse",
        "required": True
    }],
    outputs=[{
        "name": "parsed_json",
        "type": "object",
        "description": "Parsed JSON object",
        "required": True
    }]
)
class JSONParsingTool:
    """Tool for parsing JSON from text responses."""
    
    def parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempts to parse JSON from text, with fallback to extract JSON if parsing fails."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error(f"Failed to find JSON in response: {text}")
                return None
            try:
                return json.loads(text[start_idx:end_idx])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {str(e)}")
                return None 
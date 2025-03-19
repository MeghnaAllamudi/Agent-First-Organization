import logging
import json
from typing import Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP


logger = logging.getLogger(__name__)


@register_worker
class ArgumentClassifier(BaseWorker):
    """A worker that classifies user arguments into different types."""
    
    description = "Classifies user arguments into emotional, logical, or ethical categories."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.stream_response = MODEL.get("stream_response", True)
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize the action graph
        self.action_graph = self._create_action_graph()
        
        # Set default categories if none provided
        self.categories = {
            "emotional": {
                "description": "Arguments based on feelings, emotions, and personal experiences",
                "examples": [
                    "This makes me feel very concerned",
                    "I'm worried about the impact",
                    "This is deeply troubling"
                ]
            },
            "logical": {
                "description": "Arguments based on facts, data, and reasoning",
                "examples": [
                    "The data shows a clear trend",
                    "This follows from basic principles",
                    "The evidence supports this conclusion"
                ]
            },
            "ethical": {
                "description": "Arguments based on moral principles and values",
                "examples": [
                    "This raises important ethical concerns",
                    "We have a moral obligation to",
                    "This conflicts with our values"
                ]
            }
        }
        
        self.classification_prompt = PromptTemplate.from_template(
            """Classify the following user argument into one or more categories.
            
            Categories:
            {categories}
            
            User Argument: {user_argument}
            
            Respond in the following JSON format:
            {{
                "dominant_type": "primary category (emotional/logical/ethical)",
                "secondary_types": ["list of secondary categories"],
                "confidence": float (0-1),
                "reasoning": "explanation for classification"
            }}
            
            Classification:"""
        )

    def _validate_classification(self, classification: Dict[str, Any]) -> bool:
        """Validates the classification output format."""
        required_fields = ["dominant_type", "secondary_types", "confidence", "reasoning"]
        
        # Check required fields
        if not all(field in classification for field in required_fields):
            logger.error("Missing required fields in classification")
            return False
            
        # Validate confidence score
        if not 0 <= classification["confidence"] <= 1:
            logger.error("Invalid confidence score in classification")
            return False
            
        # Validate types
        valid_types = list(self.categories.keys())
        if classification["dominant_type"] not in valid_types:
            logger.error("Invalid dominant type in classification")
            return False
            
        if not all(t in valid_types for t in classification["secondary_types"]):
            logger.error("Invalid secondary types in classification")
            return False
            
        return True

    def _classify_argument(self, user_argument: str) -> Optional[Dict[str, Any]]:
        """Classifies the user argument into categories."""
        try:
            # Format categories for prompt
            categories_str = "\n".join(
                f"- {cat}: {info['description']}\n  Examples: {', '.join(info['examples'])}"
                for cat, info in self.categories.items()
            )
            
            # Format the prompt
            formatted_prompt = self.classification_prompt.format(
                categories=categories_str,
                user_argument=user_argument
            )
            
            # Generate classification
            chain = self.llm | StrOutputParser()
            classification_str = chain.invoke(formatted_prompt)
            
            # Parse the classification
            classification = json.loads(classification_str)
            
            # Validate the classification
            if not self._validate_classification(classification):
                return None
                
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying argument: {str(e)}")
            return None

    def _create_action_graph(self):
        """Creates the action graph for argument classification."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add classification node
        workflow.add_node("classifier", self._classify_arguments)
        
        # Add edges
        workflow.add_edge(START, "classifier")
        
        return workflow

    def _classify_arguments(self, state: MessageState) -> MessageState:
        """Classifies all arguments in the state."""
        # Get user message
        user_message = state.get("user_message")
        if not user_message:
            return state
            
        # Access content directly from ConvoMessage object
        user_argument = user_message.content if hasattr(user_message, 'content') else ""
        
        if not user_argument:
            return state
            
        # Classify the argument
        classification = self._classify_argument(user_argument)
        
        if classification:
            state["argument_classification"] = classification
        
        return state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the argument classification workflow."""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result 
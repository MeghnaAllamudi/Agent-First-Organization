import logging
import json
from typing import Dict, Any, List, Optional

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
    """A worker that classifies debate arguments into different types."""
    
    description = "Analyzes and classifies debate arguments into emotional, logical, ethical, or mixed categories."

    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Default confidence threshold
        self.confidence_threshold = 0.6
        
        self.classification_prompt = PromptTemplate.from_template(
            """Analyze the following argument and classify it into one or more categories.
            Provide confidence scores for each category and explain your reasoning.
            
            Argument: {argument}
            
            Categories to consider:
            1. Emotional (pathos) - Appeals to emotions, values, beliefs
            2. Logical (logos) - Appeals to reason, facts, data
            3. Ethical (ethos) - Appeals to credibility, authority, character
            
            Respond in the following JSON format:
            {{
                "dominant_type": "emotional/logical/ethical/mixed",
                "scores": {{
                    "emotional": float (0-1),
                    "logical": float (0-1),
                    "ethical": float (0-1)
                }},
                "reasoning": "explanation of classification",
                "key_phrases": ["list of key phrases that indicate the type"],
                "confidence": float (0-1)  # Overall confidence in the classification
            }}
            
            Classification:"""
        )

    def _validate_classification(self, classification: Dict[str, Any]) -> bool:
        """Validates the classification output format and scores."""
        required_fields = ["dominant_type", "scores", "reasoning", "key_phrases", "confidence"]
        
        # Check required fields
        if not all(field in classification for field in required_fields):
            logger.error("Missing required fields in classification")
            return False
            
        # Validate scores
        scores = classification["scores"]
        if not all(0 <= score <= 1 for score in scores.values()):
            logger.error("Invalid score values in classification")
            return False
            
        # Validate confidence
        if not 0 <= classification["confidence"] <= 1:
            logger.error("Invalid confidence value in classification")
            return False
            
        # Validate dominant type
        valid_types = ["emotional", "logical", "ethical", "mixed"]
        if classification["dominant_type"] not in valid_types:
            logger.error("Invalid dominant type in classification")
            return False
            
        return True

    def _analyze_argument(self, argument: str) -> Optional[Dict[str, Any]]:
        """Analyzes a single argument and returns its classification."""
        try:
            # Format the prompt with the argument
            formatted_prompt = self.classification_prompt.format(argument=argument)
            
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
            logger.error(f"Error analyzing argument: {str(e)}")
            return None

    def _create_action_graph(self):
        """Creates the action graph for argument classification."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add classification node
        workflow.add_node("classifier", self._classify_argument)
        
        # Add edges
        workflow.add_edge(START, "classifier")
        
        return workflow

    def _classify_argument(self, state: MessageState) -> MessageState:
        """Classifies the argument in the user's message."""
        # Get the argument from the user's message
        argument = state.get("user_message", {}).get("content", "")
        
        # Analyze the argument
        classification = self._analyze_argument(argument)
        
        if classification:
            # Check confidence threshold
            if classification["confidence"] >= self.confidence_threshold:
                state["argument_classification"] = classification
                state["message_flow"] = classification
            else:
                # If confidence is too low, mark as uncertain
                classification["confidence"] = 0.0
                classification["dominant_type"] = "uncertain"
                state["argument_classification"] = classification
                state["message_flow"] = classification
        else:
            # Handle classification failure
            error_classification = {
                "dominant_type": "error",
                "scores": {"emotional": 0.0, "logical": 0.0, "ethical": 0.0},
                "reasoning": "Failed to classify argument",
                "key_phrases": [],
                "confidence": 0.0
            }
            state["argument_classification"] = error_classification
            state["message_flow"] = error_classification
        
        return state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the argument classification workflow."""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result 
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP


logger = logging.getLogger(__name__)


class PersuasionWorker(BaseWorker):
    """Abstract base class for persuasion workers."""
    
    def __init__(self, persuasion_type: str):
        super().__init__()
        self.persuasion_type = persuasion_type
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Load persuasion-specific prompt
        self.persuasion_prompt = self._get_persuasion_prompt()
        
    def _get_persuasion_prompt(self) -> PromptTemplate:
        """Get the appropriate prompt template based on persuasion type."""
        prompts = {
            "pathos": """You are an expert in emotional persuasion (pathos).
            Analyze the following argument and provide a counter-argument that appeals to emotions, values, and beliefs.
            
            Original Argument: {argument}
            Classification: {classification}
            
            Respond in the following JSON format:
            {{
                "counter_argument": "your emotional counter-argument",
                "emotional_appeals": ["list of emotional appeals used"],
                "key_phrases": ["list of key emotional phrases"],
                "effectiveness_score": float (0-1)  # Estimated effectiveness of the counter-argument
            }}
            
            Counter-Argument:""",
            
            "logos": """You are an expert in logical persuasion (logos).
            Analyze the following argument and provide a counter-argument that appeals to reason, facts, and data.
            
            Original Argument: {argument}
            Classification: {classification}
            
            Respond in the following JSON format:
            {{
                "counter_argument": "your logical counter-argument",
                "logical_appeals": ["list of logical appeals used"],
                "key_phrases": ["list of key logical phrases"],
                "effectiveness_score": float (0-1)  # Estimated effectiveness of the counter-argument
            }}
            
            Counter-Argument:""",
            
            "ethos": """You are an expert in ethical persuasion (ethos).
            Analyze the following argument and provide a counter-argument that appeals to credibility, authority, and character.
            
            Original Argument: {argument}
            Classification: {classification}
            
            Respond in the following JSON format:
            {{
                "counter_argument": "your ethical counter-argument",
                "ethical_appeals": ["list of ethical appeals used"],
                "key_phrases": ["list of key ethical phrases"],
                "effectiveness_score": float (0-1)  # Estimated effectiveness of the counter-argument
            }}
            
            Counter-Argument:"""
        }
        
        return PromptTemplate.from_template(prompts.get(self.persuasion_type, ""))

    def _validate_counter_argument(self, counter_arg: Dict[str, Any]) -> bool:
        """Validates the counter-argument output format."""
        required_fields = ["counter_argument", f"{self.persuasion_type}_appeals", 
                         "key_phrases", "effectiveness_score"]
        
        # Check required fields
        if not all(field in counter_arg for field in required_fields):
            logger.error(f"Missing required fields in {self.persuasion_type} counter-argument")
            return False
            
        # Validate effectiveness score
        if not 0 <= counter_arg["effectiveness_score"] <= 1:
            logger.error("Invalid effectiveness score in counter-argument")
            return False
            
        return True

    def _generate_counter_argument(self, argument: str, classification: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generates a counter-argument based on the persuasion type."""
        try:
            # Format the prompt
            formatted_prompt = self.persuasion_prompt.format(
                argument=argument,
                classification=classification
            )
            
            # Generate counter-argument
            chain = self.llm | StrOutputParser()
            counter_arg_str = chain.invoke(formatted_prompt)
            
            # Parse the counter-argument
            counter_arg = json.loads(counter_arg_str)
            
            # Validate the counter-argument
            if not self._validate_counter_argument(counter_arg):
                return None
                
            return counter_arg
            
        except Exception as e:
            logger.error(f"Error generating {self.persuasion_type} counter-argument: {str(e)}")
            return None

    def _create_action_graph(self):
        """Creates the action graph for persuasion."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add persuasion node
        workflow.add_node("persuasion", self._apply_persuasion)
        
        # Add edges
        workflow.add_edge(START, "persuasion")
        
        return workflow

    def _apply_persuasion(self, state: MessageState) -> MessageState:
        """Applies the persuasion strategy to the argument."""
        # Get the argument and classification
        argument = state.get("user_message", {}).get("content", "")
        classification = state.get("argument_classification", {})
        
        # Generate counter-argument
        counter_arg = self._generate_counter_argument(argument, classification)
        
        if counter_arg:
            state[f"{self.persuasion_type}_response"] = counter_arg
            state["message_flow"] = counter_arg
        else:
            # Handle generation failure
            error_response = {
                "counter_argument": f"Failed to generate {self.persuasion_type} counter-argument",
                f"{self.persuasion_type}_appeals": [],
                "key_phrases": [],
                "effectiveness_score": 0.0
            }
            state[f"{self.persuasion_type}_response"] = error_response
            state["message_flow"] = error_response
        
        return state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the persuasion workflow."""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result


@register_worker
class PathosWorker(PersuasionWorker):
    """Worker for emotional persuasion (pathos)."""
    
    description = "Generates emotional counter-arguments that appeal to feelings and values."
    
    def __init__(self):
        super().__init__("pathos")


@register_worker
class LogosWorker(PersuasionWorker):
    """Worker for logical persuasion (logos)."""
    
    description = "Generates logical counter-arguments that appeal to reason and facts."
    
    def __init__(self):
        super().__init__("logos")


@register_worker
class EthosWorker(PersuasionWorker):
    """Worker for ethical persuasion (ethos)."""
    
    description = "Generates ethical counter-arguments that appeal to credibility and character."
    
    def __init__(self):
        super().__init__("ethos") 
import logging
import json
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
    
    def __init__(self, persuasion_type: str, techniques: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.persuasion_type = persuasion_type
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize the action graph
        self.action_graph = self._create_action_graph()
        
        # Set default techniques if none provided
        self.techniques = techniques or self._get_default_techniques()
        
        # Load persuasion-specific prompt
        self.persuasion_prompt = self._get_persuasion_prompt()
        
    def _get_default_techniques(self) -> Dict[str, Any]:
        """Get default techniques based on persuasion type."""
        defaults = {
            "pathos": {
                "emotional_storytelling": {
                    "description": "Use personal stories or narratives to create emotional connection",
                    "examples": [
                        "Share a relatable experience",
                        "Describe emotional impact",
                        "Use vivid imagery"
                    ]
                },
                "value_appeals": {
                    "description": "Appeal to shared values and beliefs",
                    "examples": [
                        "Connect to core values",
                        "Highlight moral principles",
                        "Emphasize shared goals"
                    ]
                },
                "emotional_language": {
                    "description": "Use emotionally charged words and phrases",
                    "examples": [
                        "Choose impactful adjectives",
                        "Use emotional metaphors",
                        "Include feeling words"
                    ]
                },
                "empathy": {
                    "description": "Show understanding and connection with audience's perspective",
                    "examples": [
                        "Acknowledge feelings",
                        "Show shared concerns",
                        "Express understanding"
                    ]
                }
            },
            "logos": {
                "data_analysis": {
                    "description": "Use statistics, facts, and data to support arguments",
                    "examples": [
                        "Cite relevant statistics",
                        "Present research findings",
                        "Use numerical evidence"
                    ]
                },
                "logical_reasoning": {
                    "description": "Apply deductive and inductive reasoning",
                    "examples": [
                        "Use syllogisms",
                        "Apply cause-effect relationships",
                        "Make logical connections"
                    ]
                },
                "evidence_based": {
                    "description": "Support claims with concrete evidence",
                    "examples": [
                        "Reference studies",
                        "Cite expert opinions",
                        "Use case studies"
                    ]
                },
                "comparative_analysis": {
                    "description": "Compare and contrast different perspectives",
                    "examples": [
                        "Evaluate alternatives",
                        "Analyze pros and cons",
                        "Consider different scenarios"
                    ]
                }
            },
            "ethos": {
                "credibility_building": {
                    "description": "Establish trust and authority",
                    "examples": [
                        "Demonstrate expertise",
                        "Show experience",
                        "Reference credentials"
                    ]
                },
                "character_appeal": {
                    "description": "Appeal to moral character and values",
                    "examples": [
                        "Show integrity",
                        "Demonstrate honesty",
                        "Express ethical principles"
                    ]
                },
                "trustworthiness": {
                    "description": "Build trust through transparency and consistency",
                    "examples": [
                        "Acknowledge limitations",
                        "Show consistency",
                        "Be transparent"
                    ]
                },
                "authority_establishment": {
                    "description": "Establish authority through knowledge and experience",
                    "examples": [
                        "Reference expertise",
                        "Share relevant experience",
                        "Cite authoritative sources"
                    ]
                }
            }
        }
        return defaults.get(self.persuasion_type, {})
        
    def _get_persuasion_prompt(self) -> PromptTemplate:
        """Get the appropriate prompt template based on persuasion type."""
        prompts = {
            "pathos": """You are an expert in emotional persuasion (pathos).
            Analyze the following argument and provide a counter-argument that appeals to emotions, values, and beliefs.
            
            Original Argument: {argument}
            Classification: {classification}
            
            Available Techniques:
            {techniques}
            
            Respond in the following JSON format:
            {{
                "counter_argument": "your emotional counter-argument",
                "techniques_used": ["list of techniques applied"],
                "emotional_focus": "primary emotional target",
                "effectiveness_score": float (0-1)  # Estimated effectiveness of the counter-argument
            }}
            
            Counter-Argument:""",
            
            "logos": """You are an expert in logical persuasion (logos).
            Analyze the following argument and provide a counter-argument that appeals to reason, facts, and data.
            
            Original Argument: {argument}
            Classification: {classification}
            
            Available Techniques:
            {techniques}
            
            Respond in the following JSON format:
            {{
                "counter_argument": "your logical counter-argument",
                "techniques_used": ["list of techniques applied"],
                "logical_focus": "primary logical approach",
                "effectiveness_score": float (0-1)  # Estimated effectiveness of the counter-argument
            }}
            
            Counter-Argument:""",
            
            "ethos": """You are an expert in ethical persuasion (ethos).
            Analyze the following argument and provide a counter-argument that appeals to credibility, authority, and character.
            
            Original Argument: {argument}
            Classification: {classification}
            
            Available Techniques:
            {techniques}
            
            Respond in the following JSON format:
            {{
                "counter_argument": "your ethical counter-argument",
                "techniques_used": ["list of techniques applied"],
                "ethical_focus": "primary ethical appeal",
                "effectiveness_score": float (0-1)  # Estimated effectiveness of the counter-argument
            }}
            
            Counter-Argument:"""
        }
        
        return PromptTemplate.from_template(prompts.get(self.persuasion_type, ""))

    def _validate_counter_argument(self, counter_arg: Dict[str, Any]) -> bool:
        """Validates the counter-argument output format."""
        required_fields = ["counter_argument", "techniques_used", 
                         f"{self.persuasion_type}_focus", "effectiveness_score"]
        
        # Check required fields
        if not all(field in counter_arg for field in required_fields):
            logger.error(f"Missing required fields in {self.persuasion_type} counter-argument")
            return False
            
        # Validate effectiveness score
        if not 0 <= counter_arg["effectiveness_score"] <= 1:
            logger.error("Invalid effectiveness score in counter-argument")
            return False
            
        # Validate techniques used
        valid_techniques = list(self.techniques.keys())
        if not all(technique in valid_techniques for technique in counter_arg["techniques_used"]):
            logger.error(f"Invalid techniques in {self.persuasion_type} counter-argument")
            return False
            
        return True

    def _generate_counter_argument(self, argument: str, classification: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generates a counter-argument based on the persuasion type."""
        try:
            # Format techniques for prompt
            techniques_str = "\n".join(
                f"- {technique}: {info['description']}\n"
                f"  Examples: {', '.join(info['examples'])}"
                for technique, info in self.techniques.items()
            )
            
            # Format the prompt
            formatted_prompt = self.persuasion_prompt.format(
                argument=argument,
                classification=json.dumps(classification, indent=2),
                techniques=techniques_str
            )
            
            # Generate counter-argument
            chain = self.llm | StrOutputParser()
            counter_arg_str = chain.invoke(formatted_prompt)
            
            # Try to parse the counter-argument
            try:
                counter_arg = json.loads(counter_arg_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                start_idx = counter_arg_str.find("{")
                end_idx = counter_arg_str.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    logger.error(f"Failed to find JSON in response: {counter_arg_str}")
                    return None
                try:
                    counter_arg = json.loads(counter_arg_str[start_idx:end_idx])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from response: {str(e)}")
                    return None
            
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
        user_message = state.get("user_message")
        if not user_message:
            return state
            
        argument = user_message.content if hasattr(user_message, 'content') else ""
        classification = state.get("argument_classification", {})
        
        # Generate counter-argument
        counter_arg = self._generate_counter_argument(argument, classification)
        
        if counter_arg:
            state[f"{self.persuasion_type}_response"] = counter_arg
            # Ensure message_flow is a string
            state["message_flow"] = counter_arg.get("counter_argument", "")
        else:
            # Handle generation failure
            error_response = {
                "counter_argument": f"Failed to generate {self.persuasion_type} counter-argument",
                "techniques_used": [],
                f"{self.persuasion_type}_focus": "none",
                "effectiveness_score": 0.0
            }
            state[f"{self.persuasion_type}_response"] = error_response
            state["message_flow"] = error_response.get("counter_argument", "")
        
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
    
    def __init__(self, techniques: Optional[Dict[str, Any]] = None):
        super().__init__("pathos", techniques)


@register_worker
class LogosWorker(PersuasionWorker):
    """Worker for logical persuasion (logos)."""
    
    description = "Generates logical counter-arguments that appeal to reason and facts."
    
    def __init__(self, techniques: Optional[Dict[str, Any]] = None):
        super().__init__("logos", techniques)


@register_worker
class EthosWorker(PersuasionWorker):
    """Worker for ethical persuasion (ethos)."""
    
    description = "Generates ethical counter-arguments that appeal to credibility and character."
    
    def __init__(self, techniques: Optional[Dict[str, Any]] = None):
        super().__init__("ethos", techniques) 
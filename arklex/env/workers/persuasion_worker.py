import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.tools import ToolFactory, ArgumentValidationTool, JSONParsingTool, TechniqueFormattingTool, ErrorHandlingTool
from arklex.utils.database import DebateDatabase


logger = logging.getLogger(__name__)


@register_worker
class PersuasionWorker(BaseWorker):
    """A worker that generates counter-arguments using different persuasion techniques."""
    
    description = "Generates counter-arguments using pathos, logos, or ethos persuasion techniques."
    
    # Define prompts for each persuasion type
    PROMPTS = {
        "pathos": PromptTemplate.from_template(
            """You are an expert in emotional persuasion (pathos). Your task is to generate a strong counter-argument that OPPOSES the given argument by appealing to emotions, values, and beliefs.

Original Argument: {argument}
Classification: {classification}

Available Techniques:
{techniques}

Important Guidelines:
1. You MUST take the opposing position to the original argument. Do not weigh both sides.
2. Your response should leverage database recommendations for the most effective persuasion strategy.
3. Track effectiveness metrics to improve future persuasion attempts.

You MUST respond with a valid JSON object containing these EXACT fields:
{{
    "counter_argument": "your direct opposing argument using emotional appeals",
    "techniques_used": ["list of specific techniques used from the available techniques"],
    "pathos_focus": "primary emotion targeted",
    "effectiveness_score": 0.95
}}

Example response:
{{
    "counter_argument": "While social media can spread news quickly, it often spreads misinformation and causes emotional distress through constant exposure to negative content",
    "techniques_used": ["emotional_storytelling", "emotional_language"],
    "pathos_focus": "anxiety and stress",
    "effectiveness_score": 0.95
}}"""
        ),
        
        "logos": PromptTemplate.from_template(
            """You are an expert in logical persuasion (logos). Your task is to generate a strong counter-argument that OPPOSES the given argument using logic, facts, and data.

Original Argument: {argument}
Classification: {classification}

Available Techniques:
{techniques}

Important Guidelines:
1. You MUST take the opposing position to the original argument. Do not weigh both sides.
2. Your response should leverage database recommendations for the most effective persuasion strategy.
3. Track effectiveness metrics to improve future persuasion attempts.

You MUST respond with a valid JSON object containing these EXACT fields:
{{
    "counter_argument": "your direct opposing argument using logical reasoning",
    "techniques_used": ["list of specific techniques used from the available techniques"],
    "logos_focus": "primary logical approach used",
    "effectiveness_score": 0.95
}}

Example response:
{{
    "counter_argument": "Studies show that rapid news spread on social media leads to increased misinformation and decreased fact-checking, undermining its societal benefits",
    "techniques_used": ["data_analysis", "evidence_based"],
    "logos_focus": "causal analysis",
    "effectiveness_score": 0.95
}}"""
        ),
        
        "ethos": PromptTemplate.from_template(
            """You are an expert in ethical persuasion (ethos). Your task is to generate a strong counter-argument that OPPOSES the given argument by appealing to credibility, authority, and ethical principles.

Original Argument: {argument}
Classification: {classification}

Available Techniques:
{techniques}

Important Guidelines:
1. You MUST take the opposing position to the original argument. Do not weigh both sides.
2. Your response should leverage database recommendations for the most effective persuasion strategy.
3. Track effectiveness metrics to improve future persuasion attempts.

You MUST respond with a valid JSON object containing these EXACT fields:
{{
    "counter_argument": "your direct opposing argument using ethical appeals",
    "techniques_used": ["list of specific techniques used from the available techniques"],
    "ethos_focus": "primary ethical principle invoked",
    "effectiveness_score": 0.95
}}

Example response:
{{
    "counter_argument": "The rapid spread of news on social media often compromises journalistic integrity and ethical reporting standards, leading to a decline in public trust",
    "techniques_used": ["credibility_building", "character_appeal"],
    "ethos_focus": "journalistic integrity",
    "effectiveness_score": 0.95
}}"""
        )
    }
    
    # Define techniques for each persuasion type
    TECHNIQUES = {
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PersuasionWorker.
        
        Args:
            config: Optional configuration dictionary from the worker config
        """
        super().__init__()
        self.config = config or {}
        
        # Set a default persuasion type
        self.persuasion_type = "logos"  # Default to logos as safest option
        
        # Get custom techniques if defined in config
        if isinstance(self.config, dict) and isinstance(self.config.get("techniques"), dict):
            self.custom_techniques = self.config["techniques"]
        else:
            self.custom_techniques = None
            
        # Initialize LLM
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize tools
        self.tools = {
            "json_tool": JSONParsingTool(),
            "validation_tool": ArgumentValidationTool(
                required_fields=["counter_argument", "techniques_used", "logos_focus", "effectiveness_score"],
                valid_techniques=list(self.TECHNIQUES["logos"].keys())
            ),
            "error_tool": ErrorHandlingTool(),
            "technique_tool": TechniqueFormattingTool()
        }
        
        # Initialize the action graph
        self.action_graph = self._create_action_graph()

    def _determine_persuasion_type(self, msg_state: MessageState) -> str:
        """Determine which persuasion type to use based on the state."""
        # Get classification and determine persuasion type
        classification = msg_state.get("argument_classification", {})
        if not isinstance(classification, dict):
            classification = {}
            
        # First check slots from the database worker (following DatabaseWorker pattern)
        slots = msg_state.get("slots", {})
        best_technique_from_slots = None
        
        if slots and "best_technique" in slots:
            best_technique_from_slots = slots["best_technique"]
            if best_technique_from_slots in ["pathos", "logos", "ethos"]:
                logger.info(f"📦 Found best_technique in slots: {best_technique_from_slots}")
        
        # Log persuasion scores if available
        persuasion_scores = msg_state.get("persuasion_scores", {})
        if persuasion_scores:
            logger.info("🔍 CURRENT EFFECTIVENESS SCORES:")
            for p_type, score in persuasion_scores.items():
                logger.info(f"  - {p_type.upper()}: {score:.2f}")
        
        # Check persuasion type selection priority:
        # 1. First check if we have a best_persuasion_type from the DebateDatabaseWorker
        best_persuasion_type = msg_state.get("best_persuasion_type") or best_technique_from_slots
        if best_persuasion_type and best_persuasion_type in ["pathos", "logos", "ethos"]:
            logger.info(f"💡 Found best_persuasion_type in state: {best_persuasion_type}")
            persuasion_type = best_persuasion_type
            
            # Get confidence if available
            confidence = msg_state.get("recommendation_confidence", 0)
            confidence_str = f" (confidence: {confidence:.2f})" if confidence else ""
            logger.info(f"✅ USING DATABASE RECOMMENDATION: {persuasion_type.upper()} persuasion{confidence_str}")
            
            return persuasion_type
        
        # 2. Otherwise use classification to determine type
        else:
            # Log that we're falling back to classification because no recommendation was found
            logger.warning("⚠️ No best_persuasion_type found in state")
            logger.warning(f"⚠️ Available state keys: {list(msg_state.keys())}")
            
            # Default to a type based on the classification
            if isinstance(classification, dict):
                persuasion_type = "logos"  # Default to logos
                
                # Try to determine based on classification
                if classification.get("type") == "emotional":
                    persuasion_type = "logos"  # Counter emotional with logical
                elif classification.get("type") == "logical":
                    persuasion_type = "pathos"  # Counter logical with emotional
                elif classification.get("type") == "ethical":
                    persuasion_type = "ethos"  # Counter ethical with ethical
            else:
                # If no classification, default to logos
                persuasion_type = "logos"
                
            logger.info(f"✅ USING CLASSIFICATION-BASED FALLBACK: {persuasion_type.upper()} persuasion")
            return persuasion_type

    def execute(self, state: MessageState) -> MessageState:
        """Execute the worker to generate a persuasive counter-argument."""
        logger.info("PersuasionWorker executing...")
        
        # Process the orchestrator message if available
        orchestrator_message = state.get("orchestrator_message")
        
        # Determine which persuasion type to use
        persuasion_type = self.persuasion_type  # Default
        
        # Try to get from orchestrator message first
        if orchestrator_message and hasattr(orchestrator_message, "attribute"):
            if hasattr(orchestrator_message.attribute, "get"):
                attr_persuasion_type = orchestrator_message.attribute.get("persuasion_type")
                if attr_persuasion_type:
                    logger.info(f"Using persuasion type from orchestrator: {attr_persuasion_type}")
                    persuasion_type = attr_persuasion_type
        
        # Try to get from database recommendation
        if "best_persuasion_type" in state:
            persuasion_type = state["best_persuasion_type"]
            logger.info(f"Using persuasion type from database: {persuasion_type}")
        
        # IMPORTANT: Set persuasion type in state for evaluator and database worker to find
        state["current_persuasion_type"] = persuasion_type
        state["just_used_persuasion_type"] = persuasion_type
        logger.info(f"🔥 Set current_persuasion_type in state: {persuasion_type}")
        
        # Extract user message
        user_message = state.get("user_message", {})
        user_content = user_message.get("content", "") if isinstance(user_message, dict) else str(user_message)
        
        # Extract classification if available
        argument_classification = state.get("argument_classification", {})
        classification = argument_classification.get("classification", "logical")
        
        # Generate the persuasive response
        response = self._generate_persuasion_response(user_content, classification, persuasion_type)
        
        # Store the response in a type-specific key for the evaluator to find
        state[f"{persuasion_type}_persuasion_response"] = response
        logger.info(f"Created {persuasion_type}_persuasion_response in state")
        
        # Set the message flow (this is what actually gets returned to the user)
        state["message_flow"] = response.get("counter_argument", "I disagree with your position.")
        
        # Set status
        state["status"] = "success"
        
        return state

    def _generate_persuasion_response(self, argument: str, classification: str, persuasion_type: str) -> Dict[str, Any]:
        """Generate a persuasive counter-argument using the specified technique."""
        logger.info(f"Generating {persuasion_type} persuasion response...")
        
        try:
            # Get the appropriate techniques list
            techniques = self._get_techniques(persuasion_type)
            
            # Format the techniques for the prompt
            techniques_formatted = "\n".join([
                f"- {name}: {details['description']}\n  Examples: {', '.join(details['examples'])}"
                for name, details in techniques.items()
            ])
            
            # Get the prompt for the selected persuasion type
            prompt = self.PROMPTS.get(persuasion_type, self.PROMPTS["logos"])
            
            # Format the prompt
            formatted_prompt = prompt.format(
                argument=argument,
                classification=classification,
                techniques=techniques_formatted
            )
            
            # Generate response
            chain = self.llm | StrOutputParser()
            response_str = chain.invoke(formatted_prompt)
            
            # Parse the JSON response
            try:
                # First clean up any markdown code block indicators
                cleaned_response = response_str
                if "```json" in response_str:
                    cleaned_response = response_str.replace("```json", "").replace("```", "").strip()
                
                response = json.loads(cleaned_response)
                
                # Add metadata
                response["persuasion_type"] = persuasion_type
                response["timestamp"] = datetime.now().isoformat()
                
                logger.info(f"Generated {persuasion_type} counter-argument successfully")
                return response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {response_str}")
                logger.error(f"JSON decode error: {str(e)}")
                return {
                    "counter_argument": "I disagree with your position, but I'm having trouble articulating my counter-argument.",
                    "techniques_used": [],
                    f"{persuasion_type}_focus": "general",
                    "persuasion_type": persuasion_type,
                    "effectiveness_score": 0.5
                }
                
        except Exception as e:
            logger.error(f"Error generating persuasion response: {str(e)}")
            return {
                "counter_argument": "I disagree with your position, but I'm experiencing some technical difficulties.",
                "techniques_used": [],
                f"{persuasion_type}_focus": "error",
                "persuasion_type": persuasion_type,
                "effectiveness_score": 0.5
            }

    def _get_techniques(self, persuasion_type: str) -> Dict[str, Dict[str, Any]]:
        """Get the techniques for a given persuasion type."""
        if persuasion_type in self.TECHNIQUES:
            return self.TECHNIQUES[persuasion_type]
        else:
            logger.error(f"Unknown persuasion type: {persuasion_type}")
            return self.TECHNIQUES["logos"]

    def _create_action_graph(self) -> StateGraph:
        """Creates the action graph for persuasion handling."""
        workflow = StateGraph(MessageState)
        workflow.add_node("persuasion_handler", self.execute)
        workflow.add_edge(START, "persuasion_handler")
        return workflow 
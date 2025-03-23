import logging
import json
import re
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
# Remove the general prompts import if it exists
# from arklex.env.prompts import load_prompts

# Import from prompts_for_debate_opp
import importlib.util
import os
import sys

# Get the path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, root_dir)

# Import the prompts from prompts_for_debate_opp.py
spec = importlib.util.spec_from_file_location("debate_prompts", os.path.join(root_dir, "prompts_for_debate_opp.py"))
debate_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debate_prompts)

logger = logging.getLogger(__name__)


@register_worker
class EffectivenessEvaluator(BaseWorker):
    """A worker that evaluates the effectiveness of counter-arguments."""
    
    description = "Evaluates the effectiveness of counter-arguments based on multiple criteria."
    
    # Class-level variables to track conversation turns and prevent duplicate evaluations
    _conversation_turns = 0
    _has_user_stated_position = False
    _last_evaluated_turn = 0  # Track the last turn we evaluated to prevent duplicate calls
    _last_user_message_id = None  # Track the ID of the last user message we evaluated
    _execution_count = 0  # Track how many times execute is called
    _last_evaluated_content = None  # Track the last user content we evaluated
    # Track which requests have been processed (keyed by request ID)
    _processed_requests = set()
    
    # Standardized weights for criteria - defined once and used throughout
    STANDARD_CRITERIA = {
        "relevance": {
            "weight": 0.25,
            "description": "How directly the counter-argument addresses the original points",
        },
        "persuasiveness": {
            "weight": 0.35,
            "description": "How convincing the counter-argument is",
        },
        "credibility": {
            "weight": 0.25,
            "description": "How well-supported the counter-argument is",
        },
        "emotional_impact": {
            "weight": 0.15,
            "description": "How well the counter-argument connects emotionally",
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.name = "EffectivenessEvaluator"
        self.description = "Evaluates the effectiveness of persuasion strategies"
        self.config = config or {}
        
        # Setup the LLM
        self.stream_response = MODEL.get("stream_response", True)
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Extract criteria from config if provided
        if config and "criteria" in config:
            self.criteria = config["criteria"]
        else:
            # Use the standardized criteria defined at class level
            self.criteria = self.STANDARD_CRITERIA
            
            # Add examples to each criterion
            self.criteria["relevance"]["examples"] = [
                "Directly responds to the main point",
                "Addresses key concerns",
                "Stays focused on the topic"
            ]
            
            self.criteria["persuasiveness"]["examples"] = [
                "Uses compelling evidence",
                "Presents clear reasoning",
                "Builds a strong case"
            ]
            
            self.criteria["credibility"]["examples"] = [
                "Cites reliable sources",
                "Uses expert opinions",
                "Presents verifiable facts"
            ]
            
            self.criteria["emotional_impact"]["examples"] = [
                "Appeals to shared values",
                "Evokes appropriate emotional response",
                "Creates connection with audience"
            ]
        
        # Validate weights sum approximately to 1
        total_weight = sum(criterion["weight"] for criterion in self.criteria.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Criteria weights sum to {total_weight}, which is not exactly 1.0. Adjusting weights.")
            # Normalize weights to sum to 1.0
            factor = 1.0 / total_weight
            for criterion in self.criteria.values():
                criterion["weight"] = criterion["weight"] * factor
            logger.info(f"Weights normalized. New sum: {sum(criterion['weight'] for criterion in self.criteria.values())}")
            
        # Define the evaluation prompt template
        self.evaluation_prompt = PromptTemplate.from_template(
            """You are an expert at evaluating the effectiveness of debate counter-arguments. Your task is to evaluate a counter-argument's effectiveness based on specific criteria.
            
Original Argument: {original_argument}

Counter-Argument: {counter_argument}

Persuasion Type Used: {persuasion_type}

Evaluation Criteria: {criteria}

IMPORTANT CONTEXT ABOUT PERSUASION TYPE:
- If the persuasion type is "pathos", focus on emotional impact, storytelling, and personal connection. Weaknesses should relate to insufficient emotional appeal, lack of storytelling, or missing personal connection.
- If the persuasion type is "logos", focus on logic, evidence, and reasoning. Weaknesses should relate to insufficient evidence, flawed reasoning, or lack of data.
- If the persuasion type is "ethos", focus on credibility, expertise, and ethical principles. Weaknesses should relate to insufficient authority, lack of ethical grounding, or missing credibility markers.

YOUR EVALUATION MUST FOCUS ON WEAKNESSES AND SUGGESTIONS THAT ARE RELEVANT TO THE PERSUASION TYPE BEING USED.

IMPORTANT: You MUST respond with a valid JSON object containing the following fields:
- criteria_scores: An object with scores (0.0 to 1.0) for each criterion
- strengths: Array of key strengths relevant to the persuasion type
- weaknesses: Array of key weaknesses SPECIFIC to the persuasion type
- improvement_suggestions: Array of suggestions for improvement SPECIFIC to the persuasion type
- reasoning: String explaining your evaluation

EXAMPLE RESPONSE FORMAT:
{{
  "criteria_scores": {{
    "relevance": 0.85,
    "persuasiveness": 0.75,
    "credibility": 0.80,
    "emotional_impact": 0.65
  }},
  "strengths": [
    "Directly addresses the main claim",
    "Uses strong evidence to support points"
  ],
  "weaknesses": [
    "Could be more concise",
    "Some supporting points lack depth"
  ],
  "improvement_suggestions": [
    "Add more specific data points",
    "Strengthen emotional appeal to values"
  ],
  "reasoning": "The counter-argument effectively challenges the original position with strong evidence but could improve emotional engagement."
}}

YOUR EVALUATION MUST BE VALID JSON. Do not include any text before or after the JSON object. Do not use markdown formatting."""
        )

    def execute(self, state: MessageState) -> MessageState:
        """Evaluate the effectiveness of the user's argument and generate suggestions."""
        print("\n================================================================================")
        print(f"📊 EFFECTIVENESS EVALUATOR EXECUTING")
        print(f"================================================================================\n")
        
        # Use the helper method to ensure minimal state exists
        state = self._ensure_minimal_state(state)
        
        # Print available state keys for debugging
        print(f"🔑 AVAILABLE STATE KEYS: {list(state.keys())}")
        print(f"🔑 GLOBAL STATE KEYS: {list(state['metadata']['global_state'].keys())}")
        
        # Get request ID from metadata
        request_id = None
        if "chat_id" in state["metadata"] and "turn_id" in state["metadata"]:
            # Create a unique request ID combining chat_id and turn_id
            request_id = f"{state['metadata']['chat_id']}_{state['metadata']['turn_id']}"
            print(f"📝 Request ID: {request_id}")
            
            # CRITICAL: Check if this request has already been processed
            if request_id in EffectivenessEvaluator._processed_requests:
                print(f"🔄 SKIPPING: This request has already been processed (Request ID: {request_id})")
                print(f"📊 EFFECTIVENESS EVALUATOR COMPLETED (SKIPPED)")
                print(f"================================================================================\n")
                return state
        
        # Get user message content for deduplication
        user_content = None
        if "user_message" in state:
            user_msg = state["user_message"]
            if hasattr(user_msg, 'content'):
                user_content = user_msg.content
            elif isinstance(user_msg, dict) and 'content' in user_msg:
                user_content = user_msg['content']
            else:
                user_content = str(user_msg)
        
        # CRITICAL: Skip processing if we've already evaluated this exact content
        if user_content and user_content == EffectivenessEvaluator._last_evaluated_content:
            print("🔄 SKIPPING: Content already evaluated in this session")
            print(f"📊 EFFECTIVENESS EVALUATOR COMPLETED (SKIPPED)")
            print(f"================================================================================\n")
            return state
            
        # CRITICAL: Track number of executions and store content
        EffectivenessEvaluator._execution_count += 1
        if user_content:
            EffectivenessEvaluator._last_evaluated_content = user_content
        print(f"🧮 EFFECTIVENESS EVALUATOR EXECUTION COUNT: {EffectivenessEvaluator._execution_count}")
        
        # Check for argument classification data
        if "argument_classification" in state:
            print(f"✅ FOUND ARGUMENT CLASSIFICATION DATA")
            arg_class = state["argument_classification"]
            if isinstance(arg_class, dict):
                for key, value in arg_class.items():
                    print(f"   - {key}: {value}")
        else:
            print(f"⚠️ NO ARGUMENT CLASSIFICATION FOUND - User message may not have been classified")
        
        # Get trajectory and count actual messages
        trajectory = state.get("trajectory", [])
        message_count = len(trajectory)
        
        print(f"🔄 Current message count: {message_count}")
        
        # Check for user argument classification - if it exists, user has stated position
        user_has_position = False
        if "argument_classification" in state:
            user_has_position = True
            print(f"✅ User has stated a position (found argument_classification)")
        elif "user_message" in state:
            user_content = None
            user_msg = state["user_message"]
            if hasattr(user_msg, 'content'):
                user_content = user_msg.content
            elif isinstance(user_msg, dict) and 'content' in user_msg:
                user_content = user_msg['content']
            else:
                user_content = str(user_msg)
                
            if user_content and len(user_content) > 50:
                user_has_position = True
                print(f"✅ User position inferred from substantial message length ({len(user_content)} chars)")
                
        print(f"User has position: {user_has_position}")
        
        # If user has a position and we have a counter-argument to evaluate
        if user_has_position and "bot_message" in state:
            print(f"✅ User has position and counter-argument exists - proceeding with evaluation")
            try:
                # Get user_persuasion_type, prioritizing direct state value
                user_persuasion_type = state.get("user_persuasion_type")
                print(f"📊 User persuasion type: {user_persuasion_type}")
                
                # If user used pathos (emotional arguments) verify and note it
                if user_persuasion_type == "pathos":
                    print(f"🔥 USER USED PATHOS ARGUMENTS - THIS IS SIGNIFICANT FOR PERSUASIVE RESPONSE")
                    
                    # Special handling for pathos arguments
                    # Call the action graph for extended evaluation
                    result = self._evaluate_arguments(state)
                    
                    # Update the state with our special effectiveness evaluation
                    state.update(result)
                    
                    # Extract the evaluated effectiveness score
                    effectiveness_score = result.get("evaluated_effectiveness_score", 0.0)
                    print(f"📊 EVALUATED EFFECTIVENESS SCORE: {effectiveness_score:.2f}")
                    
                    # For now, remove explicit database worker call
                    # This will let the taskgraph handle the worker sequence
                    
                    # CRITICAL: Mark this request as processed if we have a request ID
                    if request_id:
                        EffectivenessEvaluator._processed_requests.add(request_id)
                        print(f"📝 Marked request {request_id} as processed")
                    
                    print(f"📊 EFFECTIVENESS EVALUATOR COMPLETED")
                    print(f"================================================================================\n")
                    
                    return state
            except Exception as e:
                print(f"⚠️ ERROR in pathos evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Normal path for non-pathos arguments
        try:
            # Get argument classification from the argument classifier
            argument_classification = state.get("argument_classification", {})
            
            # Get user message
            user_message = state.get("user_message", {})
            if hasattr(user_message, 'content'):
                user_content = user_message.content
            elif isinstance(user_message, dict) and 'content' in user_message:
                user_content = user_message['content']
            else:
                user_content = str(user_message)
            
            # Generate the effectiveness evaluation
            evaluation_result = self._evaluate_effectiveness(user_content, argument_classification)
            
            if evaluation_result:
                # Update the state with the evaluation
                for key, value in evaluation_result.items():
                    state[key] = value
                
                # Extract the evaluated effectiveness score
                effectiveness_score = evaluation_result.get("evaluated_effectiveness_score", 0.0)
                print(f"📊 EVALUATED EFFECTIVENESS SCORE: {effectiveness_score:.2f}")
                
                # Prepare data for database update (but don't call the worker directly)
                # The database worker will use this data in the next step of the taskgraph
                if "bot_message" in state and hasattr(state["bot_message"], "content"):
                    counter_argument = state["bot_message"].content
                    persuasion_type = evaluation_result.get("current_persuasion_type", user_persuasion_type or "logos")
                    response_key = f"{persuasion_type}_persuasion_response"
                    state[response_key] = {
                        "counter_argument": counter_argument,
                        "effectiveness_score": evaluation_result["evaluated_effectiveness_score"]
                    }
                    print(f"📊 PREPARED {persuasion_type.upper()} DATA FOR DATABASE UPDATE")
            else:
                print(f"⚠️ NO EVALUATION RESULT GENERATED")
        except Exception as e:
            print(f"⚠️ ERROR in standard evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # CRITICAL: Mark this request as processed if we have a request ID
        if request_id:
            EffectivenessEvaluator._processed_requests.add(request_id)
            print(f"📝 Marked request {request_id} as processed")
        
        print(f"📊 EFFECTIVENESS EVALUATOR COMPLETED")
        print(f"================================================================================\n")
        
        return state
    
    def _generate_pathos_specific_evaluation(self, state: MessageState) -> Dict:
        """Generate a pathos-specific effectiveness evaluation for emotional arguments."""
        try:
            # Get user message
            user_message = state.get("user_message", {})
            if hasattr(user_message, 'content'):
                user_content = user_message.content
            elif isinstance(user_message, dict) and 'content' in user_message:
                user_content = user_message['content']
            else:
                user_content = str(user_message)
                
            # Create a pathos-focused evaluation prompt
            prompt = f"""As an expert in emotional persuasion (pathos), analyze this emotional argument:

"{user_content}"

Identify 3 specific emotional weaknesses that could be addressed with MORE emotional storytelling.
Focus ONLY on how to make the argument MORE emotionally compelling, NOT more logical or credible.

Return your analysis as JSON:
{{
  "weaknesses": [
    "Emotional weakness 1: Not enough personal connection or vulnerability",
    "Emotional weakness 2: Lacks vivid emotional imagery that triggers strong feelings",
    "Emotional weakness 3: Missing emotional contrast between pain and relief"
  ],
  "persuasion_suggestion": "Use deeply personal stories with vivid emotional details",
  "persuasion_scores": {{
    "pathos": 0.9,
    "logos": 0.3,
    "ethos": 0.4
  }},
  "recommended_persuasion_type": "pathos"
}}
"""
            
            # Generate evaluation
            final_chain = self.llm | StrOutputParser()
            evaluation_result = final_chain.invoke(prompt)
            
            try:
                # Parse the result as JSON
                pathos_report = json.loads(evaluation_result)
                print(f"✅ Generated pathos-specific evaluation")
                return pathos_report
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse pathos evaluation as JSON")
                # Return a default pathos report
                return {
                    "weaknesses": [
                        "Emotional weakness: Not enough personal storytelling",
                        "Emotional weakness: Lacks vivid emotional imagery",
                        "Emotional weakness: Missing emotional connection with audience"
                    ],
                    "persuasion_suggestion": "Use deeply personal stories with vivid emotional details",
                    "persuasion_scores": {
                        "pathos": 0.9,
                        "logos": 0.3,
                        "ethos": 0.4
                    },
                    "recommended_persuasion_type": "pathos"
                }
        except Exception as e:
            print(f"⚠️ Error generating pathos evaluation: {str(e)}")
            # Return a default pathos report
            return {
                "weaknesses": [
                    "Emotional weakness: Not enough personal storytelling",
                    "Emotional weakness: Lacks vivid emotional imagery",
                    "Emotional weakness: Missing emotional connection with audience"
                ],
                "persuasion_suggestion": "Use deeply personal stories with vivid emotional details",
                "persuasion_scores": {
                    "pathos": 0.9,
                    "logos": 0.3,
                    "ethos": 0.4
                },
                "recommended_persuasion_type": "pathos"
            }
    
    def _generate_pathos_suggestions(self, state: MessageState) -> List[str]:
        """Generate pathos-specific suggestions based on the user's emotional appeal.
        Generates suggestions that will be useful for the PersuasionWorker.
        """
        try:
            # Extract user message
            user_message = state.get("user_message", {})
            if hasattr(user_message, 'content'):
                user_content = user_message.content
            elif isinstance(user_message, dict) and 'content' in user_message:
                user_content = user_message['content']
            else:
                user_content = str(user_message)
            
            # Generate pathos-focused suggestions
            suggestions = [
                {
                    "type": "pathos",
                    "description": "Use emotional storytelling to connect with the user's values",
                    "example": "Share a personal story about how this issue has emotionally affected you or someone close to you"
                },
                {
                    "type": "pathos",
                    "description": "Use vivid emotional language and imagery",
                    "example": "Describe emotional experiences with sensory details that create strong feelings"
                },
                {
                    "type": "pathos",
                    "description": "Appeal to shared values and emotional triggers",
                    "example": "Connect to core values like family, freedom, or security that resonate emotionally"
                }
            ]
            
            print(f"✅ Generated pathos-specific suggestions:")
            for i, suggestion in enumerate(suggestions):
                print(f"   {i+1}. {suggestion['description']}")
            
            # Store suggestions in both state locations for maximum accessibility
            self._store_suggestions_in_state(None, suggestions)
            return suggestions
        except Exception as e:
            print(f"⚠️ ERROR generating pathos suggestions: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return default suggestions on error
            default_suggestions = [
                {"type": "pathos", "description": "Use emotional storytelling", "example": "Share a personal story"}
            ]
            
            # Store suggestions in both state locations for maximum accessibility
            self._store_suggestions_in_state(None, default_suggestions)
            return default_suggestions

    def _evaluate_effectiveness(self, user_content, argument_classification):
        """Evaluate the effectiveness of counter-arguments and generate recommendations.
        
        Args:
            user_content: The user's message content
            argument_classification: Classification data for the argument
            
        Returns:
            Dictionary containing evaluation results, persuasion scores, and recommendations
        """
        try:
            # Determine the user's persuasion type from argument classification
            user_persuasion_type = "logos"  # Default to logos
            
            if isinstance(argument_classification, dict) and "dominant_type" in argument_classification:
                dominant_type = argument_classification.get("dominant_type")
                type_mapping = {
                    "emotional": "pathos",
                    "logical": "logos",
                    "ethical": "ethos"
                }
                if dominant_type in type_mapping:
                    user_persuasion_type = type_mapping[dominant_type]
                    print(f"🔍 Determined user persuasion type: {user_persuasion_type.upper()} (from {dominant_type})")
            
            # Generate a simulated evaluation result
            # In a real implementation, this would call the LLM to evaluate
            evaluation_result = {
                "weaknesses": [
                    "Could be more focused on key points",
                    "Lacks sufficient supporting evidence",
                    "Emotional connection could be stronger"
                ],
                "persuasion_suggestion": "Consider using more concrete examples to strengthen your argument",
                "persuasion_scores": {
                    "pathos": 0.5,
                    "logos": 0.7,
                    "ethos": 0.6
                }
            }
            
            # Adjust scores based on the determined persuasion type
            # Ensure the user's primary persuasion type scores highest
            if user_persuasion_type == "pathos":
                evaluation_result["persuasion_scores"]["pathos"] = max(0.8, evaluation_result["persuasion_scores"].get("pathos", 0))
            elif user_persuasion_type == "logos":
                evaluation_result["persuasion_scores"]["logos"] = max(0.8, evaluation_result["persuasion_scores"].get("logos", 0))
            elif user_persuasion_type == "ethos":
                evaluation_result["persuasion_scores"]["ethos"] = max(0.8, evaluation_result["persuasion_scores"].get("ethos", 0))
            
            print(f"✅ Generated effectiveness evaluation")
            return evaluation_result
            
        except Exception as e:
            print(f"⚠️ Error in _evaluate_effectiveness: {str(e)}")
            traceback.print_exc()
            
            # Return a default evaluation in case of errors
            return {
                "weaknesses": ["Argument could be strengthened with more supporting evidence"],
                "persuasion_suggestion": "Consider adding more concrete examples",
                "persuasion_scores": {
                    "pathos": 0.5,
                    "logos": 0.6,
                    "ethos": 0.5
                }
            }

    def _generate_suggestions(self, evaluation_result, persuasion_type):
        """Generate specific suggestions based on the evaluation results and persuasion type.
        
        Args:
            evaluation_result: The evaluation result dictionary
            persuasion_type: The persuasion type to generate suggestions for
            
        Returns:
            List of suggestion dictionaries appropriate for the persuasion type
        """
        try:
            print(f"🔍 Generating suggestions for persuasion type: {persuasion_type.upper()}")
            
            # Extract weaknesses and strengths for context
            weaknesses = evaluation_result.get("weaknesses", [])
            strengths = evaluation_result.get("strengths", []) if "strengths" in evaluation_result else []
            
            # For pathos, use the dedicated method
            if persuasion_type == "pathos":
                suggestions = [
                    {
                        "type": "pathos",
                        "description": "Use emotional storytelling to connect with the user's values",
                        "example": "Share a personal story about how this issue has emotionally affected you or someone close to you"
                    },
                    {
                        "type": "pathos",
                        "description": "Use vivid emotional language and imagery",
                        "example": "Describe emotional experiences with sensory details that create strong feelings"
                    },
                    {
                        "type": "pathos",
                        "description": "Appeal to shared values and emotional triggers",
                        "example": "Connect to core values like family, freedom, or security that resonate emotionally"
                    }
                ]
                
                print(f"✅ Generated pathos-specific suggestions:")
                for i, suggestion in enumerate(suggestions):
                    print(f"   {i+1}. {suggestion['description']}")
                
                return suggestions
            
            # For logos, generate logical reasoning suggestions
            elif persuasion_type == "logos":
                suggestions = [
                    {
                        "type": "logos",
                        "description": "Use data and statistics to support your argument",
                        "example": "Cite specific research findings with percentages or numbers"
                    },
                    {
                        "type": "logos",
                        "description": "Apply logical reasoning with clear cause-effect relationships",
                        "example": "Show how A leads to B through step-by-step logical analysis"
                    },
                    {
                        "type": "logos",
                        "description": "Address counterarguments with evidence",
                        "example": "Acknowledge opposing views and refute them with factual evidence"
                    }
                ]
                
                print(f"✅ Generated logos-specific suggestions:")
                for i, suggestion in enumerate(suggestions):
                    print(f"   {i+1}. {suggestion['description']}")
                
                return suggestions
            
            # For ethos, generate ethical/credibility suggestions
            elif persuasion_type == "ethos":
                suggestions = [
                    {
                        "type": "ethos",
                        "description": "Cite credible authorities and experts",
                        "example": "Reference respected institutions or experts in the field"
                    },
                    {
                        "type": "ethos",
                        "description": "Appeal to ethical principles and moral frameworks",
                        "example": "Connect your argument to widely accepted ethical standards"
                    },
                    {
                        "type": "ethos",
                        "description": "Demonstrate your credibility and trustworthiness",
                        "example": "Show understanding of multiple perspectives and fairness"
                    }
                ]
                
                print(f"✅ Generated ethos-specific suggestions:")
                for i, suggestion in enumerate(suggestions):
                    print(f"   {i+1}. {suggestion['description']}")
                
                return suggestions
            
            # Generic suggestions if type is unclear
            else:
                generic_suggestions = [
                    {
                        "type": "general",
                        "description": "Use a mix of logical, emotional, and ethical appeals",
                        "example": "Combine facts with stories and credibility to strengthen your argument"
                    },
                    {
                        "type": "general",
                        "description": "Address counterarguments directly",
                        "example": "Acknowledge opposing views and respond to them specifically"
                    },
                    {
                        "type": "general",
                        "description": "Provide specific examples",
                        "example": "Use concrete examples that illustrate your points clearly"
                    }
                ]
                
                print(f"✅ Generated generic suggestions:")
                for i, suggestion in enumerate(generic_suggestions):
                    print(f"   {i+1}. {suggestion['description']}")
                
                return generic_suggestions
                
        except Exception as e:
            print(f"⚠️ Error generating suggestions: {str(e)}")
            traceback.print_exc()
            
            # Return default suggestions in case of errors
            default_suggestions = [
                {
                    "type": "general",
                    "description": "Provide more specific examples to support your points",
                    "example": "Use concrete examples that clearly illustrate your position"
                },
                {
                    "type": "general",
                    "description": "Address potential counterarguments more directly",
                    "example": "Acknowledge and respond to opposing viewpoints"
                }
            ]
            
            return default_suggestions
    
    def _store_suggestions_in_state(self, state, suggestions):
        """Helper method to store suggestions in both state and global_state."""
        # Early return if state is None or not a dictionary
        if state is None or not isinstance(state, dict):
            print("⚠️ Cannot store suggestions - state is None or not a dictionary")
            return
            
        # Store in direct state
        state["suggestions"] = suggestions
        
        # Store in global state
        if "metadata" not in state:
            state["metadata"] = {}
        if "global_state" not in state["metadata"]:
            state["metadata"]["global_state"] = {}
        state["metadata"]["global_state"]["suggestions"] = suggestions
        
        print(f"✅ Stored suggestions in both state and global_state")

    def _evaluate_arguments(self, msg_state: MessageState) -> MessageState:
        """Evaluate the effectiveness of different persuasion types.
        This is the main evaluation function handling pathos, logos, and ethos evaluations.
        
        Args:
            msg_state: The current message state
            
        Returns:
            Updated state with effectiveness evaluations
        """
        print("\n================================================================================")
        print(f"📊 EVALUATING PERSUASION EFFECTIVENESS")
        print(f"================================================================================\n")
        
        try:
            # Extract the user's argument from the message state
            user_text = self._get_user_text(msg_state)
            if not user_text:
                print(f"⚠️ NO USER TEXT FOUND - Cannot evaluate effectiveness")
                return msg_state
            print(f"📄 Found user text for evaluation: {user_text[:100]}...")
            
            # Extract the counter-argument from the message state
            counter_arg_text = self._get_counter_argument_text(msg_state)
            if not counter_arg_text:
                print(f"⚠️ NO COUNTER-ARGUMENT FOUND - Cannot evaluate effectiveness")
                return msg_state
            print(f"📄 Found counter-argument for evaluation: {counter_arg_text[:100]}...")
            
            # Get the user's persuasion type (from ArgumentClassifier)
            user_persuasion_type = None
            
            # First check global state (higher priority)
            if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "user_persuasion_type" in msg_state["metadata"]["global_state"]:
                user_persuasion_type = msg_state["metadata"]["global_state"]["user_persuasion_type"]
                print(f"🌐 FOUND user_persuasion_type in global state: {user_persuasion_type}")
            # Then check direct state (lower priority)
            elif "user_persuasion_type" in msg_state:
                user_persuasion_type = msg_state["user_persuasion_type"]
                print(f"📄 FOUND user_persuasion_type in direct state: {user_persuasion_type}")
                
            # Get the bot's persuasion type that was actually used in the previous response
            bot_persuasion_type = None
            
            # First check global state (higher priority)
            if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "current_persuasion_type" in msg_state["metadata"]["global_state"]:
                bot_persuasion_type = msg_state["metadata"]["global_state"]["current_persuasion_type"]
                print(f"🌐 FOUND bot_persuasion_type (current_persuasion_type) in global state: {bot_persuasion_type}")
            # Then check direct state (lower priority)
            elif "current_persuasion_type" in msg_state:
                bot_persuasion_type = msg_state["current_persuasion_type"]
                print(f"📄 FOUND bot_persuasion_type (current_persuasion_type) in direct state: {bot_persuasion_type}")
            
            # If we can't determine the bot's persuasion type, check for counter_persuasion_type
            if not bot_persuasion_type:
                if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "counter_persuasion_type" in msg_state["metadata"]["global_state"]:
                    bot_persuasion_type = msg_state["metadata"]["global_state"]["counter_persuasion_type"]
                    print(f"🌐 FOUND bot_persuasion_type (counter_persuasion_type) in global state: {bot_persuasion_type}")
                elif "counter_persuasion_type" in msg_state:
                    bot_persuasion_type = msg_state["counter_persuasion_type"]
                    print(f"📄 FOUND bot_persuasion_type (counter_persuasion_type) in direct state: {bot_persuasion_type}")
            
            # Default to logos if we still don't know the bot's persuasion type
            if not bot_persuasion_type:
                bot_persuasion_type = "logos"
                print(f"⚠️ Could not determine bot's persuasion type, defaulting to: {bot_persuasion_type}")
            
            # Set counter_persuasion_type to match the user's persuasion type
            counter_persuasion_type = user_persuasion_type if user_persuasion_type else "logos"
            msg_state["counter_persuasion_type"] = counter_persuasion_type
            if "metadata" not in msg_state:
                msg_state["metadata"] = {}
            if "global_state" not in msg_state["metadata"]:
                msg_state["metadata"]["global_state"] = {}
            msg_state["metadata"]["global_state"]["counter_persuasion_type"] = counter_persuasion_type
            print(f"🌟 NEW STRATEGY: Setting counter_persuasion_type to {counter_persuasion_type.upper()} to match user's strategy")
            
            # Now set this as the recommended strategy for the next response
            msg_state["current_persuasion_type"] = counter_persuasion_type
            msg_state["metadata"]["global_state"]["current_persuasion_type"] = counter_persuasion_type
            
            # Generate evaluation prompt
            evaluation_prompt = self._generate_evaluation_prompt(user_text, counter_arg_text, bot_persuasion_type)
            
            # Evaluate effectiveness
            try:
                final_chain = self.llm | StrOutputParser()
                evaluation_result = final_chain.invoke(evaluation_prompt)
                
                try:
                    # Parse evaluation result
                    evaluation = json.loads(evaluation_result)
                    
                    # Create a default effectiveness score of 0.5
                    if "overall_score" not in evaluation:
                        evaluation["overall_score"] = 0.5
                    
                    # Store the evaluated effectiveness score
                    msg_state["evaluated_effectiveness_score"] = evaluation["overall_score"]
                    msg_state["metadata"]["global_state"]["evaluated_effectiveness_score"] = evaluation["overall_score"]
                    
                    # Store the full evaluation
                    msg_state["effectiveness_evaluation"] = evaluation
                    msg_state["metadata"]["global_state"]["effectiveness_evaluation"] = evaluation
                    
                    print(f"📊 EFFECTIVENESS SCORE: {evaluation['overall_score']:.2f}")
                    
                except json.JSONDecodeError:
                    print(f"⚠️ Failed to parse evaluation result as JSON")
                    evaluation = {"overall_score": 0.5}
                    msg_state["evaluated_effectiveness_score"] = evaluation["overall_score"]
                    msg_state["metadata"]["global_state"]["evaluated_effectiveness_score"] = evaluation["overall_score"]
            
            except Exception as e:
                print(f"⚠️ Error in LLM evaluation: {str(e)}")
                evaluation = {"overall_score": 0.5}
                msg_state["evaluated_effectiveness_score"] = evaluation["overall_score"]
                msg_state["metadata"]["global_state"]["evaluated_effectiveness_score"] = evaluation["overall_score"]
            
            return msg_state
                
        except Exception as e:
            print(f"⚠️ ERROR IN EFFECTIVENESS EVALUATION: {str(e)}")
            return msg_state

    def _create_default_evaluation(self):
        """Create a default evaluation when parsing fails."""
        print(f"Creating default evaluation due to parsing errors")
        criteria_scores = {}
        
        # Use the standardized criteria weights
        for criterion in self.STANDARD_CRITERIA:
            if criterion == "relevance":
                criteria_scores[criterion] = 0.7
            elif criterion == "persuasiveness":
                criteria_scores[criterion] = 0.6
            elif criterion == "credibility":
                criteria_scores[criterion] = 0.5
            elif criterion == "emotional_impact":
                criteria_scores[criterion] = 0.4
        
        return {
            "criteria_scores": criteria_scores,
            "overall_score": 0.55,
            "strengths": ["Generated basic counter-argument addressing the topic"],
            "weaknesses": ["Lacks specific evidence or citations to support claims"],
            "improvement_suggestions": ["Provide more concrete examples or data to strengthen argument"]
        }

    def _validate_evaluation(self, evaluation):
        """Validates the evaluation output format."""
        required_fields = [
            "criteria_scores",
            "strengths",
            "weaknesses",
            "improvement_suggestions",
            "reasoning"
        ]
        
        # Check required fields
        if not all(field in evaluation for field in required_fields):
            logger.error("Missing required fields in evaluation")
            return False
            
        # Validate criteria scores
        criteria_scores = evaluation["criteria_scores"]
        required_criteria = list(self.criteria.keys())
        if not all(criterion in criteria_scores for criterion in required_criteria):
            logger.error("Missing required criteria in evaluation")
            return False
            
        if not all(0 <= score <= 1 for score in criteria_scores.values()):
            logger.error("Invalid criteria scores in evaluation")
            return False
            
        return True

    def _create_action_graph(self):
        """
        Create the action graph for the worker.
        This method is required by BaseWorker.
        """
        # Since this worker uses a single action and doesn't need a complex action graph,
        # we return a simple dictionary with a single action
        return {
            "evaluate_effectiveness": {
                "description": "Evaluate the effectiveness of counter-arguments",
                "execution": self._evaluate_arguments
            }
        }

    def _get_counter_argument_text(self, msg_state):
        """Extract counter-argument text from the state.
        
        Args:
            msg_state: The current message state
            
        Returns:
            The counter-argument text or None if not found
        """
        counter_arg_text = None
        
        # Print detailed debug info for bot_message
        if "bot_message" in msg_state:
            bot_msg = msg_state["bot_message"]
            print(f"📄 DEBUG: bot_message type: {type(bot_msg).__name__}")
            if isinstance(bot_msg, dict):
                print(f"📄 DEBUG: bot_message keys: {list(bot_msg.keys())}")
            elif hasattr(bot_msg, '__dict__'):
                print(f"📄 DEBUG: bot_message attributes: {list(bot_msg.__dict__.keys())}")
        
        # Try to find the counter-argument in various places
        if "bot_message" in msg_state:
            bot_msg = msg_state["bot_message"]
            print(f"📄 DEBUG: Checking bot_message for counter-argument")
            
            # Method 1: If bot_message is an object with content attribute
            if hasattr(bot_msg, 'content'):
                counter_arg_text = bot_msg.content
                print(f"📄 DEBUG: Found counter-argument in bot_message.content (object)")
            
            # Method 2: If bot_message is a dict with content key
            elif isinstance(bot_msg, dict) and "content" in bot_msg:
                counter_arg_text = bot_msg["content"]
                print(f"📄 DEBUG: Found counter-argument in bot_message['content'] (dict)")
            
            # Method 3: If bot_message has a message attribute
            elif hasattr(bot_msg, 'message'):
                message = bot_msg.message
                if hasattr(message, 'content'):
                    counter_arg_text = message.content
                    print(f"📄 DEBUG: Found counter-argument in bot_message.message.content")
                elif isinstance(message, dict) and "content" in message:
                    counter_arg_text = message["content"]
                    print(f"📄 DEBUG: Found counter-argument in bot_message.message['content']")
            
            # Method 4: Try to convert to string as fallback
            else:
                try:
                    counter_arg_text = str(bot_msg)
                    print(f"📄 DEBUG: Converted bot_message to string for counter-argument")
                except:
                    pass
        
        # Also check assistant_message and response fields
        if not counter_arg_text and "assistant_message" in msg_state:
            assistant_msg = msg_state["assistant_message"]
            if hasattr(assistant_msg, 'content'):
                counter_arg_text = assistant_msg.content
                print(f"📄 DEBUG: Found counter-argument in assistant_message.content (object)")
            elif isinstance(assistant_msg, dict) and "content" in assistant_msg:
                counter_arg_text = assistant_msg["content"]
                print(f"📄 DEBUG: Found counter-argument in assistant_message['content'] (dict)")
        
        if not counter_arg_text and "response" in msg_state:
            response = msg_state["response"]
            if isinstance(response, str):
                counter_arg_text = response
                print(f"📄 DEBUG: Found counter-argument in response (string)")
            elif isinstance(response, dict) and "content" in response:
                counter_arg_text = response["content"]
                print(f"📄 DEBUG: Found counter-argument in response['content'] (dict)")
            elif hasattr(response, 'content'):
                counter_arg_text = response.content
                print(f"📄 DEBUG: Found counter-argument in response.content (object)")
        
        if not counter_arg_text and "messages" in msg_state:
            print(f"📄 DEBUG: Searching for counter-argument in messages array")
            for i, msg in enumerate(reversed(msg_state["messages"])):
                role = None
                if isinstance(msg, dict):
                    role = msg.get("role")
                elif hasattr(msg, 'role'):
                    role = msg.role
                
                print(f"📄 DEBUG: Message {i} role: {role}")
                
                if (isinstance(msg, dict) and msg.get("role") == "assistant") or \
                   (hasattr(msg, 'role') and msg.role == "assistant" and hasattr(msg, 'content')):
                    if isinstance(msg, dict):
                        counter_arg_text = msg.get("content")
                        print(f"📄 DEBUG: Found counter-argument in messages[{i}]['content'] (dict)")
                    else:
                        counter_arg_text = msg.content
                        print(f"📄 DEBUG: Found counter-argument in messages[{i}].content (object)")
                    break
        
        # Also check 'orchestrator_message' field which may contain bot responses
        if not counter_arg_text and "orchestrator_message" in msg_state:
            orch_msg = msg_state["orchestrator_message"]
            print(f"📄 DEBUG: Checking orchestrator_message for counter-argument")
            
            if hasattr(orch_msg, 'content'):
                counter_arg_text = orch_msg.content
                print(f"📄 DEBUG: Found counter-argument in orchestrator_message.content")
            elif isinstance(orch_msg, dict) and "content" in orch_msg:
                counter_arg_text = orch_msg["content"]
                print(f"📄 DEBUG: Found counter-argument in orchestrator_message['content']")
            elif hasattr(orch_msg, 'message') and hasattr(orch_msg.message, 'content'):
                counter_arg_text = orch_msg.message.content
                print(f"📄 DEBUG: Found counter-argument in orchestrator_message.message.content")
        
        # Check 'trajectory' for bot responses
        if not counter_arg_text and "trajectory" in msg_state:
            trajectory = msg_state["trajectory"]
            print(f"📄 DEBUG: Checking trajectory for counter-argument")
            
            if isinstance(trajectory, list) and len(trajectory) > 0:
                # Try to find the last bot message in the trajectory
                for i, step in enumerate(reversed(trajectory)):
                    if isinstance(step, dict) and "role" in step and step["role"] == "assistant":
                        if "content" in step:
                            counter_arg_text = step["content"]
                            print(f"📄 DEBUG: Found counter-argument in trajectory[{len(trajectory)-i-1}]['content']")
                            break
        
        # If we still don't have a counter-argument, create a default one
        if not counter_arg_text:
            print(f"⚠️ COULD NOT FIND COUNTER-ARGUMENT - USING PLACEHOLDER FOR EVALUATION")
            counter_arg_text = "This is a placeholder counter-argument for evaluation purposes."
            
        return counter_arg_text

    def _get_user_text(self, msg_state):
        """Extract user response text from the state.
        
        Args:
            msg_state: The current message state
            
        Returns:
            The user response text or None if not found
        """
        user_text = None
        
        # Get user content from various possible locations in state
        if "user_message" in msg_state:
            user_message_obj = msg_state["user_message"]
            # Handle different types of user_message
            if hasattr(user_message_obj, 'content'):
                user_text = user_message_obj.content
                print(f"📄 DEBUG: Found user text in user_message.content (object)")
            elif isinstance(user_message_obj, dict) and "content" in user_message_obj:
                user_text = user_message_obj["content"] 
                print(f"📄 DEBUG: Found user text in user_message['content'] (dict)")
            else:
                # Try to convert to string
                try:
                    user_text = str(user_message_obj)
                    print(f"📄 DEBUG: Converted user_message to string for user text")
                except:
                    pass
        
        if not user_text and "messages" in msg_state:
            print(f"📄 DEBUG: Searching in messages array for user text")
            for i, msg in enumerate(reversed(msg_state["messages"])):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_text = msg.get("content")
                    print(f"📄 DEBUG: Found user text in messages[{i}]['content'] (dict)")
                    break
                elif hasattr(msg, 'role') and msg.role == "user" and hasattr(msg, 'content'):
                    user_text = msg.content
                    print(f"📄 DEBUG: Found user text in messages[{i}].content (object)")
                    break
        
        if not user_text and "content" in msg_state:
            user_text = msg_state["content"]
            print(f"📄 DEBUG: Found user text in content field")
        
        return user_text

    def _get_suggestion_for_type(self, persuasion_type, evaluation_result):
        """Generate a specific suggestion for improving a particular persuasion type.
        
        Args:
            persuasion_type: The type of persuasion (pathos, logos, ethos)
            evaluation_result: The evaluation details
            
        Returns:
            A specific suggestion to improve that persuasion type
        """
        strengths = evaluation_result.get("strengths", [])
        weaknesses = evaluation_result.get("weaknesses", [])
        
        if persuasion_type == "pathos":
            # For pathos, suggest emotional storytelling and personal experiences
            pathos_suggestions = [
                "Share a deeply personal or emotional experience that illustrates your point",
                "Use vivid emotional imagery and sensory details to create an emotional impact",
                "Appeal to shared human experiences and feelings that resonate with the audience",
                "Tell a compelling story with emotional weight that makes the audience feel connected",
                "Focus on how people feel about the issue rather than just factual information"
            ]
            # Select a suggestion that doesn't contradict the existing strengths
            for suggestion in pathos_suggestions:
                if not any(s.lower() in suggestion.lower() for s in strengths):
                    return suggestion
            return pathos_suggestions[0]  # Default to first suggestion if all overlap with strengths
                
        elif persuasion_type == "logos":
            # For logos, suggest more evidence and logical structure
            logos_suggestions = [
                "Incorporate statistics or studies to support claims with concrete evidence",
                "Improve logical structure with clearer cause-effect relationships",
                "Strengthen analytical reasoning by addressing counterarguments directly",
                "Add more specific facts, data, or expert opinions to reinforce logical arguments",
                "Present a step-by-step logical analysis that leads to your conclusion"
            ]
            # Select a suggestion that doesn't contradict the existing strengths
            for suggestion in logos_suggestions:
                if not any(s.lower() in suggestion.lower() for s in strengths):
                    return suggestion
            return logos_suggestions[0]  # Default to first suggestion if all overlap with strengths
                
        elif persuasion_type == "ethos":
            # For ethos, suggest more credibility and ethical appeals
            ethos_suggestions = [
                "Cite recognized authorities or ethical frameworks to strengthen credibility",
                "Appeal more directly to fundamental ethical principles and shared moral values",
                "Demonstrate greater ethical character and trustworthiness in your position",
                "Reference consensus views from respected experts or institutions",
                "Connect your argument to established ethical standards and traditions"
            ]
            # Select a suggestion that doesn't contradict the existing strengths
            for suggestion in ethos_suggestions:
                if not any(s.lower() in suggestion.lower() for s in strengths):
                    return suggestion
            return ethos_suggestions[0]  # Default to first suggestion if all overlap with strengths
                
        else:
            return "Consider using a more diverse range of persuasive techniques to strengthen your argument"

    def _ensure_evaluation_matches_persuasion_type(self, evaluation, persuasion_type):
        """Post-process the evaluation to ensure it matches the persuasion type."""
        if persuasion_type == "pathos":
            # For pathos, ensure emotional_impact is at least 0.5
            if "criteria_scores" in evaluation and "emotional_impact" in evaluation["criteria_scores"]:
                if evaluation["criteria_scores"]["emotional_impact"] < 0.5:
                    evaluation["criteria_scores"]["emotional_impact"] = 0.5
                    
            # Ensure weaknesses relate to emotional appeal
            pathos_weaknesses = [
                "Lacks emotional depth in storytelling",
                "Missing personal connection with the audience",
                "Could use more vivid emotional imagery",
                "Fails to evoke strong emotional response",
                "Doesn't create sufficient emotional resonance"
            ]
            
            # Ensure improvement suggestions are pathos-oriented
            pathos_suggestions = [
                "Share more emotionally powerful personal stories",
                "Use more vivid emotional language and first-person perspective",
                "Appeal directly to the heart with personal experiences that create emotional connection",
                "Paint a more vivid emotional picture that resonates with the audience's feelings",
                "Include more emotionally impactful examples that create empathy"
            ]
            
            # Replace any non-pathos weaknesses
            if "weaknesses" in evaluation:
                for i, weakness in enumerate(evaluation["weaknesses"]):
                    # Check if the weakness sounds like a logos or ethos-based criticism
                    if any(word in weakness.lower() for word in ["evidence", "data", "fact", "logic", "research", "statistic", "study", "authority", "expert", "credential"]):
                        # Replace with pathos-oriented weakness
                        evaluation["weaknesses"][i] = pathos_weaknesses[i % len(pathos_weaknesses)]
            
            # Replace any non-pathos suggestions
            if "improvement_suggestions" in evaluation:
                for i, suggestion in enumerate(evaluation["improvement_suggestions"]):
                    # Check if the suggestion sounds like a logos or ethos-based recommendation
                    if any(word in suggestion.lower() for word in ["evidence", "data", "fact", "logic", "research", "statistic", "study", "authority", "expert", "credential", "case studies", "systemic"]):
                        # Replace with pathos-oriented suggestion
                        evaluation["improvement_suggestions"][i] = pathos_suggestions[i % len(pathos_suggestions)]
                
        elif persuasion_type == "logos":
            # For logos, ensure credibility and persuasiveness are high
            if "criteria_scores" in evaluation and "credibility" in evaluation["criteria_scores"]:
                if evaluation["criteria_scores"]["credibility"] < 0.5:
                    evaluation["criteria_scores"]["credibility"] = 0.5
            
            # Ensure weaknesses relate to logical reasoning
            logos_weaknesses = [
                "Lacks sufficient evidence or data to support claims",
                "Reasoning contains logical fallacies or gaps",
                "Missing important statistical support",
                "Could use more expert citations",
                "Analysis lacks sufficient depth"
            ]
            
            # Ensure improvement suggestions are logos-oriented
            logos_suggestions = [
                "Include more specific statistics and data points",
                "Strengthen logical reasoning with clearer cause-effect relationships",
                "Support claims with verified research findings",
                "Address logical counterarguments more directly",
                "Present more factual evidence from reliable sources"
            ]
            
            # Replace any non-logos weaknesses/suggestions as needed
            if "weaknesses" in evaluation:
                for i, weakness in enumerate(evaluation["weaknesses"]):
                    if any(word in weakness.lower() for word in ["emotion", "feeling", "story", "personal", "heart", "authority", "moral", "ethical"]):
                        evaluation["weaknesses"][i] = logos_weaknesses[i % len(logos_weaknesses)]
            
            if "improvement_suggestions" in evaluation:
                for i, suggestion in enumerate(evaluation["improvement_suggestions"]):
                    if any(word in suggestion.lower() for word in ["emotion", "feeling", "story", "personal", "heart", "authority", "moral", "ethical"]):
                        evaluation["improvement_suggestions"][i] = logos_suggestions[i % len(logos_suggestions)]
                        
        elif persuasion_type == "ethos":
            # For ethos, ensure credibility is high
            if "criteria_scores" in evaluation and "credibility" in evaluation["criteria_scores"]:
                if evaluation["criteria_scores"]["credibility"] < 0.5:
                    evaluation["criteria_scores"]["credibility"] = 0.5
            
            # Ensure weaknesses relate to credibility and ethics
            ethos_weaknesses = [
                "Lacks sufficient credibility or authority",
                "Missing appeals to ethical principles or moral values",
                "Could establish more trustworthiness",
                "Doesn't sufficiently connect to established ethical frameworks",
                "Needs stronger appeals to shared moral principles"
            ]
            
            # Ensure improvement suggestions are ethos-oriented
            ethos_suggestions = [
                "Cite more recognized authorities or ethical frameworks",
                "Establish stronger ethical credibility with the audience",
                "Appeal more directly to shared moral values",
                "Reference more respected institutions or experts",
                "Connect arguments more explicitly to ethical principles"
            ]
            
            # Replace any non-ethos weaknesses/suggestions as needed
            if "weaknesses" in evaluation:
                for i, weakness in enumerate(evaluation["weaknesses"]):
                    if any(word in weakness.lower() for word in ["emotion", "feeling", "story", "personal", "heart", "evidence", "data", "fact", "logic", "research", "statistic", "study"]):
                        evaluation["weaknesses"][i] = ethos_weaknesses[i % len(ethos_weaknesses)]
            
            if "improvement_suggestions" in evaluation:
                for i, suggestion in enumerate(evaluation["improvement_suggestions"]):
                    if any(word in suggestion.lower() for word in ["emotion", "feeling", "story", "personal", "heart", "evidence", "data", "fact", "logic", "research", "statistic", "study"]):
                        evaluation["improvement_suggestions"][i] = ethos_suggestions[i % len(ethos_suggestions)]
                        
        return evaluation

    def _generate_evaluation_prompt(self, original_argument: str, counter_argument: str, persuasion_type: str) -> str:
        """Generate the evaluation prompt based on the arguments and persuasion type."""
        # Simplified evaluation prompt
        return f"""
        Please evaluate the effectiveness of a counter-argument in response to the original argument.
        
        Original Argument: {original_argument}
        
        Counter Argument: {counter_argument}
        
        Persuasion Type: {persuasion_type}
        
        Evaluate the counter-argument's effectiveness on a scale from 0 to 1, where 0 is completely ineffective 
        and 1 is highly effective. Consider how well the counter-argument addresses the points in the original argument.
        
        Provide your evaluation in JSON format:
        {{
            "overall_score": 0.7,  // A decimal between 0 and 1
            "analysis": "Brief analysis of the effectiveness"
        }}
        """

# Register the effectiveness evaluator worker
register_worker(EffectivenessEvaluator) 
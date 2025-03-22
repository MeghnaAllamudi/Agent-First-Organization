import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP


logger = logging.getLogger(__name__)


@register_worker
class EffectivenessEvaluator(BaseWorker):
    """A worker that evaluates the effectiveness of counter-arguments."""
    
    description = "Evaluates the effectiveness of counter-arguments based on multiple criteria."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.stream_response = MODEL.get("stream_response", True)
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize the action graph
        self.action_graph = self._create_action_graph()
        
        # Extract criteria from config if provided
        if config and "criteria" in config:
            self.criteria = config["criteria"]
        else:
            # Set default criteria if none provided
            self.criteria = {
                "relevance": {
                    "weight": 0.3,
                    "description": "How well does it address the original argument?",
                    "examples": [
                        "Directly responds to the main point",
                        "Addresses key concerns",
                        "Stays focused on the topic"
                    ]
                },
                "persuasiveness": {
                    "weight": 0.25,
                    "description": "How convincing is the counter-argument?",
                    "examples": [
                        "Uses compelling evidence",
                        "Presents clear reasoning",
                        "Builds a strong case"
                    ]
                },
                "credibility": {
                    "weight": 0.25,
                    "description": "How credible and well-supported is the argument?",
                    "examples": [
                        "Cites reliable sources",
                        "Uses expert opinions",
                        "Presents verifiable facts"
                    ]
                },
                "emotional_impact": {
                    "weight": 0.2,
                    "description": "How effectively does it engage emotions or connect with values?",
                    "examples": [
                        "Appeals to shared values",
                        "Evokes appropriate emotional response",
                        "Creates connection with audience"
                    ]
                }
            }
        
        # Validate weights sum to 1
        total_weight = sum(criterion["weight"] for criterion in self.criteria.values())
        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError("Criteria weights must sum to 1.0")
        
        # Define the evaluation prompt template
        self.evaluation_prompt = PromptTemplate.from_template(
            """You are an expert at evaluating the effectiveness of debate counter-arguments. Your task is to evaluate a counter-argument's effectiveness based on specific criteria.

Original Argument: {original_argument}

Counter-Argument: {counter_argument}

Persuasion Type Used: {persuasion_type}

Evaluation Criteria: {criteria}

IMPORTANT: You MUST respond with a valid JSON object containing the following fields:
- criteria_scores: An object with scores (0.0 to 1.0) for each criterion
- strengths: Array of key strengths
- weaknesses: Array of key weaknesses
- improvement_suggestions: Array of suggestions for improvement
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

    def _calculate_weighted_score(self, criteria_scores: Dict[str, float]) -> float:
        """Calculates the weighted overall score from criteria scores."""
        weighted_sum = sum(
            score * self.criteria[criterion]["weight"]
            for criterion, score in criteria_scores.items()
        )
        return round(weighted_sum, 3)  # Round to 3 decimal places

    def _validate_evaluation(self, evaluation: Dict[str, Any]) -> bool:
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

    def _evaluate_argument(self, original_argument, counter_argument, persuasion_type=None):
        """
        Evaluate the effectiveness of a counter-argument.
        
        Args:
            original_argument: The original argument to counter
            counter_argument: The counter-argument to evaluate
            persuasion_type: The type of persuasion used (logos, pathos, ethos)
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Use default persuasion type if none provided
            if not persuasion_type:
                persuasion_type = "logos"  # Default to logos as safest option
                
            # Format the criteria as a JSON string for the prompt
            criteria_str = json.dumps(self.criteria, indent=2)
            
            # Format the prompt with the arguments and criteria
            formatted_prompt = self.evaluation_prompt.format(
                original_argument=original_argument,
                counter_argument=counter_argument["counter_argument"],
                persuasion_type=persuasion_type,
                criteria=criteria_str
            )
            
            # Generate evaluation
            chain = self.llm | StrOutputParser()
            evaluation_str = chain.invoke(formatted_prompt)
            
            # Try to parse JSON with better error handling
            try:
                # First, try to parse the evaluation as is
                evaluation = json.loads(evaluation_str)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the response
                print(f"\n================================================================================")
                print(f"⚠️ JSON PARSING ERROR IN EFFECTIVENESS EVALUATOR")
                print(f"Attempting to fix invalid JSON response...")
                print(f"================================================================================")
                
                import re
                # Try to extract JSON content between curly braces
                json_match = re.search(r'({.*})', evaluation_str, re.DOTALL)
                if json_match:
                    try:
                        evaluation = json.loads(json_match.group(1))
                        print(f"Successfully extracted valid JSON from response")
                    except json.JSONDecodeError:
                        # If that still fails, create a default evaluation
                        print(f"Could not extract valid JSON. Creating default evaluation.")
                        evaluation = {
                            "criteria_scores": {
                                "relevance": 0.7,
                                "persuasiveness": 0.7,
                                "credibility": 0.7,
                                "emotional_impact": 0.7
                            },
                            "overall_score": 0.7,
                            "strengths": ["Default evaluation"],
                            "weaknesses": ["Could not parse LLM response"],
                            "improvement_suggestions": ["Improve JSON formatting in response"],
                            "reasoning": "Default evaluation due to JSON parsing error"
                        }
                else:
                    # No JSON-like content found, create default evaluation
                    print(f"No JSON content found in response. Creating default evaluation.")
                    evaluation = {
                        "criteria_scores": {
                            "relevance": 0.7,
                            "persuasiveness": 0.7,
                            "credibility": 0.7,
                            "emotional_impact": 0.7
                        },
                        "overall_score": 0.7,
                        "strengths": ["Default evaluation"],
                        "weaknesses": ["Could not parse LLM response"],
                        "improvement_suggestions": ["Improve JSON formatting in response"],
                        "reasoning": "Default evaluation due to JSON parsing error"
                    }
            
            # Validate the evaluation
            if not self._validate_evaluation(evaluation):
                # Return a default evaluation if validation fails
                print(f"Evaluation validation failed. Creating default evaluation.")
                evaluation = {
                    "criteria_scores": {
                        "relevance": 0.7,
                        "persuasiveness": 0.7,
                        "credibility": 0.7,
                        "emotional_impact": 0.7
                    },
                    "overall_score": 0.7,
                    "strengths": ["Default evaluation"],
                    "weaknesses": ["Invalid evaluation format"],
                    "improvement_suggestions": ["Improve response structure"],
                    "reasoning": "Default evaluation due to validation failure"
                }
            
            # Calculate weighted overall score
            evaluation["overall_score"] = self._calculate_weighted_score(evaluation["criteria_scores"])
                
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating argument: {str(e)}")
            # Return a default evaluation in case of any error
            return {
                "criteria_scores": {
                    "relevance": 0.7,
                    "persuasiveness": 0.7,
                    "credibility": 0.7,
                    "emotional_impact": 0.7
                },
                "overall_score": 0.7,
                "strengths": ["Default evaluation due to error"],
                "weaknesses": ["Error during evaluation process"],
                "improvement_suggestions": ["Check system logs for details"],
                "reasoning": f"Error during evaluation: {str(e)}"
            }

    def _create_action_graph(self):
        """Creates the action graph for effectiveness evaluation."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add evaluation node
        workflow.add_node("evaluator", self._evaluate_arguments)
        
        # Add edges
        workflow.add_edge(START, "evaluator")
        
        return workflow

    def _evaluate_arguments(self, msg_state: MessageState) -> MessageState:
        """
        Evaluates the arguments and user response to determine effectiveness.
        """
        logger.info("Starting argument effectiveness evaluation")
        try:
            # Get user message
            user_message = msg_state.get("user_message", {})
            user_content = user_message.get("content", "") if isinstance(user_message, dict) else str(user_message)
            
            # Check for persuasion type in multiple places with enhanced logging
            persuasion_type = None
            
            # Method 0: Check for persuasion_type in orchestrator_message attribute (from taskgraph)
            orchestrator_message = msg_state.get("orchestrator_message", {})
            if orchestrator_message and hasattr(orchestrator_message, "attribute"):
                if orchestrator_message.attribute and "persuasion_type" in orchestrator_message.attribute:
                    persuasion_type = orchestrator_message.attribute.get("persuasion_type")
                    logger.info(f"Found persuasion_type in orchestrator_message: {persuasion_type}")
                elif orchestrator_message.attribute:
                    # Infer from task field if available
                    task = orchestrator_message.attribute.get("task", "")
                    if isinstance(task, str):
                        task_lower = task.lower()
                        if "pathos" in task_lower or "emotion" in task_lower:
                            persuasion_type = "pathos"
                            logger.info(f"Inferred persuasion_type from task description: {persuasion_type}")
                        elif "logos" in task_lower or "logic" in task_lower or "evidence" in task_lower:
                            persuasion_type = "logos"
                            logger.info(f"Inferred persuasion_type from task description: {persuasion_type}")
                        elif "ethos" in task_lower or "credibility" in task_lower or "authority" in task_lower:
                            persuasion_type = "ethos"
                            logger.info(f"Inferred persuasion_type from task description: {persuasion_type}")

                    # Also check if we can determine effectiveness score from the orchestrator message
                    message_value = orchestrator_message.attribute.get("value", "")
                    if isinstance(message_value, str) and "%" in message_value:
                        import re
                        percent_match = re.search(r'(\d+)%', message_value)
                        if percent_match:
                            percent_value = int(percent_match.group(1))
                            msg_state["extracted_effectiveness_score"] = round(percent_value / 100.0, 2)
                            logger.info(f"Extracted effectiveness score from message value: {msg_state['extracted_effectiveness_score']} ({percent_value}%)")
            
            # Method 1: Direct state variables (most reliable)
            if not persuasion_type and msg_state.get("current_persuasion_type"):
                persuasion_type = msg_state.get("current_persuasion_type")
                logger.info(f"Found persuasion_type in current_persuasion_type: {persuasion_type}")
            elif not persuasion_type and msg_state.get("just_used_persuasion_type"):
                persuasion_type = msg_state.get("just_used_persuasion_type")
                logger.info(f"Found persuasion_type in just_used_persuasion_type: {persuasion_type}")
            
            # Method 2: Check in slots
            elif not persuasion_type and msg_state.get("slots", {}).get("current_technique"):
                persuasion_type = msg_state["slots"]["current_technique"]
                logger.info(f"Found persuasion_type in slots.current_technique: {persuasion_type}")
            elif not persuasion_type and msg_state.get("slots", {}).get("best_technique"):
                persuasion_type = msg_state["slots"]["best_technique"]
                logger.info(f"Found persuasion_type in slots.best_technique: {persuasion_type}")
            
            # Method 3: Look for response objects
            for p_type in ["pathos", "logos", "ethos"]:
                response_key = f"{p_type}_persuasion_response"
                if not persuasion_type and response_key in msg_state:
                    persuasion_type = p_type
                    logger.info(f"Found persuasion_type from response key: {persuasion_type}")
                    break
            
            # Check for effectiveness score in orchestrator message
            has_effectiveness_score = False
            effectiveness_score = None
            
            # First check if we already extracted a score
            if "extracted_effectiveness_score" in msg_state:
                effectiveness_score = msg_state["extracted_effectiveness_score"]
                has_effectiveness_score = True
                logger.info(f"Using previously extracted effectiveness score: {effectiveness_score}")
            
            # Otherwise try to extract from the message content
            elif orchestrator_message and hasattr(orchestrator_message, "message"):
                message_content = orchestrator_message.message
                if isinstance(message_content, str) and "%" in message_content:
                    try:
                        # Extract percentage from message
                        import re
                        percent_match = re.search(r'(\d+)%', message_content)
                        if percent_match:
                            percent_value = int(percent_match.group(1))
                            effectiveness_score = round(percent_value / 100.0, 2)
                            has_effectiveness_score = True
                            logger.info(f"Extracted effectiveness score from message: {effectiveness_score} ({percent_value}%)")
                    except Exception as e:
                        logger.error(f"Error extracting effectiveness score: {str(e)}")
            
            # If we can't determine persuasion type but have a score, this is ok - log as INFO
            if not persuasion_type and has_effectiveness_score:
                # Set a default type but log as info since we have a score
                persuasion_type = "logos"  # Default
                logger.info(f"Using default persuasion_type '{persuasion_type}' with effectiveness score {effectiveness_score}")
                logger.info(f"Available state keys: {list(msg_state.keys())}")
                if orchestrator_message:
                    logger.info(f"Orchestrator message: {orchestrator_message}")
                    if hasattr(orchestrator_message, "attribute"):
                        logger.info(f"Attribute: {orchestrator_message.attribute}")
            # If we can't determine persuasion type and have no score, this is a warning
            elif not persuasion_type:
                # Set a default type
                persuasion_type = "logos"  # Default
                logger.warning(f"Could not determine persuasion_type from any source, using default '{persuasion_type}'")
                logger.info(f"Available state keys: {list(msg_state.keys())}")
                if orchestrator_message:
                    logger.info(f"Orchestrator message: {orchestrator_message}")
                    if hasattr(orchestrator_message, "attribute"):
                        logger.info(f"Attribute: {orchestrator_message.attribute}")
            
            # Get the counter-argument - try multiple sources
            counter_argument = {}
            
            # Method 1: Check in response objects
            response_key = f"{persuasion_type}_persuasion_response"
            if response_key in msg_state:
                counter_argument = msg_state[response_key]
                logger.info(f"Found counter-argument in {response_key}")
            
            # Method 2: Check in orchestrator message
            if not counter_argument and orchestrator_message:
                if hasattr(orchestrator_message, "attribute") and orchestrator_message.attribute:
                    counter_text = orchestrator_message.attribute.get("value", "")
                    counter_argument = {"counter_argument": counter_text}
                    logger.info("Found counter-argument in orchestrator_message.attribute.value")
                elif hasattr(orchestrator_message, "message"):
                    counter_text = orchestrator_message.message
                    counter_argument = {"counter_argument": counter_text}
                    logger.info("Found counter-argument in orchestrator_message.message")
            
            # Method 3: Try to find in message_flow
            if not counter_argument and msg_state.get("message_flow"):
                message_flow = msg_state["message_flow"]
                if isinstance(message_flow, list) and len(message_flow) > 0:
                    latest_message = message_flow[-1]
                    if isinstance(latest_message, dict) and "content" in latest_message:
                        counter_text = latest_message["content"]
                        counter_argument = {"counter_argument": counter_text}
                        logger.info("Found counter-argument in message_flow")
            
            # If we couldn't find the counter-argument, try to reconstruct it
            if not counter_argument:
                # Use a placeholder for the counter-argument
                counter_argument = {
                    "counter_argument": "Counter-argument not found in state. Please evaluate based on user's response."
                }
                logger.info("Could not find counter-argument, using placeholder")
            
            # Get the original argument
            original_argument = "Original argument not available"
            
            # Method 1: Check in input fields
            if "argument" in msg_state:
                original_argument = msg_state["argument"]
                logger.info("Found original argument in state.argument")
            
            # Method 2: Check in user message history
            elif msg_state.get("message_flow") and isinstance(msg_state["message_flow"], list):
                message_flow = msg_state["message_flow"]
                for message in reversed(message_flow):
                    if isinstance(message, dict) and message.get("role") == "user":
                        original_argument = message.get("content", "")
                        logger.info("Found original argument in message_flow")
                        break
            
            # Evaluate the counter-argument
            evaluation = None
            
            # If we have an effectiveness score already, use it
            if has_effectiveness_score and effectiveness_score is not None:
                evaluation = {
                    "criteria_scores": {
                        "relevance": effectiveness_score,
                        "persuasiveness": effectiveness_score,
                        "credibility": effectiveness_score,
                        "emotional_impact": effectiveness_score
                    },
                    "overall_score": effectiveness_score,
                    "strengths": ["Determined from user response"],
                    "weaknesses": ["Detailed analysis not available"],
                    "improvement_suggestions": ["Continue monitoring user engagement"],
                    "reasoning": f"Effectiveness score of {effectiveness_score} extracted from orchestrator message."
                }
                logger.info(f"Used extracted effectiveness score: {effectiveness_score}")
            else:
                # Otherwise, perform a full evaluation
                try:
                    evaluation = self._evaluate_argument(original_argument, counter_argument, persuasion_type)
                    if evaluation:
                        logger.info(f"Completed evaluation with overall score: {evaluation.get('overall_score', 'N/A')}")
                    else:
                        # Create a fallback evaluation when the actual evaluation fails
                        logger.warning("Evaluation function returned None, using fallback evaluation")
                        evaluation = {
                            "criteria_scores": {
                                "relevance": 0.7,
                                "persuasiveness": 0.7, 
                                "credibility": 0.7,
                                "emotional_impact": 0.7
                            },
                            "overall_score": 0.7,
                            "strengths": ["Default evaluation"],
                            "weaknesses": ["Detailed analysis not available"],
                            "improvement_suggestions": ["Improve state passing for better evaluations"],
                            "reasoning": "Fallback evaluation due to evaluation function failure."
                        }
                except Exception as e:
                    logger.warning(f"Error in evaluation function: {str(e)}, using fallback evaluation")
                    # Create a fallback evaluation when JSON parsing fails
                    evaluation = {
                        "criteria_scores": {
                            "relevance": 0.7,
                            "persuasiveness": 0.7,
                            "credibility": 0.7,
                            "emotional_impact": 0.7
                        },
                        "overall_score": 0.7,
                        "strengths": ["Default evaluation after error"],
                        "weaknesses": ["Error occurred during evaluation"],
                        "improvement_suggestions": ["Check logs for evaluation errors"],
                        "reasoning": f"Fallback evaluation due to error: {str(e)}"
                    }
                    
            # Now, update the state with evaluation results
            msg_state["evaluated_effectiveness_score"] = evaluation["overall_score"]
            msg_state["effectiveness_evaluation"] = evaluation
            
            # Also store it in the persuasion-specific field
            response_key = f"{persuasion_type}_persuasion_response"
            if response_key in msg_state and isinstance(msg_state[response_key], dict):
                msg_state[response_key]["effectiveness_score"] = evaluation["overall_score"]
                msg_state[response_key]["effectiveness_evaluation"] = evaluation
                logger.info(f"Updated {response_key} with effectiveness score")
            else:
                # Create the response object if it doesn't exist
                msg_state[response_key] = {
                    "counter_argument": counter_argument.get("counter_argument", ""),
                    "effectiveness_score": evaluation["overall_score"],
                    "effectiveness_evaluation": evaluation
                }
                logger.info(f"Created {response_key} with effectiveness score")
            
            # Update slots with the most recent effectiveness information
            slots = msg_state.get("slots", {})
            if not slots:
                slots = {"persuasion_techniques": {}}
                
            if "persuasion_techniques" not in slots:
                slots["persuasion_techniques"] = {}
                
            if persuasion_type not in slots["persuasion_techniques"]:
                slots["persuasion_techniques"][persuasion_type] = {"scores": []}
                
            # Add the new score
            slots["persuasion_techniques"][persuasion_type]["scores"] = \
                slots["persuasion_techniques"][persuasion_type].get("scores", []) + [evaluation["overall_score"]]
                
            # Update slots
            msg_state["slots"] = slots
            
            # Set success status
            msg_state["status"] = "success"
            
            # Print very visible output about the evaluation
            effectiveness_percent = int(evaluation["overall_score"] * 100)
            print(f"\n{'='*80}")
            print(f"🔍 EFFECTIVENESS EVALUATION: {persuasion_type.upper()}")
            print(f"   OVERALL SCORE: {effectiveness_percent}%")
            print(f"   CRITERIA SCORES:")
            for criterion, score in evaluation["criteria_scores"].items():
                print(f"   - {criterion.upper()}: {int(score * 100)}%")
            print(f"   STRENGTHS: {', '.join(evaluation['strengths'])}")
            print(f"{'='*80}\n")
            
            logger.info(f"✅ PERSUASION EFFECTIVENESS - {persuasion_type.upper()}: {evaluation['overall_score']:.2f}")
            
            return msg_state
                
        except Exception as e:
            logger.error(f"Error in effectiveness evaluation: {str(e)}")
            msg_state["status"] = "error"
            msg_state["error"] = str(e)
            return msg_state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the effectiveness evaluation workflow."""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result

# Register the effectiveness evaluator worker
register_worker(EffectivenessEvaluator) 
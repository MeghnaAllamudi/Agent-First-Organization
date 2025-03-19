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
                        "Uses accurate data",
                        "Appeals to authority"
                    ]
                },
                "emotional_impact": {
                    "weight": 0.2,
                    "description": "How effectively does it connect with the audience?",
                    "examples": [
                        "Resonates emotionally",
                        "Appeals to values",
                        "Creates engagement"
                    ]
                }
            }
        
        # Validate weights sum to 1
        total_weight = sum(criterion["weight"] for criterion in self.criteria.values())
        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError("Criteria weights must sum to 1.0")
        
        self.evaluation_prompt = PromptTemplate.from_template(
            """Evaluate the effectiveness of the following counter-argument based on multiple criteria.
            
            Original Argument: {original_argument}
            Counter-Argument: {counter_argument}
            Persuasion Type: {persuasion_type}
            
            Evaluation Criteria:
            {criteria}
            
            Respond in the following JSON format:
            {{
                "criteria_scores": {{
                    "relevance": float (0-1),
                    "persuasiveness": float (0-1),
                    "credibility": float (0-1),
                    "emotional_impact": float (0-1)
                }},
                "strengths": ["list of argument strengths"],
                "weaknesses": ["list of argument weaknesses"],
                "improvement_suggestions": ["list of suggestions for improvement"],
                "reasoning": "detailed explanation of the evaluation"
            }}
            
            Evaluation:"""
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

    def _evaluate_argument(self, original_argument: str, counter_argument: Dict[str, Any], 
                         persuasion_type: str) -> Optional[Dict[str, Any]]:
        """Evaluates the effectiveness of a counter-argument."""
        try:
            # Format criteria for prompt
            criteria_str = "\n".join(
                f"- {criterion}: {info['description']} (Weight: {info['weight']})\n"
                f"  Examples: {', '.join(info['examples'])}"
                for criterion, info in self.criteria.items()
            )
            
            # Format the prompt
            formatted_prompt = self.evaluation_prompt.format(
                original_argument=original_argument,
                counter_argument=counter_argument["counter_argument"],
                persuasion_type=persuasion_type,
                criteria=criteria_str
            )
            
            # Generate evaluation
            chain = self.llm | StrOutputParser()
            evaluation_str = chain.invoke(formatted_prompt)
            
            # Parse the evaluation
            evaluation = json.loads(evaluation_str)
            
            # Validate the evaluation
            if not self._validate_evaluation(evaluation):
                return None
            
            # Calculate weighted overall score
            evaluation["overall_score"] = self._calculate_weighted_score(evaluation["criteria_scores"])
                
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating argument: {str(e)}")
            return None

    def _create_action_graph(self):
        """Creates the action graph for effectiveness evaluation."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add evaluation node
        workflow.add_node("evaluator", self._evaluate_arguments)
        
        # Add edges
        workflow.add_edge(START, "evaluator")
        
        return workflow

    def _evaluate_arguments(self, state: MessageState) -> MessageState:
        """Evaluates all counter-arguments in the state."""
        # Get the original argument
        user_message = state.get("user_message")
        if not user_message:
            return state
            
        # Access content directly from ConvoMessage object
        original_argument = user_message.content if hasattr(user_message, 'content') else ""
        
        # Get all persuasion responses
        persuasion_types = ["pathos", "logos", "ethos"]
        evaluations = {}
        
        for p_type in persuasion_types:
            response = state.get(f"{p_type}_response")
            if response:
                evaluation = self._evaluate_argument(original_argument, response, p_type)
                if evaluation:
                    evaluations[p_type] = evaluation
                else:
                    # Handle evaluation failure
                    error_evaluation = {
                        "overall_score": 0.0,
                        "criteria_scores": {
                            criterion: 0.0 for criterion in self.criteria.keys()
                        },
                        "strengths": [],
                        "weaknesses": [f"Failed to evaluate {p_type} argument"],
                        "improvement_suggestions": [],
                        "reasoning": f"Error evaluating {p_type} argument"
                    }
                    evaluations[p_type] = error_evaluation
        
        # Update state with evaluations
        state["argument_evaluations"] = evaluations
        
        # Find the best argument
        if evaluations:
            best_type = max(evaluations.items(), 
                          key=lambda x: x[1]["overall_score"])[0]
            state["best_argument"] = {
                "type": best_type,
                "evaluation": evaluations[best_type],
                "response": state.get(f"{best_type}_response")
            }
        
        return state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the effectiveness evaluation workflow."""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result

# Register the effectiveness evaluator worker
register_worker(EffectivenessEvaluator) 
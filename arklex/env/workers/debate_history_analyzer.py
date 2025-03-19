import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from arklex.env.workers.default_worker import DefaultWorker
from langgraph.graph import StateGraph, START

logger = logging.getLogger(__name__)

class DebateHistoryAnalyzer(DefaultWorker):
    """Worker for analyzing debate history and effectiveness scores."""
    
    description = "Analyzes past debate performance and recommends the most effective persuasion type based on historical data."
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # Initialize the action graph
        self.action_graph = self._create_action_graph()
        
        self.prompt_template = """You are a debate history analyzer. Your task is to analyze past debate performance and recommend the most effective persuasion type.

Current Topic: {topic}
Current Category: {category}

Debate History:
{history}

Based on the debate history above:
1. Analyze the effectiveness scores for each persuasion type (pathos, logos, ethos)
2. Consider the recency of debates (more recent debates should have higher weight)
3. Consider the topic relevance (debates on similar topics should have higher weight)
4. Recommend the persuasion type that has historically performed best

Output Format:
{
    "analysis": {
        "pathos": {
            "avg_score": float,
            "recent_trend": "increasing" | "decreasing" | "stable",
            "sample_size": int
        },
        "logos": {
            "avg_score": float,
            "recent_trend": "increasing" | "decreasing" | "stable",
            "sample_size": int
        },
        "ethos": {
            "avg_score": float,
            "recent_trend": "increasing" | "decreasing" | "stable",
            "sample_size": int
        }
    },
    "recommendation": {
        "best_type": "pathos" | "logos" | "ethos",
        "confidence": float,
        "reasoning": "string explaining why this type is recommended"
    }
}"""

    def _analyze_history(self, state: MessageState) -> MessageState:
        """Analyze debate history and generate recommendations."""
        # Get current debate information
        user_message = state.get("user_message")
        if not user_message:
            return state
            
        topic = state.get("topic", "general")
        category = state.get("category", "general")
        
        # Get debate history
        history = state.get("debate_history", [])
        
        # Format the prompt
        prompt = self.prompt_template.format(
            topic=topic,
            category=category,
            history=self._format_history(history)
        )
        
        try:
            response = self.llm.invoke(prompt)
            analysis = self._parse_response(response)
            self._validate_analysis(analysis)
            
            # Update state with analysis
            state["debate_analysis"] = analysis
            
        except Exception as e:
            logger.error(f"Error analyzing debate history: {str(e)}")
            # Set default analysis on error
            state["debate_analysis"] = {
                "analysis": {
                    "pathos": {"avg_score": 0.0, "recent_trend": "stable", "sample_size": 0},
                    "logos": {"avg_score": 0.0, "recent_trend": "stable", "sample_size": 0},
                    "ethos": {"avg_score": 0.0, "recent_trend": "stable", "sample_size": 0}
                },
                "recommendation": {
                    "best_type": "logos",  # Default to logos if analysis fails
                    "confidence": 0.5,
                    "reasoning": "Default recommendation due to analysis error"
                }
            }
        
        return state

    def _create_action_graph(self):
        """Creates the action graph for debate history analysis."""
        workflow = StateGraph(MessageState)
        
        # Add analysis node
        workflow.add_node("history_analyzer", self._analyze_history)
        
        # Add edges
        workflow.add_edge(START, "history_analyzer")
        
        return workflow

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the debate history analysis workflow."""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format debate history for the prompt."""
        if not history:
            return "No debate history available."
            
        formatted_history = []
        for debate in history:
            formatted_history.append(
                f"Date: {debate['timestamp']}\n"
                f"Topic: {debate['topic']}\n"
                f"Category: {debate['category']}\n"
                f"Persuasion Type: {debate['persuasion_type']}\n"
                f"Effectiveness Score: {debate['effectiveness_score']}\n"
                f"---"
            )
        return "\n".join(formatted_history)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format."""
        try:
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise

    def _validate_analysis(self, analysis: Dict[str, Any]) -> None:
        """Validate the analysis structure and values."""
        required_keys = ["analysis", "recommendation"]
        analysis_keys = ["pathos", "logos", "ethos"]
        persuasion_keys = ["avg_score", "recent_trend", "sample_size"]
        recommendation_keys = ["best_type", "confidence", "reasoning"]
        
        # Check structure
        for key in required_keys:
            if key not in analysis:
                raise ValueError(f"Missing required key: {key}")
                
        for key in analysis_keys:
            if key not in analysis["analysis"]:
                raise ValueError(f"Missing persuasion type: {key}")
            for subkey in persuasion_keys:
                if subkey not in analysis["analysis"][key]:
                    raise ValueError(f"Missing analysis field: {subkey} for {key}")
                    
        for key in recommendation_keys:
            if key not in analysis["recommendation"]:
                raise ValueError(f"Missing recommendation field: {key}")
                
        # Validate values
        if analysis["recommendation"]["best_type"] not in analysis_keys:
            raise ValueError(f"Invalid persuasion type: {analysis['recommendation']['best_type']}")
            
        if not 0 <= analysis["recommendation"]["confidence"] <= 1:
            raise ValueError("Confidence must be between 0 and 1")

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the debate history analyzer."""
        try:
            # Extract required parameters
            history = kwargs.get("history", [])
            topic = kwargs.get("topic", "")
            category = kwargs.get("category", "")
            
            if not topic or not category:
                raise ValueError("Topic and category are required")
                
            # Analyze history
            analysis = self._analyze_history(history, topic, category)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in debate history analyzer: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 
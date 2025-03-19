import logging
import json
from typing import Dict, Any, List, Optional
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
class DebateHistoryWorker(BaseWorker):
    """A worker that tracks and analyzes debate history to improve argument selection."""
    
    description = "Tracks debate history and analyzes patterns to improve argument effectiveness."

    def __init__(self):
        super().__init__()
        self.stream_response = MODEL.get("stream_response", True)
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize the action graph
        self.action_graph = self._create_action_graph()
        
        # Initialize debate history structure
        self.debate_history = {
            "user_preferences": {},  # Track user's preferred argument types
            "successful_arguments": {},  # Track successful arguments by topic
            "debate_sessions": [],  # Track complete debate sessions
            "effectiveness_metrics": {  # Track effectiveness metrics over time
                "pathos": [],
                "logos": [],
                "ethos": []
            }
        }
        
        self.analysis_prompt = PromptTemplate.from_template(
            """Analyze the debate history and provide insights for improving argument selection.
            
            Current Debate Session:
            Topic: {topic}
            User Argument: {user_argument}
            Selected Response Type: {response_type}
            Effectiveness Score: {effectiveness_score}
            
            Historical Data:
            {historical_data}
            
            Provide analysis in the following JSON format:
            {{
                "user_preferences": {{
                    "preferred_types": ["list of preferred argument types"],
                    "avoided_types": ["list of avoided argument types"],
                    "confidence": float (0-1)
                }},
                "topic_insights": {{
                    "successful_patterns": ["list of successful argument patterns"],
                    "avoided_patterns": ["list of patterns to avoid"],
                    "confidence": float (0-1)
                }},
                "recommendations": {{
                    "argument_type": "suggested argument type",
                    "reasoning": "explanation for recommendation",
                    "confidence": float (0-1)
                }}
            }}
            
            Analysis:"""
        )

    def _update_user_preferences(self, user_id: str, response_type: str, 
                               effectiveness_score: float) -> None:
        """Updates user preferences based on argument effectiveness."""
        if user_id not in self.debate_history["user_preferences"]:
            self.debate_history["user_preferences"][user_id] = {
                "preferred_types": {},
                "total_debates": 0,
                "average_effectiveness": 0.0,
                "debates": []  # Track individual debates with timestamps
            }
        
        user_prefs = self.debate_history["user_preferences"][user_id]
        user_prefs["total_debates"] += 1
        
        # Update preferred types
        if response_type not in user_prefs["preferred_types"]:
            user_prefs["preferred_types"][response_type] = {
                "count": 0,
                "total_score": 0.0
            }
        
        pref_type = user_prefs["preferred_types"][response_type]
        pref_type["count"] += 1
        pref_type["total_score"] += effectiveness_score
        
        # Add debate with timestamp
        user_prefs["debates"].append({
            "type": response_type,
            "score": effectiveness_score,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update average effectiveness
        if user_prefs["debates"]:
            user_prefs["average_effectiveness"] = sum(d["score"] for d in user_prefs["debates"]) / len(user_prefs["debates"])
        else:
            user_prefs["average_effectiveness"] = 0.0

    def _update_successful_arguments(self, topic: str, response_type: str, 
                                   argument: str, effectiveness_score: float) -> None:
        """Updates successful arguments by topic."""
        if topic not in self.debate_history["successful_arguments"]:
            self.debate_history["successful_arguments"][topic] = {
                "arguments": [],
                "effectiveness_by_type": {}
            }
        
        topic_data = self.debate_history["successful_arguments"][topic]
        
        # Update effectiveness by type
        if response_type not in topic_data["effectiveness_by_type"]:
            topic_data["effectiveness_by_type"][response_type] = []
        
        # Add new effectiveness score with timestamp
        topic_data["effectiveness_by_type"][response_type].append({
            "score": effectiveness_score,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add successful argument
        topic_data["arguments"].append({
            "type": response_type,
            "argument": argument,
            "effectiveness": effectiveness_score,
            "timestamp": datetime.now().isoformat()
        })

    def _update_effectiveness_metrics(self, response_type: str, 
                                    effectiveness_score: float) -> None:
        """Updates effectiveness metrics for each argument type."""
        self.debate_history["effectiveness_metrics"][response_type].append({
            "score": effectiveness_score,
            "timestamp": datetime.now().isoformat()
        })

    def _analyze_debate_history(self, topic: str, user_argument: str, 
                              response_type: str, effectiveness_score: float, max_retries: int = 3) -> Dict[str, Any]:
        """Analyzes debate history to provide insights."""
        for attempt in range(max_retries):
            try:
                # Format historical data
                historical_data = {
                    "user_preferences": self.debate_history["user_preferences"],
                    "successful_arguments": self.debate_history["successful_arguments"].get(topic, {}),
                    "effectiveness_metrics": self.debate_history["effectiveness_metrics"]
                }
                
                # Format the prompt
                formatted_prompt = self.analysis_prompt.format(
                    topic=topic,
                    user_argument=user_argument,
                    response_type=response_type,
                    effectiveness_score=effectiveness_score,
                    historical_data=json.dumps(historical_data, indent=2)
                )
                
                # Generate analysis
                chain = self.llm | StrOutputParser()
                analysis_str = chain.invoke(formatted_prompt)
                
                try:
                    # Try to parse the analysis
                    analysis = json.loads(analysis_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from the response
                    start_idx = analysis_str.find("{")
                    end_idx = analysis_str.rfind("}") + 1
                    if start_idx == -1 or end_idx == 0:
                        logger.error(f"Failed to find JSON in response (attempt {attempt + 1}/{max_retries}): {analysis_str}")
                        continue
                    try:
                        analysis = json.loads(analysis_str[start_idx:end_idx])
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from response (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        continue
                
                # Validate the analysis structure
                required_keys = ["user_preferences", "topic_insights", "recommendations"]
                if not all(key in analysis for key in required_keys):
                    logger.error(f"Invalid analysis format (attempt {attempt + 1}/{max_retries})")
                    continue
                
                return analysis
                
            except Exception as e:
                logger.error(f"Error analyzing debate history (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return None
                continue
        
        return None

    def _create_action_graph(self):
        """Creates the action graph for debate history analysis."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add history analysis node
        workflow.add_node("history_analyzer", self._process_debate_history)
        
        # Add edges
        workflow.add_edge(START, "history_analyzer")
        
        return workflow

    def _process_debate_history(self, state: MessageState) -> MessageState:
        """Processes and updates debate history based on current debate session."""
        # Get current debate information
        user_message = state.get("user_message")
        if not user_message:
            return state
            
        user_id = user_message.user_id if hasattr(user_message, 'user_id') else "anonymous"
        topic = state.get("topic", "general")
        user_argument = user_message.content if hasattr(user_message, 'content') else ""
        
        # Get best argument information
        best_argument = state.get("best_argument", {})
        if not best_argument:
            return state
            
        response_type = best_argument.get("type")
        evaluation = best_argument.get("evaluation", {})
        effectiveness_score = evaluation.get("overall_score", 0.0)
        
        # Update history
        self._update_user_preferences(user_id, response_type, effectiveness_score)
        self._update_successful_arguments(topic, response_type, 
                                       best_argument.get("response", {}).get("content", ""),
                                       effectiveness_score)
        self._update_effectiveness_metrics(response_type, effectiveness_score)
        
        # Analyze history
        analysis = self._analyze_debate_history(topic, user_argument, 
                                              response_type, effectiveness_score)
        
        if analysis:
            state["debate_insights"] = analysis
            # Ensure message_flow is a string
            state["message_flow"] = f"Debate history analyzed. Best performing persuasion type: {analysis.get('recommendation', {}).get('best_type', 'unknown')}"
        else:
            state["message_flow"] = "Failed to analyze debate history"
        
        return state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the debate history analysis workflow."""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result

# Register the debate history worker
register_worker(DebateHistoryWorker) 
import logging
import os
import json
from typing import Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.database import DebateDatabase
from arklex.env.prompts import load_prompts

logger = logging.getLogger(__name__)

@register_worker
class DebateDatabaseWorker(BaseWorker):
    """Simple worker for debate-related database operations.
    
    This worker directly handles:
    1. Reading effectiveness scores for persuasion techniques
    2. Recommending the best persuasion type
    3. Updating effectiveness scores after evaluation
    """
    
    description = "Manages database operations for debate agents and recommends persuasion techniques"
    
    # Database path for consistency
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                          "logs", "debate_history.db")
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the worker with minimal dependencies."""
        super().__init__()
        self.config = config or {}
        
        logger.info("Initializing DebateDatabaseWorker")
        
        # Create logs directory if needed
        os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
        
        # Initialize the database - no complex setup
        try:
            logger.info(f"Opening debate database at {self.DB_PATH}")
            self.db = DebateDatabase(db_path=self.DB_PATH)
            
            # Initialize persuasion scores if they don't exist
            self._initialize_persuasion_scores()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            self.db = None
            
        # Initialize LLM for enhanced prompting (if needed)
        try:
            self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
                model=MODEL["model_type_or_path"], timeout=30000
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            self.llm = None
        
        # Create action graph even though we don't actually use it
        self.action_graph = self._create_action_graph()
        
        logger.info("DebateDatabaseWorker initialization complete")
    
    def _create_action_graph(self):
        """Create a minimal action graph that satisfies the abstract method requirement.
        
        We don't actually use this graph for execution, but having it keeps the framework happy.
        """
        logger.info("Creating minimal action graph")
        
        workflow = StateGraph(MessageState)
        
        # Define simple functions for nodes
        def update_score(state):
            return self._update_score(state)
            
        def get_best_type(state):
            return self._get_best_type(state)
            
        # Add nodes
        workflow.add_node("UpdateScore", update_score)
        workflow.add_node("GetBestType", get_best_type)
        
        # Add edges
        workflow.add_edge(START, "GetBestType")
        workflow.add_edge("UpdateScore", "GetBestType")
        
        logger.info("Action graph created successfully (but not used)")
        return workflow
    
    def execute(self, state: MessageState) -> MessageState:
        """Main entry point for execution.
        
        This method determines what operation to perform based on the state
        and executes it directly without a complex graph structure.
        """
        logger.info("DebateDatabaseWorker executing...")
        
        # Initialize default slots
        slots = state.get("slots", {})
        if not slots or "persuasion_techniques" not in slots:
            slots = {
                "persuasion_techniques": {
                    "pathos": {"last_used": None, "scores": []},
                    "logos": {"last_used": None, "scores": []},
                    "ethos": {"last_used": None, "scores": []}
                },
                "current_technique": None,
                "best_technique": "logos"  # Default
            }
        state["slots"] = slots
        
        # Determine what operation to perform
        if self._should_update_score(state):
            logger.info("Updating effectiveness score")
            state = self._update_score(state)
        
        # Always get the best type for recommendations
        logger.info("Getting best persuasion type")
        state = self._get_best_type(state)
        
        # Log results
        best_type = state.get("best_persuasion_type", "logos").upper()
        logger.info(f"💼 DATABASE WORKER COMPLETE - USE {best_type} FOR NEXT RESPONSE")
        
        return state
    
    def _should_update_score(self, state: MessageState) -> bool:
        """Determine if we should update effectiveness scores."""
        # If we have a persuasion response with a score, we should update
        for p_type in ["pathos", "logos", "ethos"]:
            response_key = f"{p_type}_persuasion_response"
            if response_key in state:
                response = state.get(response_key)
                if isinstance(response, dict) and "effectiveness_score" in response:
                    logger.info(f"Found {p_type} persuasion response with score")
                    return True
        
        # If the current_persuasion_type is set, we should update
        current_type = state.get("current_persuasion_type") or state.get("just_used_persuasion_type")
        if current_type in ["pathos", "logos", "ethos"]:
            logger.info(f"Found current persuasion type: {current_type}")
            return True
        
        return False
    
    def _update_score(self, state: MessageState) -> MessageState:
        """Update the effectiveness score for the current persuasion type."""
        if not self.db:
            logger.error("Database not initialized, cannot update scores")
            return state
        
        updated_types = []
        
        # Try to get the persuasion type and score
        current_type = state.get("current_persuasion_type") or state.get("just_used_persuasion_type")
        if current_type in ["pathos", "logos", "ethos"]:
            # Look for a score in the response
            response_key = f"{current_type}_persuasion_response"
            response = state.get(response_key)
            
            if response and isinstance(response, dict) and "effectiveness_score" in response:
                score = response.get("effectiveness_score")
                try:
                    score_float = float(score)
                    if self.db.update_effectiveness_score(current_type, score_float):
                        # Make this extremely visible in the console output
                        logger.info(f"✅ UPDATED SCORE: {current_type.upper()} = {score_float:.2f}")
                        print(f"\n{'='*80}\n📝 DATABASE UPDATE: {current_type.upper()} EFFECTIVENESS = {int(score_float*100)}%\n{'='*80}\n")
                        
                        # Update slots too
                        slots = state.get("slots", {})
                        if "persuasion_techniques" in slots:
                            technique_data = slots["persuasion_techniques"].get(current_type, {})
                            technique_data["scores"] = technique_data.get("scores", []) + [score_float]
                            slots["persuasion_techniques"][current_type] = technique_data
                            slots["current_technique"] = current_type
                            state["slots"] = slots
                            
                        updated_types.append(current_type)
                except (ValueError, TypeError):
                    logger.error(f"Invalid effectiveness score for {current_type}: {score}")
        
        # Check other persuasion responses too
        for p_type in ["pathos", "logos", "ethos"]:
            if p_type in updated_types:
                continue  # Already processed
                
            response_key = f"{p_type}_persuasion_response"
            response = state.get(response_key)
            
            if response and isinstance(response, dict) and "effectiveness_score" in response:
                score = response.get("effectiveness_score")
                try:
                    score_float = float(score)
                    if self.db.update_effectiveness_score(p_type, score_float):
                        # Make this extremely visible in the console output
                        logger.info(f"✅ UPDATED SCORE: {p_type.upper()} = {score_float:.2f}")
                        print(f"\n{'='*80}\n📝 DATABASE UPDATE: {p_type.upper()} EFFECTIVENESS = {int(score_float*100)}%\n{'='*80}\n")
                        
                        # Update slots too
                        slots = state.get("slots", {})
                        if "persuasion_techniques" in slots:
                            technique_data = slots["persuasion_techniques"].get(p_type, {})
                            technique_data["scores"] = technique_data.get("scores", []) + [score_float]
                            slots["persuasion_techniques"][p_type] = technique_data
                            state["slots"] = slots
                            
                        updated_types.append(p_type)
                except (ValueError, TypeError):
                    logger.error(f"Invalid effectiveness score for {p_type}: {score}")
        
        # Set status based on updates
        if updated_types:
            state["status"] = "success"
            state["response"] = f"Updated effectiveness scores for: {', '.join(updated_types)}"
            logger.info(f"Successfully updated scores for: {', '.join(updated_types)}")
        else:
            state["status"] = "no_update"
            state["response"] = "No effectiveness scores were updated"
            logger.info("No persuasion scores were updated")
        
        return state
    
    def _get_best_type(self, state: MessageState) -> MessageState:
        """Get the best persuasion type based on historical effectiveness."""
        if not self.db:
            logger.error("Database not initialized, using default type")
            state["best_persuasion_type"] = "logos"
            state["persuasion_scores"] = {"pathos": 0.5, "logos": 0.5, "ethos": 0.5}
            return state
        
        try:
            # Get current scores for all types
            current_scores = {}
            for p_type in ["pathos", "logos", "ethos"]:
                score = self.db.get_effectiveness_score(p_type)
                current_scores[p_type] = score if score is not None else 0.5
            
            # Find best type
            best_score = -1
            best_type = "logos"  # Default
            for p_type, score in current_scores.items():
                if score is not None and score > best_score:
                    best_score = score
                    best_type = p_type
            
            # Calculate confidence based on score spread
            scores = list(current_scores.values())
            if len(scores) > 1:
                # Higher spread = higher confidence
                max_score = max(scores)
                avg_others = sum([s for s in scores if s != max_score]) / (len(scores) - 1)
                confidence = min(1.0, max(0.0, (max_score - avg_others) * 2))
            else:
                confidence = 0.5  # Default confidence
            
            # Update slots
            slots = state.get("slots", {})
            slots["best_technique"] = best_type
            state["slots"] = slots
            
            # Update state with recommendation
            state["best_persuasion_type"] = best_type
            state["persuasion_scores"] = current_scores
            state["recommendation_confidence"] = confidence
            
            # Make this extremely visible in the console output
            logger.info(f"🔮 BEST PERSUASION TYPE: {best_type.upper()} (score: {current_scores[best_type]:.2f})")
            logger.info(f"📊 SCORES: pathos={current_scores['pathos']:.2f}, logos={current_scores['logos']:.2f}, ethos={current_scores['ethos']:.2f}")
            
            # Print all scores to console for high visibility
            print(f"\n{'='*80}")
            print(f"📊 PERSUASION EFFECTIVENESS SCORES:")
            print(f"   PATHOS: {int(current_scores['pathos']*100)}%")
            print(f"   LOGOS:  {int(current_scores['logos']*100)}%")
            print(f"   ETHOS:  {int(current_scores['ethos']*100)}%")
            print(f"   BEST TYPE FOR NEXT ARGUMENT: {best_type.upper()} ({int(current_scores[best_type]*100)}%)")
            print(f"{'='*80}\n")
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting best persuasion type: {str(e)}")
            # Fallback to default
            state["best_persuasion_type"] = "logos"
            state["persuasion_scores"] = {"pathos": 0.5, "logos": 0.5, "ethos": 0.5}
            state["recommendation_confidence"] = 0.5
            return state
            
    @classmethod
    def get_database_instance(cls) -> Optional[DebateDatabase]:
        """Get a database instance using the same path."""
        try:
            db = DebateDatabase(db_path=cls.DB_PATH)
            return db
        except Exception as e:
            logger.error(f"Error creating database instance: {str(e)}")
            return None

    def _initialize_persuasion_scores(self):
        """Initialize persuasion scores if they don't exist."""
        if not self.db:
            logger.error("Database not available, cannot initialize scores")
            return
            
        try:
            # Check if each persuasion type has an entry
            for p_type in ["pathos", "logos", "ethos"]:
                score = self.db.get_effectiveness_score(p_type)
                
                # If no score exists, add an initial score
                if score is None:
                    logger.info(f"Initializing {p_type} persuasion type with default score")
                    # Start with logos having slightly higher score by default
                    initial_score = 0.6 if p_type == "logos" else 0.5
                    self.db.update_effectiveness_score(p_type, initial_score)
                    logger.info(f"Successfully initialized {p_type} with score {initial_score}")
        except Exception as e:
            logger.error(f"Error initializing persuasion scores: {str(e)}") 
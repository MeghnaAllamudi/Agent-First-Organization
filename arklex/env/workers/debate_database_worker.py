import logging
import os
import json
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.env.tools.debate_db import (
    read_debate_history,
    store_debate_record,
    update_debate_record,
    get_persuasion_stats
)

logger = logging.getLogger(__name__)

@register_worker
class DebateDatabaseWorker(BaseWorker):
    """Worker for debate-related database operations.
    
    This worker handles:
    1. Reading effectiveness scores for persuasion techniques
    2. Recommending the best persuasion type
    3. Storing and updating debate records
    4. Analyzing debate history
    """
    
    description = "Manages database operations for debate agents and recommends persuasion techniques"
    
    # Database path for consistency
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                          "logs", "debate_history.db")
    
    # Track initialization state
    _DB_INITIALIZED = False
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the worker."""
        super().__init__()
        self.config = config or {}
        
        logger.info("Initializing DebateDatabaseWorker")
        
        # Create logs directory if needed
        os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
        
        # Initialize the database - no complex setup
        try:
            logger.info(f"Opening debate database at {self.DB_PATH}")
            self.db = DebateDatabase(db_path=self.DB_PATH)
            
            # Initialize persuasion scores if they don't exist, but only once
            if not DebateDatabaseWorker._DB_INITIALIZED:
                self._initialize_persuasion_scores()
                DebateDatabaseWorker._DB_INITIALIZED = True
                logger.info("Database initialized successfully (first time)")
            else:
                logger.info("Using already initialized database")
            
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
        
        # Create action graph
        self.action_graph = self._create_action_graph()
        logger.info("DebateDatabaseWorker initialization complete")
    
    def execute(self, msg_state: MessageState) -> MessageState:
        """Execute database operations based on the message state.
        
        Args:
            msg_state: Current message state
            
        Returns:
            Updated message state
        """
        operation = msg_state.get("operation", "read")
        logger.info(f"Executing operation: {operation}")
        
        try:
            # Handle different operations
            if operation == "store":
                # Store a new debate record
                result = store_debate_record(
                    persuasion_technique=msg_state.get("persuasion_technique"),
                    effectiveness_score=msg_state.get("effectiveness_score"),
                    suggestion=msg_state.get("suggestion")
                )
                msg_state["store_result"] = result
                
            elif operation == "update":
                # Update an existing record
                result = update_debate_record(
                    record_id=msg_state.get("record_id"),
                    persuasion_technique=msg_state.get("persuasion_technique"),
                    effectiveness_score=msg_state.get("effectiveness_score"),
                    suggestion=msg_state.get("suggestion")
                )
                msg_state["update_result"] = result
                
            elif operation == "read":
                # Read debate history
                records = read_debate_history(
                    limit=msg_state.get("limit", 100),
                    persuasion_type=msg_state.get("persuasion_type")
                )
                msg_state["debate_records"] = records
                
            elif operation == "stats":
                # Get persuasion statistics
                stats = get_persuasion_stats(
                    technique=msg_state.get("technique")
                )
                msg_state["persuasion_stats"] = stats
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            msg_state["status"] = "success"
            return msg_state
            
        except Exception as e:
            logger.error(f"Error executing database operation: {str(e)}")
            msg_state["status"] = "error"
            msg_state["error"] = str(e)
            return msg_state

    def _create_action_graph(self) -> StateGraph:
        """Create the action graph for this worker."""
        # Simple graph that just executes the worker
        workflow = StateGraph(StateGraph.basic_validate)
        
        # Add single node that processes the state
        workflow.add_node("process", self.execute)
        
        # Add start and end edges
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        return workflow.compile()

    def _reset_all_scores(self, db_connection=None):
        """Reset all persuasion type scores to exactly 0.33.
        
        This method first clears all history to ensure a fresh start, then
        adds new records with exactly 0.33 scores for each persuasion type.
        """
        print("🔄 RESETTING ALL PERSUASION SCORES TO 0.33")
        
        if db_connection is None:
            db_connection = self.db
        
        try:
            # Clear all existing records first to ensure a true reset
            db_connection.clear_all_persuasion_history()
            print(f"   ✅ Cleared all previous persuasion history")
            
            # Now set fresh scores of exactly 0.33
            for persuasion_type in ["logos", "pathos", "ethos"]:
                self._update_effectiveness_score(persuasion_type, 0.33, db_connection)
                print(f"   ✅ Reset {persuasion_type.upper()} score to exactly 0.33")
            
            # Verify the scores were set correctly
            for persuasion_type in ["logos", "pathos", "ethos"]:
                score = self._get_effectiveness_score(persuasion_type, db_connection)
                print(f"   ✓ Verified {persuasion_type.upper()} score is {score:.2f}")
            
            return True
        except Exception as e:
            print(f"❌ ERROR RESETTING SCORES: {str(e)}")
            return False

    def _initialize_db(self, conn):
        """Initialize the database schema if it doesn't exist."""
        import traceback
        
        try:
            # Create the persuasion_effectiveness table if it doesn't exist
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persuasion_effectiveness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persuasion_type TEXT NOT NULL UNIQUE,
                    effectiveness_score REAL NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            return True
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            traceback.print_exc()
            return False
    
    def _get_persuasion_score(self, conn, persuasion_type):
        """Get the effectiveness score for a persuasion type directly from SQLite."""
        import traceback
        
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT effectiveness_score FROM persuasion_effectiveness WHERE persuasion_type = ?", 
                (persuasion_type,)
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
            else:
                # If no score exists yet, return the default
                return 0.33
        except Exception as e:
            print(f"Error getting persuasion score: {str(e)}")
            traceback.print_exc()
            return 0.33
    
    def _update_persuasion_score(self, conn, persuasion_type, score):
        """Update the effectiveness score for a persuasion type directly in SQLite."""
        import traceback
        
        try:
            # Ensure score is a valid float between 0 and 1
            score = float(score)
            if score < 0:
                score = 0.0
            elif score > 1:
                score = 1.0
                
            cursor = conn.cursor()
            
            # Check if the persuasion type already exists
            cursor.execute(
                "SELECT COUNT(*) FROM persuasion_effectiveness WHERE persuasion_type = ?",
                (persuasion_type,)
            )
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Update existing record
                cursor.execute(
                    "UPDATE persuasion_effectiveness SET effectiveness_score = ?, updated_at = CURRENT_TIMESTAMP WHERE persuasion_type = ?",
                    (score, persuasion_type)
                )
            else:
                # Insert new record
                cursor.execute(
                    "INSERT INTO persuasion_effectiveness (persuasion_type, effectiveness_score) VALUES (?, ?)",
                    (persuasion_type, score)
                )
                
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating persuasion score: {str(e)}")
            traceback.print_exc()
            return False

    def _update_effectiveness_scores(self, msg_state, db_connection=None):
        """Update persuasion effectiveness scores."""
        try:
            print(f"\n================================================================================")
            print(f"💾 DATABASE UPDATE OPERATION")
            print(f"================================================================================\n")
            
            should_close = False
            if db_connection is None:
                db_connection = self._get_db_connection()
                should_close = True
            
            # Get the persuasion type and effectiveness score
            persuasion_type = None
            effectiveness_score = None
            
            # Method 1: From state variables (most reliable)
            if "current_persuasion_type" in msg_state and "evaluated_effectiveness_score" in msg_state:
                persuasion_type = msg_state["current_persuasion_type"]
                effectiveness_score = msg_state["evaluated_effectiveness_score"]
                print(f"Found persuasion_type ({persuasion_type}) and effectiveness_score ({effectiveness_score}) in state variables")
            
            # Method 2: From effectiveness_evaluation object
            elif "effectiveness_evaluation" in msg_state and isinstance(msg_state["effectiveness_evaluation"], dict):
                eval_obj = msg_state["effectiveness_evaluation"]
                if "overall_score" in eval_obj:
                    effectiveness_score = eval_obj["overall_score"]
                    # Try to find persuasion type from other state variables
                    if "current_persuasion_type" in msg_state:
                        persuasion_type = msg_state["current_persuasion_type"]
                    print(f"Found effectiveness_score ({effectiveness_score}) in effectiveness_evaluation object")
            
            # Default persuasion type if needed
            if effectiveness_score is not None and not persuasion_type:
                persuasion_type = "logos"  # Default
                print(f"No persuasion type found, defaulting to: {persuasion_type}")
            
            # Get user's persuasion type using consistent approach
            user_persuasion_type = None
            counter_type = None
            
            # First priority: Check 'user_persuasion_type' directly 
            if "user_persuasion_type" in msg_state:
                user_persuasion_type = msg_state.get("user_persuasion_type")
                print(f"✅ FOUND USER PERSUASION TYPE: {user_persuasion_type.upper()}")
            
            # Second priority: Check 'argument_classification' for dominant_type
            elif "argument_classification" in msg_state and isinstance(msg_state["argument_classification"], dict):
                classification = msg_state["argument_classification"]
                if "dominant_type" in classification:
                    dominant_type = classification["dominant_type"]
                    # Map the dominant type to persuasion_type
                    type_mapping = {
                        "emotional": "pathos",
                        "logical": "logos", 
                        "ethical": "ethos"
                    }
                    
                    if dominant_type in type_mapping:
                        user_persuasion_type = type_mapping[dominant_type]
                        # Set in state for other workers to use
                        msg_state["user_persuasion_type"] = user_persuasion_type
                        print(f"✅ DERIVED USER PERSUASION TYPE: {user_persuasion_type.upper()} (from {dominant_type})")
            
            # If we found user_persuasion_type, determine the counter type
            if user_persuasion_type:
                # Standard counter-type mapping
                counter_mapping = {
                    "pathos": "logos",   # Counter emotional with logical
                    "logos": "ethos",    # Counter logical with ethical
                    "ethos": "pathos"    # Counter ethical with emotional
                }
                
                if user_persuasion_type in counter_mapping:
                    counter_type = counter_mapping[user_persuasion_type]
                    msg_state["counter_persuasion_type"] = counter_type
                    print(f"✅ SET COUNTER TYPE: {counter_type.upper()} (to counter user's {user_persuasion_type.upper()})")
            
            # Update the database if we have the necessary information
            updated_scores = {}
            if persuasion_type and effectiveness_score is not None:
                # Update the bot's persuasion technique effectiveness
                print(f"UPDATING DATABASE: Writing effectiveness score {effectiveness_score:.2f} for {persuasion_type}")
                update_result = self._update_effectiveness_score(persuasion_type, effectiveness_score, db_connection)
                if update_result:
                    print(f"✅ DATABASE UPDATE SUCCESSFUL: {persuasion_type} effectiveness = {effectiveness_score:.2f}")
                else:
                    print(f"❌ DATABASE UPDATE FAILED: Could not update {persuasion_type} effectiveness")
                
                # Get updated scores for all persuasion types
                for p_type in ["logos", "pathos", "ethos"]:
                    updated_scores[p_type] = self._get_effectiveness_score(p_type, db_connection)
                
                # Determine the best persuasion type based on updated scores
                best_type = self._get_best_persuasion_type(updated_scores)
                
                # IMPORTANT: Override best_persuasion_type with counter_type if available
                if counter_type:
                    print(f"\n🧠 PERSUASION STRATEGY: USING {counter_type.upper()} TO COUNTER USER'S {user_persuasion_type.upper()}")
                    best_type = counter_type
                else:
                    print(f"\n🧠 PERSUASION STRATEGY: USING DATABASE BEST TYPE {best_type.upper()} ({int(updated_scores.get(best_type, 0) * 100)}%)")
                
                # Print summary
                print(f"\n📊 UPDATED EFFECTIVENESS SCORES:")
                print(f"   PATHOS: {int(updated_scores.get('pathos', 0) * 100)}%")
                print(f"   LOGOS:  {int(updated_scores.get('logos', 0) * 100)}%")
                print(f"   ETHOS:  {int(updated_scores.get('ethos', 0) * 100)}%")
                
                # Update the state with consistent variables
                msg_state["persuasion_scores"] = updated_scores
                msg_state["best_persuasion_type"] = best_type
                msg_state["next_persuasion_type"] = best_type
            else:
                print(f"❌ COULD NOT UPDATE DATABASE: Missing persuasion_type or effectiveness_score")
                # Get current scores anyway
                for p_type in ["logos", "pathos", "ethos"]:
                    updated_scores[p_type] = self._get_effectiveness_score(p_type, db_connection)
                
                # Still provide some value by reading current scores
                best_type = self._get_best_persuasion_type(updated_scores)
                
                # Override with counter_type if available
                if counter_type:
                    best_type = counter_type
                
                msg_state["persuasion_scores"] = updated_scores
                msg_state["best_persuasion_type"] = best_type
                msg_state["next_persuasion_type"] = best_type
            
            # Close connection
            if should_close and hasattr(db_connection, "close"):
                db_connection.close()
                
            return msg_state
            
        except Exception as e:
            logger.error(f"Error updating effectiveness scores: {str(e)}")
            print(f"❌ DATABASE UPDATE ERROR: {str(e)}")
            return msg_state

    def _read_effectiveness_scores(self, msg_state):
        """Read current effectiveness scores from the database."""
        print("\n🔍 READING EFFECTIVENESS SCORES FROM DATABASE")
        
        try:
            # Connect to database
            db_connection = self.db
            
            # Read current scores
            scores = {}
            for persuasion_type in ["logos", "pathos", "ethos"]:
                scores[persuasion_type] = self._get_effectiveness_score(persuasion_type, db_connection)
            
            # Print current scores
            print("\n📊 EFFECTIVENESS SCORES:")
            for p_type, score in scores.items():
                print(f"   - {p_type.upper()}: {score:.2f} ({int(score * 100)}%)")
            
            # Get best persuasion type based on scores
            best_type = self._get_best_persuasion_type(scores)
            
            # Check if we have user persuasion type from ArgumentClassifier
            user_persuasion_type = None
            counter_type = None
            
            # First priority: Check 'user_persuasion_type' directly from argument classifier
            if "user_persuasion_type" in msg_state:
                user_persuasion_type = msg_state.get("user_persuasion_type")
                print(f"✅ FOUND USER PERSUASION TYPE: {user_persuasion_type.upper()}")
            
            # Second priority: Check 'argument_classification' for dominant_type
            elif "argument_classification" in msg_state and isinstance(msg_state["argument_classification"], dict):
                classification = msg_state["argument_classification"]
                if "dominant_type" in classification:
                    dominant_type = classification["dominant_type"]
                    # Map the dominant type to persuasion_type
                    type_mapping = {
                        "emotional": "pathos",
                        "logical": "logos", 
                        "ethical": "ethos"
                    }
                    
                    if dominant_type in type_mapping:
                        user_persuasion_type = type_mapping[dominant_type]
                        # Set in state for other workers to use
                        msg_state["user_persuasion_type"] = user_persuasion_type
                        print(f"✅ DERIVED USER PERSUASION TYPE: {user_persuasion_type.upper()} (from {dominant_type})")
            
            # If we found user_persuasion_type, determine the counter type
            if user_persuasion_type:
                # Standard counter-type mapping
                counter_mapping = {
                    "pathos": "logos",   # Counter emotional with logical
                    "logos": "ethos",    # Counter logical with ethical
                    "ethos": "pathos"    # Counter ethical with emotional
                }
                
                if user_persuasion_type in counter_mapping:
                    counter_type = counter_mapping[user_persuasion_type]
                    msg_state["counter_persuasion_type"] = counter_type
                    print(f"✅ SET COUNTER TYPE: {counter_type.upper()} (to counter user's {user_persuasion_type.upper()})")
            
            # First priority: Use the counter type based on user's persuasion type
            if counter_type:
                best_type = counter_type
                print(f"\n🧠 PERSUASION STRATEGY: USING {counter_type.upper()} TO COUNTER USER'S {user_persuasion_type.upper()}")
            else:
                # Fallback: Use the database's best performing type
                print(f"\n🧠 PERSUASION STRATEGY: USING DATABASE BEST TYPE {best_type.upper()} ({int(scores[best_type] * 100)}%)")
            
            # Update message state
            msg_state["best_persuasion_type"] = best_type
            
            # Also set next_persuasion_type for the persuasion worker
            msg_state["next_persuasion_type"] = best_type
            
            return msg_state
        except Exception as e:
            logger.error(f"Error reading database: {str(e)}")
            msg_state["best_persuasion_type"] = "logos"  # Default fallback
            return msg_state
    
    def _initialize_persuasion_scores(self, db_connection=None):
        """Initialize persuasion scores if they don't exist."""
        if not self.db:
            logger.error("Database not available, cannot initialize scores")
            return
            
        try:
            print(f"\n================================================================================")
            print(f"💾 INITIALIZING DATABASE WITH DEFAULT SCORES")
            print(f"================================================================================\n")
            
            # Check if scores already exist first
            existing_scores = {}
            scores_exist = False
            
            for p_type in ["pathos", "logos", "ethos"]:
                score = self.db.get_effectiveness_score(p_type)
                if score is not None:
                    existing_scores[p_type] = score
                    scores_exist = True
            
            # Only clear and reset if no scores exist yet
            if not scores_exist:
                print(f"🆕 NO SCORES FOUND: Performing first-time initialization")
                # Clear all previous history for first-time setup
                self.db.clear_all_persuasion_history()
                print(f"✅ CLEARED DATABASE FOR FIRST-TIME SETUP")
                
                # Initialize each persuasion type with exactly 0.33
                for p_type in ["pathos", "logos", "ethos"]:
                    # All persuasion types start with equal score - no bias
                    initial_score = 0.33
                    self.db.update_effectiveness_score(p_type, initial_score)
                    print(f"✅ INITIALIZED {p_type.upper()} with score {initial_score:.2f}")
                    logger.info(f"Successfully initialized {p_type} with score {initial_score}")
            else:
                print(f"📊 SCORES ALREADY EXIST: Using existing persuasion scores")
                for p_type, score in existing_scores.items():
                    print(f"✅ {p_type.upper()} already has score {score:.2f}")
                
            # Get current scores
            scores = {}
            for p_type in ["pathos", "logos", "ethos"]:
                scores[p_type] = self.db.get_effectiveness_score(p_type)
                
            # Show current values
            print(f"\n================================================================================")
            print(f"💾 DATABASE INITIAL STATE:")
            print(f"   PATHOS: {int(scores.get('pathos', 0.33) * 100)}%")
            print(f"   LOGOS:  {int(scores.get('logos', 0.33) * 100)}%")
            print(f"   ETHOS:  {int(scores.get('ethos', 0.33) * 100)}%")
            print(f"================================================================================\n")
                
        except Exception as e:
            logger.error(f"Error initializing persuasion scores: {str(e)}")

    def _get_db_connection(self):
        """Get a database connection."""
        from arklex.utils.database import DebateDatabase
        import os
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Set up the database path
        db_path = os.path.join(logs_dir, "debate_history.db")
        
        # Initialize the database
        return DebateDatabase(db_path=db_path)
        
    def _get_effectiveness_score(self, persuasion_type, db_connection=None):
        """Get the effectiveness score for a persuasion type."""
        should_close = False
        if db_connection is None:
            db_connection = self._get_db_connection()
            should_close = True
            
        score = db_connection.get_effectiveness_score(persuasion_type)
        
        if should_close and hasattr(db_connection, "close"):
            db_connection.close()
            
        return score if score is not None else 0.33
        
    def _update_effectiveness_score(self, persuasion_type, score, db_connection=None):
        """Update the effectiveness score for a persuasion type."""
        should_close = False
        if db_connection is None:
            db_connection = self._get_db_connection()
            should_close = True
            
        result = db_connection.update_effectiveness_score(persuasion_type, score)
        
        if should_close and hasattr(db_connection, "close"):
            db_connection.close()
            
        return result
        
    def _get_best_persuasion_type(self, scores):
        """Determine the best persuasion type based on scores."""
        if not scores:
            return "logos"  # Default
            
        best_score = -1
        best_type = "logos"
        
        for p_type, score in scores.items():
            if score is not None and score > best_score:
                best_score = score
                best_type = p_type
                
        return best_type 

    def _resolve_db_path(self):
        """Resolve the path to the database file."""
        import os
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Set up the database path
        db_path = os.path.join(logs_dir, "debate_history.db")
        return db_path
        
    def _connect_to_db(self, db_path):
        """Connect to the database."""
        import sqlite3
        import traceback
        
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(db_path)
            return conn
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            traceback.print_exc()
            return None 

    def _store_effectiveness_scores(self, msg_state):
        """Store the effectiveness scores in the database."""
        print(f"\n================================================================================")
        print(f"🧰 DATABASE WORKER: STORING EFFECTIVENESS SCORES")
        print(f"================================================================================\n")
        
        try:
            # Debug message for state
            print(f"🔍 AVAILABLE STATE KEYS: {list(msg_state.keys())}")
            
            # Extract evaluated_effectiveness_score
            effectiveness_score = None
            if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "evaluated_effectiveness_score" in msg_state["metadata"]["global_state"]:
                effectiveness_score = msg_state["metadata"]["global_state"]["evaluated_effectiveness_score"]
                print(f"🌐 FOUND effectiveness_score IN GLOBAL STATE: {effectiveness_score}")
            
            if not effectiveness_score and "evaluated_effectiveness_score" in msg_state:
                effectiveness_score = msg_state["evaluated_effectiveness_score"]
                print(f"📄 FOUND effectiveness_score IN DIRECT STATE: {effectiveness_score}")
            
            if effectiveness_score is None:
                print("⚠️ No effectiveness score found in state, skipping database update")
                return msg_state
                
            # Try to find the user's persuasion type
            user_persuasion_type = None
            
            # First check for global state
            if "metadata" in msg_state and "global_state" in msg_state["metadata"]:
                global_state = msg_state["metadata"]["global_state"]
                if "user_persuasion_type" in global_state:
                    user_persuasion_type = global_state["user_persuasion_type"]
                    print(f"🌐 FOUND user_persuasion_type in global state: {user_persuasion_type}")
                
            # Then check for direct state
            if not user_persuasion_type and "user_persuasion_type" in msg_state:
                user_persuasion_type = msg_state["user_persuasion_type"]
                print(f"📄 FOUND user_persuasion_type IN DIRECT STATE: {user_persuasion_type}")
                        
            # Find the counter persuasion type used
            counter_persuasion_type = None
            # Check global state first
            if "metadata" in msg_state and "global_state" in msg_state["metadata"]:
                global_state = msg_state["metadata"]["global_state"]
                if "counter_persuasion_type" in global_state:
                    counter_persuasion_type = global_state["counter_persuasion_type"]
                    print(f"🌐 FOUND counter_persuasion_type in global state: {counter_persuasion_type}")
            
            # Then check direct state
            if not counter_persuasion_type and "counter_persuasion_type" in msg_state:
                counter_persuasion_type = msg_state["counter_persuasion_type"]
                print(f"📄 FOUND counter_persuasion_type IN DIRECT STATE: {counter_persuasion_type}")
                
            # Store persuasion suggestions if available
            persuasion_suggestions = None
            if "persuasion_suggestions" in msg_state:
                persuasion_suggestions = msg_state["persuasion_suggestions"]
                print(f"📄 FOUND persuasion_suggestions in state")
            
            # If we have all necessary information, update the database
            if user_persuasion_type and counter_persuasion_type and effectiveness_score is not None:
                try:
                    # Connect to database
                    db = self._get_database()
                    if not db:
                        print("⚠️ Cannot connect to database")
                        return msg_state
                    
                    # Generate a timestamp for this record
                    timestamp = datetime.now().isoformat()
                    
                    # Create a record
                    record = {
                        "user_persuasion_type": user_persuasion_type,
                        "counter_persuasion_type": counter_persuasion_type,
                        "effectiveness_score": effectiveness_score,
                        "timestamp": timestamp
                    }
                    
                    # Add persuasion suggestions if available
                    if persuasion_suggestions:
                        record["persuasion_suggestions"] = persuasion_suggestions
                    
                    # Insert the record
                    db.insert_effectiveness_score(record)
                    
                    print(f"✅ STORED EFFECTIVENESS SCORE IN DATABASE: {user_persuasion_type.upper()} argued against with {counter_persuasion_type.upper()} scored {effectiveness_score:.2f}")
                    if persuasion_suggestions:
                        print(f"✅ STORED PERSUASION SUGGESTIONS IN DATABASE for {persuasion_suggestions['recommended_type'].upper()}")
                    
                    # Calculate average effectiveness for each persuasion type
                    self._update_average_scores(db, msg_state)
                    
                    return msg_state
                    
                except Exception as e:
                    print(f"⚠️ Error storing effectiveness score: {str(e)}")
                    return msg_state
            else:
                print(f"⚠️ Missing information for database update:")
                print(f"   User persuasion type: {user_persuasion_type}")
                print(f"   Counter persuasion type: {counter_persuasion_type}")
                print(f"   Effectiveness score: {effectiveness_score}")
                return msg_state
                
        except Exception as e:
            print(f"⚠️ Error in _store_effectiveness_scores: {str(e)}")
            return msg_state

    def _get_best_persuasion_type(self, msg_state: MessageState) -> MessageState:
        """Get the best persuasion type based on historical effectiveness."""
        print(f"\n================================================================================")
        print(f"🧰 DATABASE WORKER: DETERMINING BEST PERSUASION TYPE")
        print(f"================================================================================\n")
        
        # Try to find the user's persuasion type first
        user_persuasion_type = None
        
        # First check for global state
        if "metadata" in msg_state and "global_state" in msg_state["metadata"]:
            global_state = msg_state["metadata"]["global_state"]
            if "user_persuasion_type" in global_state:
                user_persuasion_type = global_state["user_persuasion_type"]
                print(f"🌐 FOUND user_persuasion_type in global state: {user_persuasion_type}")
                
        # Then check for direct state
        if not user_persuasion_type and "user_persuasion_type" in msg_state:
            user_persuasion_type = msg_state["user_persuasion_type"]
            print(f"📄 FOUND user_persuasion_type IN DIRECT STATE: {user_persuasion_type}")
            
        # If still no user persuasion type found, check argument classification
        if not user_persuasion_type and "argument_classification" in msg_state:
            try:
                classification = msg_state.get("argument_classification", {})
                if isinstance(classification, dict) and "dominant_type" in classification:
                    dominant_type = classification.get("dominant_type")
                    type_mapping = {
                        "emotional": "pathos",
                        "logical": "logos",
                        "ethical": "ethos"
                    }
                    
                    if dominant_type in type_mapping:
                        user_persuasion_type = type_mapping[dominant_type]
                        print(f"📝 DERIVED user_persuasion_type from argument_classification: {user_persuasion_type}")
            except Exception as e:
                print(f"⚠️ Error parsing argument classification: {str(e)}")
                
        if not user_persuasion_type:
            print(f"⚠️ No user_persuasion_type found, defaulting to logos")
            user_persuasion_type = "logos"
            
        try:
            # Connect to database
            db = self._get_database()
            if not db:
                print("⚠️ Cannot connect to database")
                msg_state["best_persuasion_type"] = "logos"  # Default
                return msg_state
                
            # Get the best persuasion type to counter the user's type
            scores = db.get_average_effectiveness_scores()
            
            # Get the most recent persuasion suggestions
            recent_suggestions = db.get_most_recent_persuasion_suggestions()
            if recent_suggestions:
                print(f"✅ FOUND recent persuasion suggestions in database")
                recommended_type = recent_suggestions.get("recommended_type", "")
                if recommended_type:
                    print(f"📊 RECOMMENDED TYPE from suggestions: {recommended_type.upper()}")
                    
                # Add the suggestions to the state for the persuasion worker
                msg_state["db_persuasion_suggestions"] = recent_suggestions
                
                # If the suggestions recommend a specific type, prioritize it
                if recommended_type in ["pathos", "logos", "ethos"]:
                    msg_state["best_persuasion_type"] = recommended_type
                    print(f"✅ USING RECOMMENDATION: {recommended_type.upper()} from suggestions")
                    
                    # Set in global state too
                    self._set_in_global_state(msg_state, "best_persuasion_type", recommended_type)
                    return msg_state
            
            # If no recommendation from suggestions, use scores
            if scores:
                print(f"📊 AVERAGE EFFECTIVENESS SCORES:")
                for p_type, score in scores.items():
                    print(f"   - {p_type.upper()}: {score:.2f}")
                
                # Find the highest scoring persuasion type
                best_type = max(scores.items(), key=lambda x: x[1])[0] if scores else "logos"
                best_score = scores.get(best_type, 0.0)
                
                # Only recommend if score is decent
                if best_score > 0.4:
                    msg_state["best_persuasion_type"] = best_type
                    msg_state["persuasion_scores"] = scores
                    msg_state["recommendation_confidence"] = best_score
                    
                    # Set in global state too
                    self._set_in_global_state(msg_state, "best_persuasion_type", best_type)
                    self._set_in_global_state(msg_state, "persuasion_scores", scores)
                    
                    print(f"✅ RECOMMENDING: {best_type.upper()} with score {best_score:.2f}")
                    return msg_state
                else:
                    print(f"⚠️ No persuasion type with high enough score found")
            else:
                print(f"⚠️ No effectiveness scores found in database")
                
            # Default mapping if we don't have good data
            counter_mapping = {
                "pathos": "logos",   # Counter emotional with logical
                "logos": "ethos",    # Counter logical with ethical
                "ethos": "pathos"    # Counter ethical with emotional
            }
            
            default_type = counter_mapping.get(user_persuasion_type, "logos")
            msg_state["best_persuasion_type"] = default_type
            
            # Set in global state too
            self._set_in_global_state(msg_state, "best_persuasion_type", default_type)
            
            print(f"✅ USING DEFAULT MAPPING: {default_type.upper()} to counter {user_persuasion_type.upper()}")
            return msg_state
            
        except Exception as e:
            print(f"⚠️ Error determining best persuasion type: {str(e)}")
            msg_state["best_persuasion_type"] = "logos"  # Default
            return msg_state 

    def process_record(self, action: str, msg_state: MessageState) -> MessageState:
        """Process the record based on the action."""
        print("\n================================================================================")
        print(f"🗄️ DEBATE DATABASE WORKER - Action: {action.upper()}")
        print(f"================================================================================\n")
        
        # Call action method based on the received action
        if action == "update":
            # Check for global state variables
            self._ensure_global_state(msg_state)
            
            # Check if we have an effectiveness score to save
            if "evaluated_effectiveness_score" in msg_state or (
                "metadata" in msg_state and 
                "global_state" in msg_state["metadata"] and 
                "evaluated_effectiveness_score" in msg_state["metadata"]["global_state"]
            ):
                print(f"✅ Found effectiveness score to save")
                return self._update_effectiveness_scores(msg_state)
            else:
                print(f"⚠️ No effectiveness score found - skipping database update")
                return msg_state
                
        elif action == "read":
            # This is a read operation - get the best persuasion type
            return self._get_best_persuasion_type(msg_state)
        else:
            print(f"⚠️ UNKNOWN ACTION: {action}")
            return msg_state

    def _ensure_global_state(self, msg_state: MessageState) -> None:
        """Ensure global state is initialized in the message state."""
        if "metadata" not in msg_state:
            msg_state["metadata"] = {}
        if "global_state" not in msg_state["metadata"]:
            msg_state["metadata"]["global_state"] = {}
        print(f"✅ Ensured global state exists in message state")

    def _update_effectiveness_scores(self, msg_state: MessageState) -> MessageState:
        """Update the effectiveness scores in the database."""
        print(f"🔄 UPDATING EFFECTIVENESS SCORES IN DATABASE")
        
        # First, identify the bot's persuasion type that we're evaluating
        bot_persuasion_type = None
        
        # Check global state first (higher priority)
        if "metadata" in msg_state and "global_state" in msg_state["metadata"]:
            if "current_persuasion_type" in msg_state["metadata"]["global_state"]:
                bot_persuasion_type = msg_state["metadata"]["global_state"]["current_persuasion_type"]
                print(f"🌐 Found bot_persuasion_type in global state (current_persuasion_type): {bot_persuasion_type}")
            elif "counter_persuasion_type" in msg_state["metadata"]["global_state"]:
                bot_persuasion_type = msg_state["metadata"]["global_state"]["counter_persuasion_type"]
                print(f"🌐 Found bot_persuasion_type in global state (counter_persuasion_type): {bot_persuasion_type}")
        
        # Check direct state (lower priority)
        if not bot_persuasion_type:
            if "current_persuasion_type" in msg_state:
                bot_persuasion_type = msg_state["current_persuasion_type"]
                print(f"📄 Found bot_persuasion_type in direct state (current_persuasion_type): {bot_persuasion_type}")
            elif "counter_persuasion_type" in msg_state:
                bot_persuasion_type = msg_state["counter_persuasion_type"]
                print(f"📄 Found bot_persuasion_type in direct state (counter_persuasion_type): {bot_persuasion_type}")
        
        # Default to logos if we can't find the persuasion type
        if not bot_persuasion_type:
            bot_persuasion_type = "logos"
            print(f"⚠️ Could not determine bot's persuasion type, defaulting to: {bot_persuasion_type}")
            
            # Store for future reference
            msg_state["current_persuasion_type"] = bot_persuasion_type
            msg_state["metadata"]["global_state"]["current_persuasion_type"] = bot_persuasion_type
        
        # Next, extract the effectiveness score
        effectiveness_score = None
        
        # Check global state first (higher priority)
        if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "evaluated_effectiveness_score" in msg_state["metadata"]["global_state"]:
            effectiveness_score = msg_state["metadata"]["global_state"]["evaluated_effectiveness_score"]
            print(f"🌐 Found effectiveness_score in global state: {effectiveness_score}")
        
        # Check direct state (lower priority)
        elif "evaluated_effectiveness_score" in msg_state:
            effectiveness_score = msg_state["evaluated_effectiveness_score"]
            print(f"📄 Found effectiveness_score in direct state: {effectiveness_score}")
        
        # Skip update if no score is found
        if effectiveness_score is None:
            print(f"⚠️ No effectiveness score found - skipping database update")
            return msg_state
        
        # Get the user's persuasion type (for logging purposes)
        user_persuasion_type = None
        
        # Check global state first (higher priority)
        if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "user_persuasion_type" in msg_state["metadata"]["global_state"]:
            user_persuasion_type = msg_state["metadata"]["global_state"]["user_persuasion_type"]
            print(f"🌐 Found user_persuasion_type in global state: {user_persuasion_type}")
        
        # Check direct state (lower priority)
        elif "user_persuasion_type" in msg_state:
            user_persuasion_type = msg_state["user_persuasion_type"]
            print(f"📄 Found user_persuasion_type in direct state: {user_persuasion_type}")
        
        # Now update the database with the effectiveness score
        try:
            # Connect to the database
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            # Create a timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert the effectiveness score
            cursor.execute(
                """
                INSERT INTO effectiveness_scores (
                    persuasion_type, 
                    score, 
                    timestamp, 
                    user_persuasion_type
                ) VALUES (?, ?, ?, ?)
                """, 
                (bot_persuasion_type, effectiveness_score, timestamp, user_persuasion_type)
            )
            
            # Commit the changes
            conn.commit()
            print(f"✅ SAVED IN DATABASE: Bot's {bot_persuasion_type.upper()} strategy had {effectiveness_score:.2f} effectiveness against user's {user_persuasion_type.upper() if user_persuasion_type else 'unknown'} strategy")
            
            # Close the connection
            conn.close()
            
            # Store the next counter persuasion type (from the EffectivenessEvaluator)
            next_persuasion_type = None
            
            # Check global state first
            if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "counter_persuasion_type" in msg_state["metadata"]["global_state"]:
                next_persuasion_type = msg_state["metadata"]["global_state"]["counter_persuasion_type"]
                print(f"🌐 Found next_persuasion_type in global state: {next_persuasion_type}")
            
            # Check direct state
            elif "counter_persuasion_type" in msg_state:
                next_persuasion_type = msg_state["counter_persuasion_type"]
                print(f"📄 Found next_persuasion_type in direct state: {next_persuasion_type}")
            
            # Use user's persuasion type as fallback for next counter
            if not next_persuasion_type:
                next_persuasion_type = user_persuasion_type if user_persuasion_type else "logos"
                print(f"⚠️ No next_persuasion_type found, using user's persuasion type: {next_persuasion_type}")
                
                # Store for future reference
                msg_state["counter_persuasion_type"] = next_persuasion_type
                msg_state["metadata"]["global_state"]["counter_persuasion_type"] = next_persuasion_type
            
            # Set this as the current persuasion type for next response
            msg_state["current_persuasion_type"] = next_persuasion_type
            msg_state["metadata"]["global_state"]["current_persuasion_type"] = next_persuasion_type
            print(f"🌟 SET current_persuasion_type to {next_persuasion_type.upper()} for next response")
            
            return msg_state
            
        except Exception as e:
            print(f"⚠️ ERROR UPDATING DATABASE: {str(e)}")
            traceback.print_exc()
            return msg_state

    def _get_best_persuasion_type(self, msg_state: MessageState) -> MessageState:
        """Get the best persuasion type based on historical effectiveness."""
        print(f"🔍 GETTING BEST PERSUASION TYPE FROM DATABASE")
        
        # First check if we already have a recommended persuasion type from the EffectivenessEvaluator
        counter_persuasion_type = None
        
        # Check global state first (higher priority)
        if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "counter_persuasion_type" in msg_state["metadata"]["global_state"]:
            counter_persuasion_type = msg_state["metadata"]["global_state"]["counter_persuasion_type"]
            print(f"🌐 Found counter_persuasion_type in global state: {counter_persuasion_type}")
        
        # Check direct state (lower priority)
        elif "counter_persuasion_type" in msg_state:
            counter_persuasion_type = msg_state["counter_persuasion_type"]
            print(f"📄 Found counter_persuasion_type in direct state: {counter_persuasion_type}")
        
        # If we have a counter_persuasion_type, use it
        if counter_persuasion_type:
            print(f"✅ Using recommended counter_persuasion_type: {counter_persuasion_type.upper()}")
            
            # Set as current_persuasion_type for the next response
            msg_state["current_persuasion_type"] = counter_persuasion_type
            
            # Store in global state
            if "metadata" not in msg_state:
                msg_state["metadata"] = {}
            if "global_state" not in msg_state["metadata"]:
                msg_state["metadata"]["global_state"] = {}
            msg_state["metadata"]["global_state"]["current_persuasion_type"] = counter_persuasion_type
            
            return msg_state
        
        # If we don't have a counter_persuasion_type, check the database
        try:
            # Connect to the database
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            # Get the average effectiveness score for each persuasion type
            cursor.execute(
                """
                SELECT 
                    persuasion_type, 
                    AVG(score) as avg_score,
                    COUNT(*) as count
                FROM 
                    effectiveness_scores
                GROUP BY 
                    persuasion_type
                ORDER BY 
                    avg_score DESC
                """
            )
            
            # Get the results
            results = cursor.fetchall()
            
            # Close the connection
            conn.close()
            
            # If we have results, use the best persuasion type
            if results:
                print(f"📊 PERSUASION TYPE EFFECTIVENESS RANKING:")
                for persuasion_type, avg_score, count in results:
                    print(f"   • {persuasion_type.upper()}: {avg_score:.2f} (from {count} records)")
                
                # Get the best persuasion type
                best_persuasion_type = results[0][0]
                print(f"🌟 BEST PERSUASION TYPE: {best_persuasion_type.upper()}")
                
                # Set as current_persuasion_type for the next response
                msg_state["current_persuasion_type"] = best_persuasion_type
                
                # Store in global state
                if "metadata" not in msg_state:
                    msg_state["metadata"] = {}
                if "global_state" not in msg_state["metadata"]:
                    msg_state["metadata"]["global_state"] = {}
                msg_state["metadata"]["global_state"]["current_persuasion_type"] = best_persuasion_type
                
                return msg_state
            else:
                # No results, return a default persuasion type
                default_type = "logos"
                print(f"⚠️ No effectiveness data in database - using default: {default_type.upper()}")
                
                # Set as current_persuasion_type for the next response
                msg_state["current_persuasion_type"] = default_type
                
                # Store in global state
                if "metadata" not in msg_state:
                    msg_state["metadata"] = {}
                if "global_state" not in msg_state["metadata"]:
                    msg_state["metadata"]["global_state"] = {}
                msg_state["metadata"]["global_state"]["current_persuasion_type"] = default_type
                
                return msg_state
        except Exception as e:
            print(f"⚠️ ERROR READING DATABASE: {str(e)}")
            traceback.print_exc()
            
            # Return a default persuasion type
            default_type = "logos"
            print(f"⚠️ Error accessing database - using default: {default_type.upper()}")
            
            # Set as current_persuasion_type for the next response
            msg_state["current_persuasion_type"] = default_type
            
            # Store in global state
            if "metadata" not in msg_state:
                msg_state["metadata"] = {}
            if "global_state" not in msg_state["metadata"]:
                msg_state["metadata"]["global_state"] = {}
            msg_state["metadata"]["global_state"]["current_persuasion_type"] = default_type
            
            return msg_state 
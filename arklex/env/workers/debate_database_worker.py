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
    
    def __init__(self):
        """Initialize the worker."""
        super().__init__()
        
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
    

    def _create_action_graph(self) -> StateGraph:
        """Create the action graph for this worker."""
        # Simple graph that just executes the worker
        workflow = StateGraph(StateGraph.basic_validate)
        
        # Add single node that processes the state
        workflow.add_node("process", self.execute)
        
        # Add start and end edges
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        return workflow 

    def execute(self, state: MessageState) -> MessageState:
        graph = self.action_graph.compile()
        result = graph.invoke(state)
        return result
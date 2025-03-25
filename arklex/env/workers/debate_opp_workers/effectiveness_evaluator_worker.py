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
from langgraph.graph import StateGraph, START
from arklex.utils.model_provider_config import PROVIDER_MAP
# Remove the general prompts import if it exists
# from arklex.env.prompts import load_prompts

import importlib.util
import os
import sys

# Get the path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, root_dir)

logger = logging.getLogger(__name__)


@register_worker
class EffectivenessEvaluator(BaseWorker):
    """A worker that evaluates the effectiveness of counter-arguments."""
    
    description = "This must run after the ArgumentClassifier after each user response. It Evaluates the effectiveness of counter-arguments based on multiple criteria."

    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000, 
            temperature = 0.0
        )
        self.action_graph = self._create_action_graph(); 
        logger.info("Effectiveness Evaluator initialized successfully")

    def _effectiveness_score(self, state: MessageState):
        """
        Evaluate how effective the bot's argument was based on the user's response.
        Calculates what percentage of the bot's argument the user agreed with.
        
        Args:
            state: The current message state containing bot and user messages
            
        Returns:
            Dict containing effectiveness score (0-100), reasoning, strengths, weaknesses,
            and improvement suggestions
        """
        # Get the bot and user messages from state
        bot_message = state.get("bot_message", None)
        user_message = state.get("user_message", None)
        
        
        # Create the evaluation prompt
        evaluation_prompt = f"""
        You are an expert in debate and persuasion analysis. Your task is to evaluate how effective the bot's argument was by determining what percentage of the argument the user agreed with or acknowledged.
        
        Please analyze the following exchange carefully:
        
        BOT'S ARGUMENT: 
        {bot_message}
        
        USER'S RESPONSE: 
        {user_message}
        
        First, identify the key points and claims made in the bot's argument. Then, carefully analyze the user's response to determine:
        
        1. Which specific points from the bot's argument the user explicitly agreed with or acknowledged as valid
        2. Which specific points the user implicitly accepted by not challenging them
        3. Which specific points the user disputed, rejected, or countered with their own arguments
        4. Any new points or direction changes the user introduced
        
        Based on your analysis, calculate an EFFECTIVENESS SCORE from 0-100 representing the percentage of the bot's argument the user agreed with:
        - 0-20: User strongly rejected almost all points
        - 21-40: User disagreed with most points but accepted some minor aspects
        - 41-60: User showed mixed agreement/disagreement
        - 61-80: User agreed with most points with some reservations
        - 81-100: User showed strong agreement with almost all points
        
        Pay special attention to:
        - Explicit statements of agreement ("you're right about X", "I agree that Y")
        - Implicit agreement (user builds upon bot's points rather than countering them)
        - Qualified agreement ("yes, but...")
        - Complete rejection or topic changes (indicates low effectiveness)
        
        Give just the number for the effectiveness score between 0-100. 
        """
        
        
        # Parse the evaluation response
        response = self.llm.invoke(evaluation_prompt)
        result_text = response.content.strip()
        
        # Ensure the score is a float between 0 and 100
        score = float(result_text)
        
        state["effectiveness_score"] = score
        
        print("EFFECTIVENESS EVALUATOR")
        print(f"effective_score: {state['effectiveness_score']}")
        print("==========================================================")
        
    def _create_action_graph(self):
        """Create a processing flow for effectiveness evaluation."""
        workflow = StateGraph(MessageState)
        workflow = StateGraph(MessageState)
        
        # Add classification node
        workflow.add_node("effectiveness_calc", self._effectiveness_score)
        
        # Add edges
        workflow.add_edge(START, "effectiveness_calc")
        
        return workflow

    def execute(self, state: MessageState) -> MessageState:
        graph = self.action_graph.compile()
        result = graph.invoke(state)
        return result
    
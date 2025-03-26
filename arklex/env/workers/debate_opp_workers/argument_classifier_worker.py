import logging
import json
import re
import traceback
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState, ConvoMessage, Slot
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP

# Import argument_classifier_prompt from prompts_for_debate_opp
import importlib.util
import os
import sys

# Get the path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, root_dir)

# Import directly from the root directory
from arklex.env.prompts_for_debate_opp import argument_classifier_prompt

logger = logging.getLogger(__name__)


@register_worker
class ArgumentClassifier(BaseWorker):
    """A worker that classifies user and bot arguments into different types."""
    
    description = "This should analyze each user and bot argument to classify them as emotional (pathos), logical (logos), or ethical (ethos). It should only run once per user repsonse"
    
    # Class-level counter to track execution calls
    _execution_count = 0
    _last_classified_content = None
    # Track which requests have been processed (keyed by request ID)
    _processed_requests = set()
    result_text = None

    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000, 
            temperature = 0.0
        )
        self.action_graph = self._create_action_graph()
    
    def _classify_arguments(self, state: MessageState):
        """Classifies all arguments in the state using llm."""
        
        user_message = state["user_message"].message
        bot_message =  bot_message = state["slots"]["bot_message"][0].value
        user_classification_prompt = f"""
            Analyze the following argument and classify it as primarily using one of these persuasion types:
            - pathos: Appeals to emotion, feelings, and personal experience
            - logos: Appeals to logic, facts, data, and rational thinking
            - ethos: Appeals to credibility, authority, ethics, and moral principles
            
            Argument: {user_message}
            
            Please respond with a single word: pathos, logos, or ethos.
            """
            
        bot_classification_prompt = f"""
            Analyze the following argument and classify it as primarily using one of these persuasion types:
            - pathos: Appeals to emotion, feelings, and personal experience
            - logos: Appeals to logic, facts, data, and rational thinking
            - ethos: Appeals to credibility, authority, ethics, and moral principles
            
            Argument: {bot_message}
            
            Please respond with a single word: pathos, logos, or ethos.
            """
            
        # Get classification from LLM
        user_result = self.llm.invoke(user_classification_prompt)
        user_result_text = user_result.content.strip()
        
        bot_result = self.llm.invoke(bot_classification_prompt)
        bot_result_text = bot_result.content.strip()
        
        state["slots"]["bot_classification"] = [Slot(
                name = "bot_classification", 
                type = "string", 
                value = bot_result_text, 
                enum = [],
                description = "This is the value that holds the classification of the bot's argument.", 
                prompt = "", 
                required = False, 
                verified = True)] 
        
        state["slots"]["user_classification"] = [Slot(
                name = "user_classification", 
                type = "string", 
                value = user_result_text, 
                enum = [],
                description = "This is the value that holds the classification of the user's argument.", 
                prompt = "", 
                required = False, 
                verified = True)] 
        
        print("ARGUMENT CLASSIFIER")
        #print(state)
        print("user_message: " + user_message[:30])
        #print("bot_message:  " + bot_message[:30])
        print("user_classification: " + state["slots"]["user_classification"][0].value)
        print("bot_classification: " + state["slots"]["bot_classification"][0].value) 
        print("==========================================================")
        
    def _create_action_graph(self):
        """Creates the action graph for argument classification."""
    
        workflow = StateGraph(MessageState)
        
        # Add classification node
        workflow.add_node("classifier", self._classify_arguments)
        
        # Add edges
        workflow.add_edge(START, "classifier")
        
        return workflow

    def execute(self, state: MessageState) -> MessageState:
        graph = self.action_graph.compile()
        result = graph.invoke(state)
        return result
    
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

from langgraph.graph import StateGraph, START
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP


logger = logging.getLogger(__name__)


@register_worker
class PersuasionWorker(BaseWorker):
    """This must run after the EffectivenessEvaluator. This worker generates counter-arguments using different persuasion techniques."""
    
    description = "This must run after the EffectivenessEvaluator. This worker generates counter-arguments using different persuasion techniques."
    
    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000, 
            temperature = 0.0
        )
        self.action_graph = self._create_action_graph()
        logger.info("PersuasionWorker initialized successfully")
        
    def _get_persuasive_strat(self, state: MessageState):
        """
        Generates the next persuasive argument for the bot based on effectiveness score.
        
        If effectiveness score < 50%, generate argument matching user's classification.
        Otherwise, continue with the bot's current classification strategy.
        
        Args:
            state: The current message state containing bot and user messages,
                  classifications, and effectiveness score
        """
        # Get relevant state information
        user_message = state.get("user_message", {}).message
            
        bot_message = state.get("bot_message", state.get("response", ""))
            
        # Get the topic from state - use either the stored topic or extract it from the bot message
        topic = state.get("topic", "")
        if not topic and bot_message:
            # Try to extract a topic from the first paragraph of the bot message
            paragraphs = bot_message.split('\n\n')
            if paragraphs:
                # Get first paragraph and limit to a reasonable length
                first_para = paragraphs[0].strip()
                if len(first_para) > 100:
                    topic = first_para[:100] + "..."
                else:
                    topic = first_para
            
        # Log the topic for debugging
        logger.info(f"Debate topic: {topic}")
            
        user_classification = state.get("user_classification", "logos")
        bot_classification = state.get("bot_classification", "logos")
        effectiveness_score = state.get("effectiveness_score", 50.0)
        
        # Log the values for debugging
        logger.info(f"User classification: {user_classification}")
        logger.info(f"Bot classification: {bot_classification}")
        logger.info(f"Effectiveness score: {effectiveness_score}")
        
        # Determine which classification to use for next argument
        classification_to_use = user_classification if effectiveness_score < 50 else bot_classification
        logger.info(f"Using classification: {classification_to_use}")
        
        persuasion_descriptions = {
            "pathos": "emotional appeals, personal stories, vivid imagery, and language that evokes feelings",
            "logos": "logical reasoning, facts, statistics, data, and structured arguments based on evidence",
            "ethos": "ethical principles, credibility, authority, expertise, moral values, and character-based appeals"
        }
        
        # Create the persuasive argument generation prompt
        prompt = f"""
        You are an expert debater skilled in various persuasion techniques. Generate a persuasive counter-argument to continue our debate.

        DEBATE TOPIC: {topic}

        USER'S LATEST ARGUMENT: 
        {user_message}

        YOUR PREVIOUS ARGUMENT:
        {bot_message}

        PERSUASION STRATEGY: Your counter-argument should primarily use {classification_to_use}-based persuasion, which focuses on {persuasion_descriptions.get(classification_to_use, "balanced reasoning")}.

        EFFECTIVENESS ANALYSIS: The effectiveness score of your previous argument was {effectiveness_score:.0f}%. 
        {"Since this was less than 50, you need to adapt by matching the user's persuasion style." if effectiveness_score < 50 else "Since this was effective, you should continue with your current persuasion approach."}

        IMPORTANT GUIDELINES:
        1. Maintain your opposition to the topic - your job is to challenge the user
        2. Use specific {classification_to_use}-based appeals in your arguments
        3. Be direct, specific, and use concrete examples to support your position
        4. Address specific points from the user's argument
        5. Keep your response conversational (about 3-4 paragraphs)
        6. End with a question or statement that encourages further discussion

        Generate a persuasive counter-argument now:
        """
        
        # Generate the counter-argument
        response = self.llm.invoke(prompt)
        counter_argument = response.content.strip()
        
        # Update state with new counter-argument
        state["new_counter_argument"] = counter_argument
        state["response"] = counter_argument
        state["bot_message"] = counter_argument
        state["persuasion_strategy_used"] = classification_to_use
        
        print("PERSUASION_WORKER")
        print("classification to use: " + classification_to_use)
        print("==========================================================")
        
    def _create_action_graph(self):
        """Create a processing flow for persuasion strategy."""
        workflow = StateGraph(MessageState)
        
        # Add persuasion strategy node
        workflow.add_node("persuasion_strategy", self._get_persuasive_strat)
        
        # Add edges
        workflow.add_edge(START, "persuasion_strategy")
        
        return workflow
        
    def execute(self, msg_state: MessageState):
        
        graph = self._create_action_graph().compile()
        result = graph.invoke(msg_state)
        return result 
    
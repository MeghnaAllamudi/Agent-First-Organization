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
from arklex.utils.graph_state import MessageState, Slot
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP


logger = logging.getLogger(__name__)


@register_worker
class PersuasionWorker(BaseWorker):
    """This must run after the EffectivenessEvaluator. This worker generates counter-arguments using different persuasion techniques."""
    
    description = "This worker generates counter-arguments using different persuasion techniques."
    
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
            
        bot_message = state["slots"]["bot_message"][0].value
            
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
            
        user_classification = state["slots"]["user_classification"][0].value
        bot_classification = state["slots"]["bot_classification"][0].value
        effectiveness_score = state["slots"]["effectiveness_score"][0].value
        
        # Log the values for debugging
        logger.info(f"User classification: {user_classification}")
        logger.info(f"Bot classification: {bot_classification}")
        logger.info(f"Effectiveness score: {effectiveness_score}")
        
        # Determine which classification to use for next argument
        classification_to_use = user_classification if effectiveness_score < 50 else bot_classification
        logger.info(f"Using classification: {classification_to_use} because effectiveness score is {effectiveness_score}")
        
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
        1. Acknowledge some of the points the user made if they are good points while maintaining your opposition in a respectful way to help challenge the user
        2. Make sure the argument is engaging and make sure it flows with the user's response, like an ongoing conversation. Be sure to address each of the user's points.
        3. Use specific {classification_to_use}-based appeals in your arguments
        4. Be direct, specific, and use concrete examples to support your position
        5. Address specific points from the user's argument and it's okay to agree with some of what the user said if they bring up good points
        6. Keep your response conversational (about 1-2 paragraphs)
        7. Show the user you are listening to what they are saying while maintaining your opposition to challenge their counter arguments
        8. End with a question or statement that encourages further discussion

        Generate a persuasive counter-argument now:
        """
        
        # Generate the counter-argument
        response = self.llm.invoke(prompt)
        counter_argument = response.content.strip()
        
        # Update state with new counter-argument
        #state["response"] = counter_argument
        
        state["slots"]["persuasion_counter"] = [Slot(
                name = "persuasion_counter", 
                type = "string", 
                value = counter_argument, 
                enum = [],
                description = "This is the counter argument that the bot should use next", 
                prompt = "", 
                required = False, 
                verified = True)] 
        
        print("PERSUASION_WORKER")
        print("classification to use: " + classification_to_use)
        print("next response: " + state["slots"]["persuasion_counter"][0].value[:20])
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
    
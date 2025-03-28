from typing import Dict, Any, Optional, Union
import logging
import random
import os
import pickle
from pathlib import Path
from datetime import datetime
import json

from langgraph.graph import StateGraph, START
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState, Slot
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
#from arklex.utils.loaders.debate_loader import DebateLoader
from arklex.utils.loader import Loader, CrawledURLObject

logger = logging.getLogger(__name__)

@register_worker
class DebateRAGWorker(BaseWorker):
    """A worker that uses RAG to enhance debate responses with relevant information."""
    
    description = "Only run this DebateRAGWorker once per conversation. Uses RAG to pick a debate topic and enhance debate responses with relevant information from a knowledge base."
    
    def __init__(self):
        super().__init__()
        self.stream_response = MODEL.get("stream_response", True)
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000, temperature = 0.0
        )
        self.rag_counter = 0
        
        # Load pre-built documents from common output directories
        self.documents = []
        possible_dirs = ["./examples/debate_opponent"]
        
        for output_dir in possible_dirs:
            filepath = os.path.join(output_dir, "documents.pkl")
            if os.path.exists(filepath):
                try:
                    logger.info(f"Loading RAG documents from {filepath}")
                    with open(filepath, "rb") as f:
                        self.documents = pickle.load(f)
                    logger.info(f"Loaded {len(self.documents)} RAG documents")

                except Exception as e:
                    logger.error(f"Error loading RAG documents from {filepath}: {str(e)}")
        
        if not self.documents:
            logger.warning("No pre-built RAG documents found. The worker may fail.")
        
        self.debate_prompt = PromptTemplate.from_template(
            """Based on the following debate topic and context, generate a structured argument.
            Consider both sides of the debate and provide evidence-based reasoning.
            
            Topic: {topic}
            Context: {context}
            
            Generate a response that includes:
            1. The topic of debate 
            2. The stance that you are taking 
            3. One argument in support of it that falls under logos persusaive strategy
            
            Keep this to a maximum of 50 words
            
            Response:"""
        )
        
        logger.info("DebateRAGWorker initialized successfully")
        
        self.action_graph = self._create_action_graph()
        
    def _pick_debate_topic(self, msg_state: MessageState) -> MessageState:
        """Executes the debate workflow."""
        new_value = self.rag_counter
        if "slots" in msg_state and "rag_counter" not in msg_state["slots"]:
            msg_state["slots"]["rag_counter"] = [Slot(
                name = "rag_counter", 
                type = "string", 
                value = self.rag_counter, 
                enum = [],
                description = "This is a counter to ensure that the DebateRAGWorker only runs once per conversation.", 
                prompt = "", 
                required = False, 
                verified = True)]  
        else:
            new_value = msg_state["slots"]["rag_counter"][0].value + 1
            msg_state["slots"]["rag_counter"][0].value = new_value
        
        if(new_value < 2):
            try:
                if not self.documents:
                    raise Exception("No documents available for debate topics")
                
                # Simplified approach: just pick a random document
                topic_doc = random.choice(self.documents)
                logger.info(f"Selected random topic URL: {topic_doc.url}")
                
                # Process the topic content
                topic_content = topic_doc.content.split('\n')
                
                # Extract a topic name from the first non-empty line or URL
                for line in topic_content:
                    if line and line.strip():
                        topic_name = line.strip()
                        break
                else:
                    # Fallback if no content found
                    topic_name = topic_doc.url.split('/')[-1].replace('-', ' ').title()
            
                    
                # Get all non-empty lines as context
                context_lines = [line for line in topic_content if line and line.strip()]
                context = "\n".join(context_lines[:20])  # Limit to 20 lines
                
                # Format the response using the debate prompt
                formatted_response = self.debate_prompt.format(
                    topic=topic_name,
                    context=context
                )
                
                # Generate the final response
                chain = self.llm | StrOutputParser()
                response = chain.invoke(formatted_response)
                
                # Update the state with the formatted response
                msg_state["message_flow"] = response
                
                if "slots" in msg_state and "bot_message" not in msg_state["slots"]:
                    msg_state["slots"]["bot_message"] = [Slot(
                        name = "bot_message", 
                        type = "string", 
                        value = response, 
                        enum = [],
                        description = "This is the last bot message before the last user response to help with argument classification", 
                        prompt = "", 
                        required = False, 
                        verified = True)]  

                msg_state["response"] = response  # Set response field too for completeness
                
                logger.info("Debate processing completed successfully")
                #jsonString = json.dumps(msg_state)
                print("RAG")
                print("current bot_message: " + msg_state["trajectory"][-2]["content"][:30] )
                print("next bot_message: " + msg_state["response"][:30])
                print("==========================================================")
                
            except Exception as e:
                logger.error(f"Error in debate processing: {str(e)}", exc_info=True)
                return msg_state
        else: 
            print("SKIPPING RAG")
            print("==========================================")
                
        return msg_state

    def _create_action_graph(self) -> StateGraph:
        """Creates the action graph for debate handling."""
        workflow = StateGraph(MessageState)
        workflow.add_node("debate_handler", self._pick_debate_topic)
        workflow.add_edge(START, "debate_handler")
        return workflow
    
    def execute(self, state: MessageState) -> MessageState:
        graph = self.action_graph.compile()
        result = graph.invoke(state)
        return result

    
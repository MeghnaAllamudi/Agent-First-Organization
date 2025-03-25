import logging
from typing import Any
import traceback

from arklex.env.workers.debate_opp_workers.debate_history_db_worker import DebateHistoryDatabaseWorker
from arklex.env.workers.worker import register_worker
from arklex.env.workers.message_worker import MessageWorker
from arklex.utils.graph_state import MessageState, ConvoMessage
from langchain_openai import ChatOpenAI
from arklex.env.prompts import load_prompts
from langchain.prompts import PromptTemplate
from arklex.utils.utils import chunk_string
from langchain_core.output_parsers import StrOutputParser
from arklex.utils.model_config import MODEL
from langgraph.graph import StateGraph, START, END
from arklex.utils.model_provider_config import PROVIDER_MAP

# Import the necessary worker classes
from arklex.env.workers.debate_opp_workers.debate_rag_worker import DebateRAGWorker
from arklex.env.workers.debate_opp_workers.argument_classifier_worker import ArgumentClassifier
from arklex.env.workers.debate_opp_workers.effectiveness_evaluator_worker import EffectivenessEvaluator
from arklex.env.workers.debate_opp_workers.persuasion_worker import PersuasionWorker

logger = logging.getLogger(__name__)

@register_worker
class DebateMessageWorker(MessageWorker):
    """A specialized MessageWorker for debate opponents that incorporates persuasion strategies."""
    
    description = "Delivers debate responses using the most effective persuasion type based on user interactions from the PersuasionWorker after each user response."

    def __init__(self):
        super().__init__()
        logger.info("DebateMessageWorker initialized")
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000, temperature = 0.0
        )
        
        self.action_graph = self._create_action_graph()
    
        
    def _create_action_graph(self):
        """
        Create a processing flow for debate interactions with the following chain:
        debate message worker -> debate rag worker -> argument classifier -> 
        effectiveness score -> persuasion type -> debate message worker
        """
        workflow = StateGraph(MessageState)
        
        # Initialize worker instances
        rag_worker = DebateRAGWorker()
        arg_classifier = ArgumentClassifier()
        effectiveness_evaluator = EffectivenessEvaluator()
        persuasion_worker = PersuasionWorker()
        database_worker = DebateHistoryDatabaseWorker()
        
        # Add worker nodes in the chain
        workflow.add_node("rag", rag_worker.execute)
        workflow.add_node("classifier", arg_classifier.execute)
        workflow.add_node("effectiveness", effectiveness_evaluator.execute)
        workflow.add_node("persuasion", persuasion_worker.execute)
        workflow.add_node("debate-update", database_worker.execute)
        
        workflow.add_edge(START,"rag")
        workflow.add_edge("rag","classifier")
        workflow.add_edge("classifier","effectiveness")
        workflow.add_edge("effectiveness","persuasion")
        workflow.add_edge("persuasion","debate-update")
        
        return workflow 
    
    def execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        
        # Ensure the message_flow is set to the latest counter-argument for proper output
        if result.get("new_counter_argument"):
            result["message_flow"] = result["new_counter_argument"]
        elif result.get("response"):
            result["message_flow"] = result["response"]
        elif result.get("bot_message"):
            result["message_flow"] = result["bot_message"]
            
        logger.info(f"DebateMessageWorker returning response: {result.get('message_flow', '')[:100]}...")
        return result
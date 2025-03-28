import logging
from typing import Any
import traceback

from arklex.env.workers.debate_opp_workers.debate_history_db_worker import DebateHistoryDatabaseWorker
from arklex.env.workers.worker import register_worker
from arklex.env.workers.message_worker import MessageWorker
from arklex.types import EventType
from arklex.utils.graph_state import MessageState, ConvoMessage, Slot
from langchain_openai import ChatOpenAI
from arklex.env.prompts import load_prompts
from langchain.prompts import PromptTemplate
from arklex.utils.utils import chunk_string
from langchain_core.output_parsers import StrOutputParser
from arklex.utils.model_config import MODEL
from langgraph.graph import StateGraph, START, END
from arklex.utils.model_provider_config import PROVIDER_MAP

# Import the necessary worker classes
from arklex.env.workers.debate_opp_workers.argument_classifier_worker import ArgumentClassifier
from arklex.env.workers.debate_opp_workers.effectiveness_evaluator_worker import EffectivenessEvaluator
from arklex.env.workers.debate_opp_workers.persuasion_worker import PersuasionWorker

logger = logging.getLogger(__name__)

@register_worker
class DebateMessageWorker(MessageWorker):
    """A specialized MessageWorker for debate opponents that incorporates persuasion strategies."""
    
    #description = "Delivers debate responses using the most effective persuasion type based on user interactions from the PersuasionWorker after each user response."
    description = "Runs after each user response in the conversation. It determines what the next response should be."

    def __init__(self):
        super().__init__()
        logger.info("DebateMessageWorker initialized")
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000, temperature = 0.0
        )
        
        self.action_graph = self._create_action_graph()
        
    def generator(self, state: MessageState) -> MessageState:
        # get the input message
        print("DEBATE MESSAGE WORKER GENERATOR")
        print("==============================================")
        user_message = state['user_message']
        orchestrator_message = state['orchestrator_message']
        message_flow = state.get('response', "") + "\n" + state.get("message_flow", "")
        
        # get the orchestrator message content
        orch_msg_content = "None" if not orchestrator_message.message else orchestrator_message.message
        orch_msg_attr = orchestrator_message.attribute
        direct_response = orch_msg_attr.get('direct_response', False)
        if direct_response:
            state["message_flow"] = ""
            state["response"] = orch_msg_content
            return state
        
        prompts = load_prompts(state["bot_config"])
        if message_flow and message_flow != "\n":
            prompt = PromptTemplate.from_template(prompts["message_flow_generator_prompt"])
            input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history, "context": message_flow})
        else:
            prompt = PromptTemplate.from_template(prompts["message_generator_prompt"])
            input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history})
        logger.info(f"Prompt: {input_prompt.text}")
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        answer = final_chain.invoke(chunked_prompt)

        state["message_flow"] = ""
        if "slots" in state and "persuasion_counter" not in state["slots"]:
            state["response"] = answer
        else: 
            state["response"] = state["slots"]["persuasion_counter"][0].value
        return state
    
    def choose_generator(self, state: MessageState):
        if state["is_stream"]:
            return "stream_generator"
        return "generator"
    
    def stream_generator(self, state: MessageState) -> MessageState:
        # get the input message
        print("DEBATE MESSAGE WORKER")
        print("==============================================")
        user_message = state['user_message']
        orchestrator_message = state['orchestrator_message']
        message_flow = state.get('response', "") + "\n" + state.get("message_flow", "")

        # get the orchestrator message content
        orch_msg_content = "None" if not orchestrator_message.message else orchestrator_message.message
        orch_msg_attr = orchestrator_message.attribute
        direct_response = orch_msg_attr.get('direct_response', False)
        if direct_response:
            state["message_flow"] = ""
            state["response"] = orch_msg_content
            return state
        
        prompts = load_prompts(state["bot_config"])
        if message_flow and message_flow != "\n":
            prompt = PromptTemplate.from_template(prompts["message_flow_generator_prompt"])
            input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history, "context": message_flow})
        else:
            prompt = PromptTemplate.from_template(prompts["message_generator_prompt"])
            input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history})
        logger.info(f"Prompt: {input_prompt.text}")
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        answer = ""
        for chunk in final_chain.stream(chunked_prompt):
            answer += chunk
            state["message_queue"].put({"event": EventType.CHUNK.value, "message_chunk": chunk})

        state["message_flow"] = ""
        #state["slots"]["bot_message"][0].value = state["response"]
        state["slots"]["persuasion_counter"] 
        if "slots" in state and "persuasion_counter" not in state["slots"]:
            state["response"] = answer
        else: 
            state["response"] = state["slots"]["persuasion_counter"][0].value
        return state
        
    def _create_action_graph(self):
        """
        Create a processing flow for debate interactions with the following chain:
        debate message worker -> debate rag worker -> argument classifier -> 
        effectiveness score -> persuasion type -> debate message worker
        """
        workflow = StateGraph(MessageState)
        
        # Initialize worker instances
        arg_classifier = ArgumentClassifier()
        effectiveness_evaluator = EffectivenessEvaluator()
        persuasion_worker = PersuasionWorker()
        database_worker = DebateHistoryDatabaseWorker()
        
        # Add worker nodes in the chain
        workflow.add_node("classifier", arg_classifier.execute)
        workflow.add_node("effectiveness", effectiveness_evaluator.execute)
        workflow.add_node("persuasion", persuasion_worker.execute)
        workflow.add_node("debate-update", database_worker.execute)
        workflow.add_node("generator", self.generator)
        workflow.add_node("stream_generator", self.stream_generator)
        
        workflow.add_edge(START,"classifier")
        workflow.add_edge("classifier","effectiveness")
        workflow.add_edge("effectiveness","persuasion")
        workflow.add_edge("persuasion","debate-update")
        workflow.add_conditional_edges("debate-update", self.choose_generator)
        workflow.add_edge("generator",END)
        workflow.add_edge("stream_generator",END)
    
        
        return workflow 
    
    def execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result
import logging
from typing import Any

from arklex.env.workers.worker import register_worker
from arklex.env.workers.message_worker import MessageWorker
from arklex.utils.graph_state import MessageState
from arklex.env.prompts import load_prompts
from langchain.prompts import PromptTemplate
from arklex.utils.utils import chunk_string
from langchain_core.output_parsers import StrOutputParser
from arklex.utils.model_config import MODEL

logger = logging.getLogger(__name__)

@register_worker
class DebateMessageWorker(MessageWorker):
    """A specialized MessageWorker for debate opponents that incorporates persuasion strategies."""
    
    description = "Delivers debate responses using the most effective persuasion type based on user interactions."

    def __init__(self):
        super().__init__()
        logger.info("DebateMessageWorker initialized")
        
    def _get_persuasion_info(self, persuasion_type: str) -> str:
        """Generate formatted info about the current persuasion type.
        
        Args:
            persuasion_type: The current persuasion strategy (pathos, logos, ethos)
            
        Returns:
            Formatted string describing the persuasion strategy
        """
        persuasion_descriptions = {
            "pathos": {
                "description": "emotional appeals and values",
                "strengths": "connecting emotionally with the audience and appealing to their values",
                "techniques": "emotional storytelling, vivid imagery, and value-based appeals"
            },
            "logos": {
                "description": "logical reasoning and factual evidence",
                "strengths": "building rational arguments supported by data and evidence",
                "techniques": "statistics, expert opinions, and causal reasoning"
            },
            "ethos": {
                "description": "ethical principles and credibility",
                "strengths": "establishing authority and appealing to ethical considerations",
                "techniques": "appeals to authority, ethical principles, and character references"
            }
        }
        
        if persuasion_type not in persuasion_descriptions:
            return ""
            
        info = persuasion_descriptions[persuasion_type]
        persuasion_info = (
            f"Using {persuasion_type} persuasion strategy ({info['description']}). "
            f"This approach excels at {info['strengths']} through {info['techniques']}."
        )
        logger.info(f"Including persuasion info in response: {persuasion_info}")
        return persuasion_info

    def generator(self, state: MessageState) -> MessageState:
        """Generate responses using the best persuasion strategy."""
        # get the input message
        user_message = state['user_message']
        orchestrator_message = state['orchestrator_message']
        message_flow = state.get('response', "") + "\n" + state.get("message_flow", "")
        
        # Get current persuasion type, prioritizing best_persuasion_type over current_persuasion_type
        best_persuasion_type = state.get("best_persuasion_type")
        current_persuasion_type = state.get("current_persuasion_type")
        persuasion_scores = state.get("persuasion_scores", {})
        
        # Log persuasion scores if available
        if persuasion_scores:
            logger.info("Current persuasion effectiveness scores:")
            for p_type, score in persuasion_scores.items():
                logger.info(f"  - {p_type}: {score:.2f}")
        
        # Determine which persuasion type to use in the response
        persuasion_type = best_persuasion_type or current_persuasion_type or ""
        if persuasion_type:
            logger.info(f"Using {persuasion_type} persuasion strategy for response generation")
            persuasion_info = self._get_persuasion_info(persuasion_type)
        else:
            logger.info("No persuasion type found in state, proceeding without strategy context")
            persuasion_info = ""

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
            # Add persuasion info to the context
            context = message_flow
            if persuasion_info:
                context += f"\n\n{persuasion_info}"
            input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history, "context": context})
        else:
            # If no message flow but we have persuasion info, use the context version of the prompt
            if persuasion_info:
                prompt = PromptTemplate.from_template(prompts["message_flow_generator_prompt"])
                input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history, "context": persuasion_info})
            else:
                prompt = PromptTemplate.from_template(prompts["message_generator_prompt"])
                input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history})
                
        logger.info(f"Prompt: {input_prompt.text}")
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        answer = final_chain.invoke(chunked_prompt)

        state["message_flow"] = ""
        state["response"] = answer
        return state
    
    def stream_generator(self, state: MessageState) -> MessageState:
        """Stream responses using the best persuasion strategy."""
        # Use the same enhanced persuasion info logic in the streaming version
        user_message = state['user_message']
        orchestrator_message = state['orchestrator_message']
        message_flow = state.get('response', "") + "\n" + state.get("message_flow", "")
        
        # Get current persuasion type, prioritizing best_persuasion_type over current_persuasion_type
        best_persuasion_type = state.get("best_persuasion_type")
        current_persuasion_type = state.get("current_persuasion_type")
        
        # Determine which persuasion type to use
        persuasion_type = best_persuasion_type or current_persuasion_type or ""
        if persuasion_type:
            logger.info(f"Using {persuasion_type} persuasion strategy for streaming response")
            persuasion_info = self._get_persuasion_info(persuasion_type)
        else:
            persuasion_info = ""

        # Process the rest using the streaming implementation
        return super().stream_generator(state) 
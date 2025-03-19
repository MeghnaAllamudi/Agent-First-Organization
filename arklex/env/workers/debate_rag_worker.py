import logging
from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.env.workers.faiss_rag_worker import FaissRAGWorker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL


logger = logging.getLogger(__name__)


class DebateRAGWorker(FaissRAGWorker):
    """A RAG worker specifically designed for debate topics and arguments."""
    
    description = "Retrieves and generates debate topics and arguments from a curated database of debate topics and structured arguments."

    def __init__(self, stream_response: bool = True):
        super().__init__(stream_response=stream_response)
        self.debate_prompt = PromptTemplate.from_template(
            """Based on the following debate topic and context, generate a structured argument.
            Consider both sides of the debate and provide evidence-based reasoning.
            
            Topic: {topic}
            Context: {context}
            
            Generate a response that includes:
            1. Main argument
            2. Supporting evidence
            3. Counterarguments
            4. Rebuttals
            
            Response:"""
        )

    def _create_action_graph(self):
        """Creates a modified action graph for debate-specific retrieval and generation."""
        workflow = super()._create_action_graph()
        
        # Add debate-specific nodes
        workflow.add_node("debate_formatter", self._format_debate_response)
        
        # Modify edges to include debate formatting
        workflow.add_conditional_edges(
            "tool_generator", 
            lambda x: "debate_formatter" if x.get("is_debate") else "end"
        )
        workflow.add_conditional_edges(
            "stream_tool_generator", 
            lambda x: "debate_formatter" if x.get("is_debate") else "end"
        )
        
        return workflow

    def _format_debate_response(self, state: MessageState) -> MessageState:
        """Formats the retrieved content into a structured debate response."""
        # Get the topic from the user's message
        topic = state.get("user_message", {}).get("topic", "")
        
        # Format the response using the debate prompt
        formatted_response = self.debate_prompt.format(
            topic=topic,
            context=state.get("message_flow", "")
        )
        
        # Generate the final response
        chain = self.llm | StrOutputParser()
        response = chain.invoke(formatted_response)
        
        # Update the state with the formatted response
        state["message_flow"] = response
        return state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the debate RAG workflow."""
        # Add debate flag to state
        msg_state["is_debate"] = True
        
        # Execute the parent workflow
        return super().execute(msg_state) 
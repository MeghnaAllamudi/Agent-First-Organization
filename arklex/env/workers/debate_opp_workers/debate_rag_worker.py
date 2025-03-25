from typing import Dict, Any, Optional
import logging
import random

from langgraph.graph import StateGraph, START
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.loaders.debate_loader import DebateLoader

logger = logging.getLogger(__name__)

@register_worker
class DebateRAGWorker(BaseWorker):
    """A worker that uses RAG to enhance debate responses with relevant information."""
    
    description = "Uses RAG to pick a debate topic and enhance debate responses with relevant information from a knowledge base. This is the first worker to run and it must use the ArgumentClassifier after."
    
    def __init__(self, tools: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.stream_response = MODEL.get("stream_response", True)
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize debate loader
        self.loader = DebateLoader()
        
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
        
        logger.info("DebateRAGWorker initialized successfully")
        
        self.action_graph = self._create_action_graph()
        
    def _pick_debate_topic(self, msg_state: MessageState) -> MessageState:
        """Executes the debate workflow."""
        logger.info("Starting debate processing")
        try:
            # Add debate flag to state
            msg_state["is_debate"] = True
            
            # Get URLs from the base topic page
            base_url = "https://www.kialo-edu.com/debate-topics-and-argumentative-essay-topics"
            urls = self.loader.get_all_urls(base_url, max_num=10)
            
            if not urls:
                raise Exception("No topics found")
                
            # Select one random topic
            topic_url = random.choice(urls)
            logger.info(f"Selected random topic URL: {topic_url}")
            
            # Create a CrawledURLObject for the topic
            from arklex.utils.loader import CrawledURLObject
            topic_obj = CrawledURLObject(
                id="topic_1",
                url=topic_url,
                content=None,
                metadata={"type": "topic"}
            )
            
            # Crawl the topic page
            topic_docs = self.loader.crawl_urls([topic_obj])
            if not topic_docs or topic_docs[0].is_error:
                raise Exception(f"Error crawling topic: {topic_docs[0].error_message if topic_docs else 'No documents'}")
                
            topic_doc = topic_docs[0]
            topic_content = topic_doc.content.split('\n')
            
            # Get topic name and arguments
            topic_name = topic_content[0].replace('Topic: ', '')
            arguments = []
            
            for line in topic_content[1:]:
                if line.startswith('PRO:') or line.startswith('CON:'):
                    arg_type = 'ethical' if 'ethical' in line.lower() else 'logical'
                    arg_content = line.split(':', 1)[1].strip()
                    arguments.append(arg_content)
            
            # Format the response using the debate prompt
            formatted_response = self.debate_prompt.format(
                topic=topic_name,
                context="\n".join(arguments)
            )
            
            # Generate the final response
            chain = self.llm | StrOutputParser()
            response = chain.invoke(formatted_response)
            
            # Update the state with the formatted response
            msg_state["message_flow"] = response
            logger.info("Debate processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in debate processing: {str(e)}", exc_info=True)
            return self._handle_error(msg_state)
            
        return msg_state

    def _handle_error(self, state: MessageState) -> MessageState:
        """Handles errors in debate processing."""
        logger.error("Handling error state")
        if hasattr(self, 'tools') and self.tools and "error_tool" in self.tools:
            error_response = self.tools["error_tool"].create_error_response("rag")
            state["message_flow"] = error_response.get("message", "An error occurred while processing the response.")
        else:
            state["message_flow"] = "I'm sorry, but I couldn't find a suitable debate topic at the moment. Could you suggest a topic you'd like to discuss?"
        return state 

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
    
# Register the effectiveness evaluator worker
register_worker(DebateRAGWorker) 
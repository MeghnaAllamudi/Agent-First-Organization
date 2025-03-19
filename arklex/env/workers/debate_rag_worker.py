import logging
from typing import Any, Dict, List
import random

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.debate_loader import DebateLoader


logger = logging.getLogger(__name__)


@register_worker
class DebateRAGWorker(BaseWorker):
    """A worker specifically designed for debate topics and arguments."""
    
    description = "Retrieves and generates debate topics and arguments from Kialo."

    def __init__(self):
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

    def _create_action_graph(self):
        """Creates an action graph for debate-specific retrieval and generation."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add nodes
        workflow.add_node("debate_formatter", self._format_debate_response)
        
        # Add edges
        workflow.add_edge(START, "debate_formatter")
        
        return workflow

    def _format_debate_response(self, state: MessageState) -> MessageState:
        """Formats the retrieved content into a structured debate response."""
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
        state["message_flow"] = response
        return state

    def execute(self, msg_state: MessageState) -> MessageState:
        """Executes the debate workflow."""
        # Add debate flag to state
        msg_state["is_debate"] = True
        
        # Execute the workflow
        workflow = self._create_action_graph()
        graph = workflow.compile()
        return graph.invoke(msg_state) 
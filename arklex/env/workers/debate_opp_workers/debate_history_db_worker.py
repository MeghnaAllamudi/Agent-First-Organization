import logging
import os

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.env.tools.utils import ToolGenerator
from arklex.env.tools.database.utils import DebateDatabaseActions
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.global_state import get_global_state, get_conversation_id_from_state


logger = logging.getLogger(__name__)


@register_worker
class DebateHistoryDatabaseWorker(BaseWorker):

    description = "At the end of every user response, log the bot's argument, the user's counter argument, the bot's persuasive strategy, the user's persuasive strategy, and the current timestamp to the debate history database for conversation tracking. This happens after every user response. It should only run once per user repsonse"

    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000, temperature = 0.0)
        self.actions = {
            "InsertConversationUpdate": "Insert last argument", 
        }
        self.DBActions = DebateDatabaseActions()
        
        self.action_graph = self._create_action_graph()

    def insert_latest_debate_argument(self, state: MessageState):
        # Insert the debate argument into database
        print("DEBATE HISTORY DATABASE - INSERTING ")
        print("============================================")
        return self.DBActions.insert_latest_debate_argument(state)
        
    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        workflow.add_node("InsertConversationUpdate", self.insert_latest_debate_argument)
        workflow.add_edge(START, "InsertConversationUpdate")
        return workflow

    def execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result

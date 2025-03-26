import os
import sqlite3
from datetime import datetime
import uuid
import logging
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from arklex.utils.utils import chunk_string
from arklex.utils.model_config import MODEL
from arklex.utils.graph_state import Slot, SlotDetail, Slots, MessageState
from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import StatusEnum


DBNAME = 'debate_history_db.sqlite'
USER_ID = "user_be6e1836-8fe9-4938-b2d0-48f810648e72"

logger = logging.getLogger(__name__)


SLOTS = [
    {
        "name": "show_name",
        "type": "string",
        "value": "",
        "description": "Name of the show",
        "prompt": "Please provide the name of the show"
    },
    {
        "name": "location",
        "type": "string",
        "value": "",
        "description": "Location of the show",
        "prompt": "Please provide the location of the show"
    },
    {
        "name": "date",
        "type": "date",
        "value": "",
        "description": "Date of the show",
        "prompt": "Please provide the date of the show"
    },
    {
        "name": "time",
        "type": "time",
        "value": "",
        "description": "Time of the show",
        "prompt": "Please provide the time of the show"
    }
]

NO_SHOW_MESSAGE = "Show is not found. Please check whether the information is correct."
MULTIPLE_SHOWS_MESSAGE = "There are multiple shows found. Please provide more details."
NO_BOOKING_MESSAGE = "You have not booked any show."


class DatabaseActions:
    def __init__(self, user_id: str=USER_ID):
        # Get the default data directory if DATA_DIR is not set
        default_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "tools", "database", "debate_db")
        data_dir = os.environ.get("DATA_DIR", default_data_dir)
        self.db_path = os.path.join(data_dir, DBNAME)
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
        self.user_id = user_id

    def log_in(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM user WHERE id = ?", (self.user_id,))
        result = cursor.fetchone()
        if result is None:
            logger.info(f"User {self.user_id} not found in the database.")
        else:
            logger.info(f"User {self.user_id} successfully logged in.")
        return result is not None

    def init_slots(self, slots: list[Slot], bot_config):
        if not slots:
            slots = SLOTS
        self.slots = []
        self.slot_prompts = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for slot in slots:
            query = f"SELECT DISTINCT {slot['name']} FROM show"
            cursor.execute(query)
            results = cursor.fetchall()
            value_list = [result[0] for result in results]
            self.slots.append(self.verify_slot(slot, value_list, bot_config))
            if not self.slots[-1].confirmed:
                self.slot_prompts.append(slot["prompt"])
        cursor.close()
        conn.close()
        return SLOTS

    def verify_slot(self, slot: Slot, value_list: list, bot_config) -> Slot:
        slot_detail = SlotDetail(**slot, verified_value="", confirmed=False)
        prompts = load_prompts(bot_config)
        prompt = PromptTemplate.from_template(prompts["database_slot_prompt"])
        input_prompt = prompt.invoke({
            "slot": {"name": slot["name"], "description": slot["description"], "slot": slot["type"]}, 
            "value": slot["value"], 
            "value_list": value_list
        })
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        logger.info(f"Chunked prompt for verifying slot: {chunked_prompt}")
        final_chain = self.llm | StrOutputParser()
        try:
            answer = final_chain.invoke(chunked_prompt)
            logger.info(f"Result for verifying slot value: {answer}")
            for value in value_list:
                if value in answer:
                    logger.info(f"Chosen slot value in the database worker: {value}")
                    slot_detail.verified_value = value
                    slot_detail.confirmed = True
                    return slot_detail
        except Exception as e:
            logger.error(f"Error occurred while verifying slot in the database worker: {e}")
        return slot_detail

    def search_show(self, msg_state: MessageState) -> MessageState:
        # Populate the slots with verified values
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT show_name, date, time, description, location, price FROM show WHERE 1 = 1"
        params = []
        for slot in self.slots:
            if slot.confirmed:
                query += f" AND {slot.name} = ?"
                params.append(slot.verified_value)
        query += " LIMIT 10"
        # Execute the query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if len(rows) == 0:
            msg_state["status"] = StatusEnum.INCOMPLETE.value
            msg_state["message_flow"] = NO_SHOW_MESSAGE
        else:
            column_names = [column[0] for column in cursor.description]
            results = [dict(zip(column_names, row)) for row in rows]
            results_df = pd.DataFrame(results)
            msg_state["status"] = StatusEnum.COMPLETE.value
            msg_state["message_flow"] = "Available shows are:\n" + results_df.to_string(index=False)
        return msg_state

    def book_show(self, msg_state: MessageState) -> MessageState:
        logger.info("Enter book show function")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT id, show_name, date, time, description, location, price FROM show WHERE 1 = 1"
        params = []
        for slot in self.slots:
            if slot.confirmed:
                query += f" AND {slot.name} = ?"
                params.append(slot.verified_value)
        # Execute the query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        logger.info(f"Rows found: {len(rows)}")
        # Check whether info is enough to book a show
        if len(rows) == 0:
            msg_state["status"] = StatusEnum.INCOMPLETE.value
            msg_state["message_flow"] = NO_SHOW_MESSAGE
        elif len(rows) > 1:
            msg_state["status"] = StatusEnum.INCOMPLETE.value
            if self.slot_prompts:
                msg_state["message_flow"] = self.slot_prompts[0]
            else:
                msg_state["message_flow"] = MULTIPLE_SHOWS_MESSAGE
        else:
            column_names = [column[0] for column in cursor.description]
            results = dict(zip(column_names, rows[0]))
            show_id = results["id"]

            # Insert a row into the booking table
            cursor.execute('''
                INSERT INTO booking (id, show_id, user_id, created_at)
                VALUES (?, ?, ?, ?)
            ''', ("booking_" + str(uuid.uuid4()),  show_id, self.user_id, datetime.now()))

            results_df = pd.DataFrame([results])
            msg_state["status"] = StatusEnum.COMPLETE.value
            msg_state["message_flow"] = "The booked show is:\n" + results_df.to_string(index=False)
        cursor.close()
        conn.close()
        return msg_state

    def check_booking(self, msg_state: MessageState) -> MessageState:
        logger.info("Enter check booking function")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
        SELECT * FROM
            booking b
            JOIN show s ON b.show_id = s.id
        WHERE
            b.user_id = ?
        """
        cursor.execute(query, (self.user_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if len(rows) == 0:
            msg_state["message_flow"] = NO_BOOKING_MESSAGE
        else:
            column_names = [column[0] for column in cursor.description]
            results = [dict(zip(column_names, row)) for row in rows]
            results_df = pd.DataFrame(results)
            msg_state["message_flow"] = "Booked shows are:\n" + results_df.to_string(index=False)
        msg_state["status"] = StatusEnum.COMPLETE.value
        return msg_state

    def cancel_booking(self, msg_state: MessageState) -> MessageState:
        logger.info("Enter cancel booking function")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
        SELECT * FROM
            booking b
            JOIN show s ON b.show_id = s.id
        WHERE
            b.user_id = ?
        """
        cursor.execute(query, (self.user_id,))
        rows = cursor.fetchall()
        if len(rows) == 0:
            msg_state["status"] = StatusEnum.COMPLETE.value
            msg_state["message_flow"] = NO_BOOKING_MESSAGE
        elif len(rows) > 1:
            msg_state["status"] = StatusEnum.INCOMPLETE.value
            if self.slot_prompts:
                msg_state["message_flow"] = self.slot_prompts[0]
            else:
                msg_state["message_flow"] = MULTIPLE_SHOWS_MESSAGE
        else:
            column_names = [column[0] for column in cursor.description]
            results = [dict(zip(column_names, row)) for row in rows]
            show = results[0]
            # Delete a row from the booking table based on show_id
            cursor.execute('''DELETE FROM booking WHERE show_id = ?
            ''', (show["id"],))
            # Respond to user the cancellation
            results_df = pd.DataFrame(results)
            msg_state["message_flow"] = "The cancelled show is:\n" + results_df.to_string(index=False)
            msg_state["status"] = StatusEnum.COMPLETE.value
        cursor.close()
        conn.close()
        return msg_state
    
    def insert_latest_debate_argument(self, msg_state: MessageState) -> MessageState:
        """
        Inserts the latest debate arguments and strategies into the debate_history table
        
        Args:
            msg_state (MessageState): The current message state containing the debate information
            
        Returns:
            MessageState: The updated message state
        """
        logger.info("Inserting latest debate argument into database")
        
        # Extract debate information from message state
        user_argument = msg_state.get("user_message", {}).get("message", "")
        # Try multiple fields where the bot's message might be stored
        bot_argument = msg_state.get("new_counter_argument", "")
        if not bot_argument:
            bot_argument = msg_state.get("response", "")
        if not bot_argument:
            bot_argument = msg_state.get("bot_message", "")
        
        user_strategy = msg_state.get("user_classification", "")
        bot_strategy = msg_state.get("persuasion_strategy_used", msg_state.get("bot_classification", ""))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert a row into the debate_history table
        cursor.execute('''
            INSERT INTO debate_history (id, user_id, user_argument, bot_argument, user_strategy, bot_strategy, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            "debate_" + str(uuid.uuid4()),
            self.user_id,
            user_argument,
            bot_argument,
            user_strategy,
            bot_strategy,
            timestamp
        ))
        
        # Commit changes and close the connection
        conn.commit()
        cursor.close()
        conn.close()
        
        # No need to modify the message state, just return it
        return msg_state
    
    def get_debate_history(self, msg_state: MessageState, limit: int = 10) -> MessageState:
        """
        Retrieves the debate history from the database
        
        Args:
            msg_state (MessageState): The current message state
            limit (int): Maximum number of records to retrieve
            
        Returns:
            MessageState: Updated message state with debate history
        """
        logger.info(f"Retrieving debate history for user {self.user_id}, limit: {limit}")
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if the debate_history table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='debate_history'")
        if not cursor.fetchone():
            logger.warning("Debate history table does not exist")
            msg_state["message_flow"] = "No debate history available."
            msg_state["status"] = StatusEnum.COMPLETE.value
            conn.close()
            return msg_state
        
        # Query the debate history
        query = """
        SELECT id, user_argument, bot_argument, user_strategy, bot_strategy, timestamp
        FROM debate_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        cursor.execute(query, (self.user_id, limit))
        rows = cursor.fetchall()
        
        if not rows:
            msg_state["message_flow"] = "You don't have any debate history yet."
            msg_state["status"] = StatusEnum.COMPLETE.value
        else:
            column_names = [column[0] for column in cursor.description]
            results = [dict(zip(column_names, row)) for row in rows]
            results_df = pd.DataFrame(results)
            
            # Format timestamps for better readability
            if 'timestamp' in results_df.columns:
                results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
                results_df['timestamp'] = results_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            msg_state["debate_history"] = results
            msg_state["message_flow"] = f"Your recent debate history:\n{results_df.to_string(index=False)}"
            msg_state["status"] = StatusEnum.COMPLETE.value
        
        cursor.close()
        conn.close()
        return msg_state
    

class DebateDatabaseActions:
    def __init__(self, user_id: str=USER_ID):
        # Get the default data directory if DATA_DIR is not set
        default_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "tools", "database", "debate_db")
        data_dir = os.environ.get("DATA_DIR", default_data_dir)
        self.db_path = os.path.join(data_dir, DBNAME)
        self.user_id = user_id

    def log_in(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM user WHERE id = ?", (self.user_id,))
        result = cursor.fetchone()
        if result is None:
            logger.info(f"User {self.user_id} not found in the database.")
        else:
            logger.info(f"User {self.user_id} successfully logged in.")
        return result is not None

    def insert_latest_debate_argument(self, msg_state: MessageState) -> MessageState:
        """
        Inserts the latest debate arguments and strategies into the debate_history table
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract debate information from message state
        user_argument = msg_state["user_message"].message
        # Try multiple fields where the bot's message might be stored
        bot_argument = msg_state["slots"]["bot_message"][0].value
        
        user_strategy = msg_state["slots"]["user_classification"][0].value
        bot_strategy = msg_state["slots"]["bot_classification"][0].value
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            cursor.execute("""
                INSERT INTO debate_history 
                (user_id, user_argument, bot_argument, user_strategy, bot_strategy, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.user_id, user_argument, bot_argument, user_strategy, bot_strategy, timestamp))
            conn.commit()
            logger.info("Successfully inserted debate argument into database")
        except Exception as e:
            logger.error(f"Error inserting debate argument: {e}")
        finally:
            cursor.close()
            conn.close()

        return msg_state
    

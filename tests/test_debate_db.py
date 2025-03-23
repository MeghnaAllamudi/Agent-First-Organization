import unittest
import os
import sqlite3
from datetime import datetime
import tempfile
import json
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

from arklex.env.tools.debate_db import (
    read_debate_history,
    store_debate_record,
    update_debate_record,
    get_persuasion_stats,
    SLOTS
)
from arklex.env.tools.debate_db.connection import DebateDBConnection
from arklex.utils.graph_state import MessageState, StatusEnum, Slot
from arklex.orchestrator.NLU.nlu import SlotFilling

class MockSlotFilling:
    def execute(self, slots, chat_history_str, metadata):
        # Just return the slots as is since they're already filled in our tests
        return slots

    def verify_needed(self, slot, chat_history_str, metadata):
        # No verification needed in tests
        return False, ""

class TestDebateDB(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MessageState for testing
        self.mock_state = MessageState(
            message_flow="",
            status=StatusEnum.INCOMPLETE.value,
            slots={},
            trajectory=[],
            metadata={},
            response=None
        )

        # Create mock slot filling API
        self.mock_slot_filling = MockSlotFilling()

        # Create test timestamp for consistent testing
        self.timestamp = datetime.now().isoformat()
        
        # Create sample test data
        self.test_record = (1, "logos", 0.85, self.timestamp, "Use more data")
        self.stats_record = ("logos", 0.85, 1, 0.85, 0.85)
        
        # Set up the connection patchers
        self.db_patcher = patch('arklex.env.tools.debate_db.connection.DebateDBConnection._init_db')
        self.mock_init_db = self.db_patcher.start()

    def tearDown(self):
        """Clean up test environment."""
        self.db_patcher.stop()

    @patch('arklex.env.tools.debate_db.connection.DebateDBConnection.get_connection')
    def test_store_debate_record(self, mock_get_conn):
        """Test storing a new debate record."""
        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        
        # Set up mock cursor behavior for SELECT after INSERT
        mock_cursor.lastrowid = 1
        mock_cursor.fetchone.return_value = (1, "logos", 0.95, self.timestamp, "Add statistical evidence")
        
        # Set up connection to return our cursor
        mock_conn.cursor.return_value = mock_cursor
        
        # Set up connection context manager
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        # Create tool instance and set up slots
        tool = store_debate_record()
        tool.slotfillapi = self.mock_slot_filling
        
        # Set up slots
        tool.slots = [
            Slot(
                name=SLOTS['persuasion_technique']['name'],
                type=SLOTS['persuasion_technique']['type'],
                description=SLOTS['persuasion_technique']['description'],
                prompt=SLOTS['persuasion_technique']['prompt'],
                required=SLOTS['persuasion_technique']['required'],
                value="logos",
                verified=True,
                enum=[]
            ),
            Slot(
                name=SLOTS['effectiveness_score']['name'],
                type=SLOTS['effectiveness_score']['type'],
                description=SLOTS['effectiveness_score']['description'],
                prompt=SLOTS['effectiveness_score']['prompt'],
                required=SLOTS['effectiveness_score']['required'],
                value=0.95,
                verified=True,
                enum=[]
            ),
            Slot(
                name=SLOTS['suggestion']['name'],
                type=SLOTS['suggestion']['type'],
                description=SLOTS['suggestion']['description'],
                prompt=SLOTS['suggestion']['prompt'],
                required=SLOTS['suggestion']['required'],
                value="Add statistical evidence",
                verified=True,
                enum=[]
            )
        ]
        
        # Execute tool
        state = tool.execute(self.mock_state)
        
        # Verify results
        self.assertEqual(state["status"], StatusEnum.COMPLETE.value)
        self.assertEqual(len(state["trajectory"]), 2)  # Tool call and result
        self.assertTrue(state["trajectory"][-1]["name"].endswith("store_debate_record"))
        result = state["trajectory"][-1]["content"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["new_record"]["persuasion_technique"], "logos")
        self.assertEqual(result["new_record"]["effectiveness_score"], 0.95)
        
        # Verify that cursor was called properly
        mock_cursor.execute.assert_any_call(unittest.mock.ANY, unittest.mock.ANY)  # INSERT call
        mock_cursor.execute.assert_any_call(unittest.mock.ANY, (1,))  # SELECT call
        mock_conn.commit.assert_called_once()

    @patch('arklex.env.tools.debate_db.connection.DebateDBConnection.get_connection')
    def test_read_debate_history(self, mock_get_conn):
        """Test reading debate history records."""
        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        
        # Set up mock cursor to return a list of records
        mock_cursor.fetchall.return_value = [(1, "logos", 0.85, self.timestamp, "Use more data")]
        
        # Set up connection to return our cursor
        mock_conn.cursor.return_value = mock_cursor
        
        # Set up connection context manager
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        # Create tool instance
        tool = read_debate_history()
        tool.slotfillapi = self.mock_slot_filling
        
        # Set up slots
        tool.slots = [
            Slot(
                name=SLOTS['limit']['name'],
                type=SLOTS['limit']['type'],
                description=SLOTS['limit']['description'],
                prompt=SLOTS['limit']['prompt'],
                required=SLOTS['limit']['required'],
                value=100,
                verified=True,
                enum=[]
            ),
            Slot(
                name=SLOTS['persuasion_type']['name'],
                type=SLOTS['persuasion_type']['type'],
                description=SLOTS['persuasion_type']['description'],
                prompt=SLOTS['persuasion_type']['prompt'],
                required=SLOTS['persuasion_type']['required'],
                value=None,
                verified=True,
                enum=[]
            )
        ]
        
        # Execute tool
        state = tool.execute(self.mock_state)
        
        # Verify results
        self.assertEqual(state["status"], StatusEnum.COMPLETE.value)
        self.assertEqual(len(state["trajectory"]), 2)  # Tool call and result
        self.assertTrue(state["trajectory"][-1]["name"].endswith("read_debate_history"))
        result = state["trajectory"][-1]["content"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["persuasion_technique"], "logos")
        
        # Verify that cursor was called with the correct query
        mock_cursor.execute.assert_called_once()

    @patch('arklex.env.tools.debate_db.connection.DebateDBConnection.get_connection')
    def test_update_debate_record(self, mock_get_conn):
        """Test updating an existing debate record."""
        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        
        # Set up mock cursor responses
        # First for the existence check
        mock_cursor.fetchone.side_effect = [
            (1,),  # Record exists check
            (1, "logos", 0.88, self.timestamp, "Updated suggestion")  # Final record fetch
        ]
        
        # Set up connection to return our cursor
        mock_conn.cursor.return_value = mock_cursor
        
        # Set up connection context manager
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        # Create tool instance
        tool = update_debate_record()
        tool.slotfillapi = self.mock_slot_filling
        
        # Set up slots
        tool.slots = [
            Slot(
                name=SLOTS['record_id']['name'],
                type=SLOTS['record_id']['type'],
                description=SLOTS['record_id']['description'],
                prompt=SLOTS['record_id']['prompt'],
                required=SLOTS['record_id']['required'],
                value=1,
                verified=True,
                enum=[]
            ),
            Slot(
                name=SLOTS['persuasion_technique']['name'],
                type=SLOTS['persuasion_technique']['type'],
                description=SLOTS['persuasion_technique']['description'],
                prompt=SLOTS['persuasion_technique']['prompt'],
                required=SLOTS['persuasion_technique']['required'],
                value="logos",
                verified=True,
                enum=[]
            ),
            Slot(
                name=SLOTS['effectiveness_score']['name'],
                type=SLOTS['effectiveness_score']['type'],
                description=SLOTS['effectiveness_score']['description'],
                prompt=SLOTS['effectiveness_score']['prompt'],
                required=SLOTS['effectiveness_score']['required'],
                value=0.88,
                verified=True,
                enum=[]
            ),
            Slot(
                name=SLOTS['suggestion']['name'],
                type=SLOTS['suggestion']['type'],
                description=SLOTS['suggestion']['description'],
                prompt=SLOTS['suggestion']['prompt'],
                required=SLOTS['suggestion']['required'],
                value="Updated suggestion",
                verified=True,
                enum=[]
            )
        ]
        
        # Execute tool
        state = tool.execute(self.mock_state)
        
        # Verify results
        self.assertEqual(state["status"], StatusEnum.COMPLETE.value)
        self.assertEqual(len(state["trajectory"]), 2)  # Tool call and result
        self.assertTrue(state["trajectory"][-1]["name"].endswith("update_debate_record"))
        result = state["trajectory"][-1]["content"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["updated_record"]["effectiveness_score"], 0.88)
        
        # Verify the cursor was called as expected
        self.assertEqual(mock_cursor.execute.call_count, 3)  # Once for check, once for update, once for fetch
        mock_conn.commit.assert_called_once()

    @patch('arklex.env.tools.debate_db.connection.DebateDBConnection.get_connection')
    def test_get_persuasion_stats(self, mock_get_conn):
        """Test getting persuasion technique statistics."""
        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        
        # Set up mock cursor to return stats when fetchall is called
        mock_cursor.fetchall.return_value = [("logos", 0.85, 1, 0.85, 0.85)]
        
        # Set up connection to return our cursor
        mock_conn.cursor.return_value = mock_cursor
        
        # Set up connection context manager
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        # Create tool instance
        tool = get_persuasion_stats()
        tool.slotfillapi = self.mock_slot_filling
        
        # Set up slots
        tool.slots = [
            Slot(
                name=SLOTS['technique']['name'],
                type=SLOTS['technique']['type'],
                description=SLOTS['technique']['description'],
                prompt=SLOTS['technique']['prompt'],
                required=SLOTS['technique']['required'],
                value=None,
                verified=True,
                enum=[]
            )
        ]
        
        # Execute tool
        state = tool.execute(self.mock_state)
        
        # Verify results
        self.assertEqual(state["status"], StatusEnum.COMPLETE.value)
        self.assertEqual(len(state["trajectory"]), 2)  # Tool call and result
        self.assertTrue(state["trajectory"][-1]["name"].endswith("get_persuasion_stats"))
        result = state["trajectory"][-1]["content"]
        self.assertIn("stats", result)
        self.assertIn("logos", result["stats"])
        self.assertEqual(result["stats"]["logos"]["average_score"], 0.85)
        self.assertEqual(result["stats"]["logos"]["usage_count"], 1)
        
        # Verify cursor was called as expected
        mock_cursor.execute.assert_called_once()
        mock_cursor.fetchall.assert_called_once()

if __name__ == "__main__":
    unittest.main() 
"""Base database connection management for debate database tools."""

import os
import sqlite3
import logging
from contextlib import contextmanager
from typing import Generator

from .schema import (
    CREATE_DEBATE_HISTORY_TABLE,
    CREATE_EFFECTIVENESS_SCORES_TABLE
)

logger = logging.getLogger(__name__)

class DebateDBConnection:
    """Base class for managing database connections.
    
    This class provides the core database connection functionality used by all debate database tools.
    It handles:
    - Connection initialization and cleanup
    - Table creation
    - Connection context management
    """
    
    def __init__(self):
        """Initialize database connection manager."""
        # Use the same database path as in the original implementation
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.db_path = os.path.join(logs_dir, "debate_history.db")
        self._init_db()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection using context manager.
        
        Usage:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Use cursor for database operations
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables using schema definitions
            cursor.execute(CREATE_DEBATE_HISTORY_TABLE)
            cursor.execute(CREATE_EFFECTIVENESS_SCORES_TABLE)
            
            conn.commit()
            logger.info("Database tables initialized successfully") 
import sqlite3
import argparse
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

DBNAME = 'debate_history_db.sqlite'  # Reusing the same database file


def build_debate_history_table(folder_path):
    """
    Adds the debate_history table to the existing database.
    This is separate from the main build_database script to avoid modifying it.
    
    Args:
        folder_path: Path to the data directory where the database is stored
    """
    db_path = Path(folder_path) / DBNAME
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}. Please run the main build_database.py script first.")
        return False
    
    # Connect to the existing database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the debate_history table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='debate_history'")
    if cursor.fetchone():
        logger.info("Debate history table already exists. Skipping creation.")
    else:
        # Create the debate_history table
        logger.info("Creating debate_history table...")
        cursor.execute('''
            CREATE TABLE debate_history (
                id VARCHAR(40) PRIMARY KEY,
                user_id VARCHAR(40),
                user_argument TEXT,
                bot_argument TEXT,
                user_strategy VARCHAR(100),
                bot_strategy VARCHAR(100),
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
            )
        ''')
        logger.info("Debate history table created successfully.")
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True, type=str, help="location of the existing database")
    args = parser.parse_args()

    if not os.path.exists(args.folder_path):
        logger.error(f"Folder path {args.folder_path} does not exist.")
        exit(1)

    success = build_debate_history_table(args.folder_path)
    if success:
        print(f"Debate history table has been added to the database at {args.folder_path}/{DBNAME}")
    else:
        print("Failed to create debate history table. See logs for details.") 
import os
import json
import argparse
import time
import logging
import subprocess
import signal
import atexit
from dotenv import load_dotenv
from pprint import pprint

import shopify

from arklex.utils.utils import init_logger
from arklex.orchestrator.orchestrator import AgentOrg
# from create import API_PORT
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS
from arklex.env.env import Env
from arklex.utils.database import DebateDatabase  # Import the database class
from arklex.utils.utils import format_chat_history  # Add import for format_chat_history

load_dotenv()
# session = shopify.Session(os.environ["SHOPIFY_SHOP_URL"], os.environ["SHOPIFY_API_VERSION"], os.environ["SHOPIFY_ACCESS_TOKEN"])
# shopify.ShopifyResource.activate_session(session)

process = None  # Global reference for the FastAPI subprocess

def pprint_with_color(data, color_code="\033[34m"):  # Default to blue
    print(color_code, end="")  # Set the color
    pprint(data)
    print("\033[0m", end="")  

def terminate_subprocess():
    """Terminate the FastAPI subprocess."""
    global process
    if process and process.poll() is None:  # Check if process is running
        logger.info(f"Terminating FastAPI process with PID: {process.pid}")
        process.terminate()  # Send SIGTERM
        process.wait()  # Ensure it stops
        logger.info("FastAPI process terminated.")

# Register cleanup function to run on program exit
atexit.register(terminate_subprocess)

# Handle signals (e.g., Ctrl+C)
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))
signal.signal(signal.SIGTERM, lambda signum, frame: exit(0))


def get_api_bot_response(args, history, user_text, parameters, env):
    data = {"text": user_text, 'chat_history': history, 'parameters': parameters}
    orchestrator = AgentOrg(config=os.path.join(args.input_dir, "taskgraph.json"), env=env)
    result = orchestrator.get_response(data)
    
    # After getting the main response, explicitly evaluate effectiveness and update database
    try:
        # Create message state for effectiveness evaluation
        from arklex.utils.graph_state import ConvoMessage, OrchestratorMessage, MessageState
        from arklex.utils.graph_state import BotConfig
        
        # Format chat history string
        chat_history_str = format_chat_history(history + [{"role": "user", "content": user_text}, {"role": "assistant", "content": result['answer']}])
        
        # Extract configuration from the orchestrator
        config = json.load(open(os.path.join(args.input_dir, "taskgraph.json")))
        bot_config = BotConfig(
            bot_id=config.get("bot_id", "default"),
            version=config.get("version", "default"),
            language=config.get("language", "EN"),
            bot_type=config.get("bot_type", "presalebot"),
            available_workers=config.get("workers", [])
        )
        
        # Set up message state for evaluation
        user_message = ConvoMessage(history=chat_history_str, message=user_text)
        bot_message = ConvoMessage(history=chat_history_str, message=result['answer'])
        orchestrator_message = OrchestratorMessage(
            message=result['answer'],
            attribute={"persuasion_type": parameters.get("current_persuasion_type", "logos")}
        )
        
        message_state = MessageState(
            sys_instruct="You are a debate opponent.", 
            bot_config=bot_config,
            user_message=user_message,
            bot_message=bot_message, 
            orchestrator_message=orchestrator_message,
            trajectory=history + [{"role": "user", "content": user_text}, {"role": "assistant", "content": result['answer']}],
            message_flow=parameters.get("worker_response", {}).get("message_flow", ""), 
            slots=parameters.get("dialog_states"),
            metadata=parameters.get("metadata")
        )
        
        # Find and initialize the EffectivenessEvaluator
        effectiveness_evaluator = None
        for worker_info in config.get("workers", []):
            if worker_info.get("name") == "EffectivenessEvaluator":
                from arklex.env.workers.effectiveness_evaluator import EffectivenessEvaluator
                effectiveness_evaluator = EffectivenessEvaluator(json.loads(worker_info.get("config", "{}")))
                break
        
        # Find and initialize the DebateDatabaseWorker
        database_worker = None
        for worker_info in config.get("workers", []):
            if worker_info.get("name") == "DebateDatabaseWorker":
                from arklex.env.workers.debate_database_worker import DebateDatabaseWorker
                database_worker = DebateDatabaseWorker(json.loads(worker_info.get("config", "{}")))
                break
        
        # Evaluate effectiveness if the evaluator is available
        if effectiveness_evaluator:
            print("\n" + "="*80)
            print("🔍 EXPLICITLY TRIGGERING EFFECTIVENESS EVALUATION AFTER USER RESPONSE")
            print("="*80)
            
            # Set current persuasion type if available
            if "current_persuasion_type" in parameters:
                message_state["current_persuasion_type"] = parameters["current_persuasion_type"]
            elif "just_used_persuasion_type" in parameters:
                message_state["current_persuasion_type"] = parameters["just_used_persuasion_type"]
            
            # Evaluate effectiveness
            eval_result = effectiveness_evaluator.execute(message_state)
            
            # Update parameters with evaluation results
            if "evaluated_effectiveness_score" in eval_result:
                parameters["evaluated_effectiveness_score"] = eval_result["evaluated_effectiveness_score"]
            if "effectiveness_evaluation" in eval_result:
                parameters["effectiveness_evaluation"] = eval_result["effectiveness_evaluation"]
            if "current_persuasion_type" in eval_result:
                parameters["current_persuasion_type"] = eval_result["current_persuasion_type"]
            
            # Update database with evaluation results if database worker is available
            if database_worker and "evaluated_effectiveness_score" in eval_result:
                print("\n" + "="*80)
                print("💾 EXPLICITLY TRIGGERING DATABASE UPDATE WITH EFFECTIVENESS SCORE")
                print("="*80)
                
                # Apply evaluated score to message state for database update
                message_state["evaluated_effectiveness_score"] = eval_result["evaluated_effectiveness_score"]
                
                # Create response with effectiveness score to update database
                persuasion_type = eval_result.get("current_persuasion_type", "logos")
                response_key = f"{persuasion_type}_persuasion_response"
                message_state[response_key] = {
                    "counter_argument": result['answer'],
                    "effectiveness_score": eval_result["evaluated_effectiveness_score"]
                }
                
                # Execute database update and get best type
                db_result = database_worker.execute(message_state)
                
                # Update parameters with database results
                if "best_persuasion_type" in db_result:
                    parameters["best_persuasion_type"] = db_result["best_persuasion_type"]
                if "persuasion_scores" in db_result:
                    parameters["persuasion_scores"] = db_result["persuasion_scores"]
    except Exception as e:
        logger.error(f"Error in explicit effectiveness evaluation: {str(e)}")
    
    return result['answer'], parameters


def start_apis():
    """Start the FastAPI subprocess and update task graph API URLs."""
    global process
    command = [
        "uvicorn",
        "arklex.orchestrator.NLU.api:app",  # Replace with proper import path
        "--port", API_PORT,
        "--host", "0.0.0.0",
        "--log-level", "info"
    ]

    # Redirect FastAPI logs to a file
    with open("./logs/api.log", "w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,  # Redirect stdout to a log file
            stderr=subprocess.STDOUT,  # Redirect stderr to the same file
            start_new_session=True  # Run in a separate process group
        )
    logger.info(f"Started FastAPI process with PID: {process.pid}")

def initialize_database():
    """Initialize and test the debate database."""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Set up the database path
        db_path = os.path.join(logs_dir, "debate_history.db")
        logger.info(f"Initializing debate database at {db_path}")
        
        # Initialize the database
        db = DebateDatabase(db_path=db_path)
        
        # Test database by getting and setting some values
        initial_scores = {}
        for p_type in ["pathos", "logos", "ethos"]:
            score = db.get_effectiveness_score(p_type)
            initial_scores[p_type] = score
            
        logger.info(f"Initial effectiveness scores: {initial_scores}")
        
        # Get all records
        records = db.get_all_records(limit=5)
        record_count = len(records)
        logger.info(f"Found {record_count} records in the database")
        
        if record_count > 0:
            logger.info(f"Latest records: {records[:3]}")
            
        logger.info("Database initialization successful")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="./examples/test")
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    parser.add_argument( '--llm-provider',type=str,default=MODEL["llm_provider"],choices=LLM_PROVIDERS)
    parser.add_argument('--log-level', type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    os.environ["DATA_DIR"] = args.input_dir
    MODEL["model_type_or_path"] = args.model
    MODEL["llm_provider"] = args.llm_provider
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logger = init_logger(log_level=log_level, filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"))

    # Initialize the database before starting the API
    db_initialized = initialize_database()
    if not db_initialized:
        logger.warning("Database initialization failed, some features may not work correctly")

    # Initialize NLU and Slotfill APIs
    # start_apis()

    # Initialize env
    config = json.load(open(os.path.join(args.input_dir, "taskgraph.json")))
    env = Env(
        tools = config.get("tools", []),
        workers = config.get("workers", []),
        slotsfillapi = config["slotfillapi"]
    )
        
    history = []
    params = {}
    user_prefix = "user"
    worker_prefix = "assistant"
    for node in config['nodes']:
        if node[1].get("type", "") == 'start':
            start_message = node[1]['attribute']["value"]
            break
    history.append({"role": worker_prefix, "content": start_message})
    pprint_with_color(f"Bot: {start_message}")
    try:
        while True:
            user_text = input("You: ")
            if user_text.lower() == "quit":
                break
            start_time = time.time()
            output, params = get_api_bot_response(args, history, user_text, params, env)
            history.append({"role": user_prefix, "content": user_text})
            history.append({"role": worker_prefix, "content": output})
            print(f"getAPIBotResponse Time: {time.time() - start_time}")
            pprint_with_color(f"Bot: {output}")
    finally:
        terminate_subprocess()  # Ensure the subprocess is terminated

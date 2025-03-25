import os
import json
import argparse
import time
import logging
import subprocess
import signal
import atexit
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from arklex.utils.utils import init_logger
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.orchestrator.generator.generator import Generator
from arklex.env.tools.RAG.build_rag import build_rag
from arklex.env.tools.database.build_database import build_database
from arklex.utils.model_config import MODEL
from arklex.utils.rate_limiter import RateLimiter
from arklex.utils.debate_loader import DebateLoader
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.env.workers.debate_rag_worker import DebateRAGWorker
from arklex.env.workers.message_worker import MessageWorker
from arklex.env.workers.default_worker import DefaultWorker
from arklex.env.workers.persuasion_worker import PersuasionWorker
from arklex.env.workers.argument_classifier import ArgumentClassifier
from arklex.env.workers.debate_database_worker import DebateDatabaseWorker
from examples.debate_opponent_tests.test_debate_workers import EffectivenessEvaluator

logger = init_logger(log_level=logging.INFO, filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"))
load_dotenv()

# Initialize rate limiter
rate_limiter = RateLimiter()

# API_PORT = "55135"
# NLUAPI_ADDR = f"http://localhost:{API_PORT}/nlu"
# SLOTFILLAPI_ADDR = f"http://localhost:{API_PORT}/slotfill"

def create_output_dir(output_dir: str):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def generate_taskgraph(args):
    model = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
    generator = Generator(args, args.config, model, args.output_dir)
    
    # Estimate tokens and wait if needed
    with open(args.config, 'r') as f:
        config_text = f.read()
    estimated_tokens = rate_limiter.estimate_tokens(config_text)
    rate_limiter.wait_if_needed(estimated_tokens)
    
    taskgraph_filepath = generator.generate()
    # Update the task graph with the API URLs
    task_graph = json.load(open(os.path.join(os.path.dirname(__file__), taskgraph_filepath)))
    task_graph["nluapi"] = ""
    task_graph["slotfillapi"] = ""
    with open(taskgraph_filepath, "w") as f:
        json.dump(task_graph, f, indent=4)


def init_worker(args):
    """Initialize workers and tools."""
    # Load config
    config = json.load(open(args.config))
    
    # Check if this is a debate-related config
    is_debate_config = "debate" in os.path.basename(args.config).lower()
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Create a dictionary of tools for easy access
    tools_dict = {tool["name"]: tool for tool in config["tools"]}
    
    # Initialize workers
    for worker_config in config["workers"]:
        worker_name = worker_config["name"]
        worker_id = worker_config["id"]
        
        # For debate configs, initialize based on worker name, not type
        if is_debate_config:
            # if worker_name == "DebateDatabaseWorker":
            #     worker = DebateDatabaseWorker()
            if worker_name == "PersuasionWorker":
                worker = PersuasionWorker()
            elif worker_name == "DebateRAGWorker":
                worker = DebateRAGWorker()
            elif worker_name == "DebateMessageWorker" or worker_name == "MessageWorker":
                worker = MessageWorker()
            elif worker_name == "DefaultWorker":
                worker = DefaultWorker()
            elif worker_name == "ArgumentClassifier":
                worker = ArgumentClassifier()
            elif worker_name == "EffectivenessEvaluator":
                worker = EffectivenessEvaluator()
            else:
                logger.warning(f"Unknown worker name for debate config: {worker_name}")
                continue
        else:
            # Original initialization code for non-debate configs
            worker_type = worker_config.get("type", "")
            
            # Parse the config if it's a string
            worker_config_data = {}
            if "config" in worker_config:
                if isinstance(worker_config["config"], str):
                    try:
                        worker_config_data = json.loads(worker_config["config"])
                    except:
                        worker_config_data = {}
                else:
                    worker_config_data = worker_config["config"]
            
            
        # Set worker name to match ID
        worker.name = worker_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./arklex/orchestrator/examples/customer_service_config.json")
    parser.add_argument('--output-dir', type=str, default="./examples/test")
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    parser.add_argument('--log-level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--task', type=str, choices=["gen_taskgraph", "init", "all"], default="all")
    args = parser.parse_args()
    MODEL["model_type_or_path"] = args.model
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = init_logger(log_level=log_level, filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.task == "all":
        generate_taskgraph(args)
        init_worker(args)
    elif args.task == "gen_taskgraph":
        generate_taskgraph(args)
    elif args.task == "init":
        init_worker(args)
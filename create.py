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
from arklex.env.workers.persuasion_worker import PathosWorker, LogosWorker, EthosWorker
from arklex.env.workers.argument_classifier import ArgumentClassifier
from arklex.env.workers.effectiveness_evaluator import EffectivenessEvaluator
from arklex.env.workers.debate_history_worker import DebateHistoryWorker
from arklex.env.tools.citation_tool import CitationTool, register_tool

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
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Initialize workers
    for worker_config in config["workers"]:
        worker_type = worker_config["type"]
        worker_id = worker_config["id"]
        worker_name = worker_config["name"]
        
        # Parse the config if it's a string
        if isinstance(worker_config["config"], str):
            worker_config["config"] = json.loads(worker_config["config"])
        
        if worker_type == "rag":
            worker = DebateRAGWorker()
        elif worker_type == "base":
            worker = MessageWorker()  # No config needed
        elif worker_type == "persuasion":
            # Use worker name to determine type
            if worker_name == "PathosWorker":
                worker = PathosWorker(worker_config["config"])
            elif worker_name == "LogosWorker":
                worker = LogosWorker(worker_config["config"])
            elif worker_name == "EthosWorker":
                worker = EthosWorker(worker_config["config"])
            else:
                logger.warning(f"Unknown persuasion worker: {worker_name}")
                continue
        elif worker_type == "analysis":
            worker = ArgumentClassifier(worker_config["config"])
        elif worker_type == "evaluation":
            worker = EffectivenessEvaluator(worker_config["config"])
        elif worker_type == "history":
            worker = DebateHistoryWorker()
        else:
            logger.warning(f"Unknown worker type: {worker_type}")
            continue
            
        # Set worker name to match ID
        worker.name = worker_id
    
    # Initialize tools
    for tool_config in config["tools"]:
        tool_type = tool_config["type"]
        tool_id = tool_config["id"]
        
        if tool_type == "reference":  # Changed from "citation" to "reference"
            tool = CitationTool()  # No config needed
        else:
            logger.warning(f"Unknown tool type: {tool_type}")
            continue
            
        # Register tool
        register_tool(tool_id, tool)


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
        init_worker(args)
        generate_taskgraph(args)
    elif args.task == "gen_taskgraph":
        generate_taskgraph(args)
    elif args.task == "init":
        init_worker(args)
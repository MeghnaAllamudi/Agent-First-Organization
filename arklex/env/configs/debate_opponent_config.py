from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.tools import ToolFactory


class DebateOpponentConfig:
    """Configuration for the debate opponent environment."""
    
    def __init__(self):
        # Initialize LLM
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize tools
        self.json_tool = ToolFactory.create_json_tool()
        self.error_tool = ToolFactory.create_error_tool()
        
        # Define the argument classification prompt
        self.classification_prompt = PromptTemplate.from_template("""You are an expert in analyzing arguments.
        Analyze the following argument and classify it based on the persuasion techniques used.
        
        Argument: {argument}
        
        Respond in the following JSON format:
        {
            "primary_technique": "main persuasion technique used",
            "secondary_techniques": ["list of other techniques used"],
            "effectiveness_score": float (0-1)  # Estimated effectiveness of the argument
        }
        
        Classification:""")
        
        # Define the debate history prompt
        self.history_prompt = PromptTemplate.from_template("""You are an expert in analyzing debate history.
        Based on the following debate history, provide insights about the effectiveness of different persuasion techniques.
        
        Debate History: {history}
        
        Respond in the following JSON format:
        {
            "technique_effectiveness": {
                "technique_name": float (0-1)  # Average effectiveness score
            },
            "overall_effectiveness": float (0-1)  # Average effectiveness across all arguments
        }
        
        Analysis:""")
        
        # Define the effectiveness evaluation prompt
        self.effectiveness_prompt = PromptTemplate.from_template("""You are an expert in evaluating argument effectiveness.
        Evaluate the effectiveness of the following counter-argument in responding to the original argument.
        
        Original Argument: {original_argument}
        Counter-Argument: {counter_argument}
        
        Respond in the following JSON format:
        {
            "effectiveness_score": float (0-1)  # Overall effectiveness score
            "strengths": ["list of argument strengths"],
            "weaknesses": ["list of argument weaknesses"]
        }
        
        Evaluation:""")
        
        # Define the persuasion techniques
        self.techniques = {
            "pathos": {
                "emotional_storytelling": {
                    "description": "Use personal stories or narratives to create emotional connection",
                    "examples": [
                        "Share a relatable experience",
                        "Describe emotional impact",
                        "Use vivid imagery"
                    ]
                },
                "value_appeals": {
                    "description": "Appeal to shared values and beliefs",
                    "examples": [
                        "Connect to core values",
                        "Highlight moral principles",
                        "Emphasize shared goals"
                    ]
                },
                "emotional_language": {
                    "description": "Use emotionally charged words and phrases",
                    "examples": [
                        "Choose impactful adjectives",
                        "Use emotional metaphors",
                        "Include feeling words"
                    ]
                },
                "empathy": {
                    "description": "Show understanding and connection with audience's perspective",
                    "examples": [
                        "Acknowledge feelings",
                        "Show shared concerns",
                        "Express understanding"
                    ]
                }
            },
            "logos": {
                "data_analysis": {
                    "description": "Use statistics, facts, and data to support arguments",
                    "examples": [
                        "Cite relevant statistics",
                        "Present research findings",
                        "Use numerical evidence"
                    ]
                },
                "logical_reasoning": {
                    "description": "Apply deductive and inductive reasoning",
                    "examples": [
                        "Use syllogisms",
                        "Apply cause-effect relationships",
                        "Make logical connections"
                    ]
                },
                "evidence_based": {
                    "description": "Support claims with concrete evidence",
                    "examples": [
                        "Reference studies",
                        "Cite expert opinions",
                        "Use case studies"
                    ]
                },
                "comparative_analysis": {
                    "description": "Compare and contrast different perspectives",
                    "examples": [
                        "Evaluate alternatives",
                        "Analyze pros and cons",
                        "Consider different scenarios"
                    ]
                }
            },
            "ethos": {
                "credibility_building": {
                    "description": "Establish trust and authority",
                    "examples": [
                        "Demonstrate expertise",
                        "Show experience",
                        "Reference credentials"
                    ]
                },
                "character_appeal": {
                    "description": "Appeal to moral character and values",
                    "examples": [
                        "Show integrity",
                        "Demonstrate honesty",
                        "Express ethical principles"
                    ]
                },
                "trustworthiness": {
                    "description": "Build trust through transparency and consistency",
                    "examples": [
                        "Acknowledge limitations",
                        "Show consistency",
                        "Be transparent"
                    ]
                },
                "authority_establishment": {
                    "description": "Establish authority through knowledge and experience",
                    "examples": [
                        "Reference expertise",
                        "Share relevant experience",
                        "Cite authoritative sources"
                    ]
                }
            }
        }
        
        # Initialize validation tool with all valid techniques
        self.validation_tool = ToolFactory.create_validation_tool(
            required_fields=["primary_technique", "secondary_techniques", "effectiveness_score"],
            valid_techniques=[technique for techniques in self.techniques.values() 
                            for technique in techniques.keys()]
        )
        
        # Initialize technique formatting tool
        self.technique_tool = ToolFactory.create_technique_tool()
        
        # Define the worker configuration
        self.worker_config = {
            "argument_classifier": {
                "class": "ArgumentClassifier",
                "config": {
                    "llm": self.llm,
                    "classification_prompt": self.classification_prompt,
                    "validation_tool": self.validation_tool,
                    "json_tool": self.json_tool,
                    "error_tool": self.error_tool
                }
            },
            "debate_history": {
                "class": "DebateHistoryWorker",
                "config": {
                    "llm": self.llm,
                    "history_prompt": self.history_prompt,
                    "json_tool": self.json_tool,
                    "error_tool": self.error_tool
                }
            },
            "effectiveness_evaluator": {
                "class": "EffectivenessEvaluator",
                "config": {
                    "llm": self.llm,
                    "effectiveness_prompt": self.effectiveness_prompt,
                    "json_tool": self.json_tool,
                    "error_tool": self.error_tool
                }
            },
            "pathos_worker": {
                "class": "PathosWorker",
                "config": {
                    "techniques": self.techniques["pathos"]
                }
            },
            "logos_worker": {
                "class": "LogosWorker",
                "config": {
                    "techniques": self.techniques["logos"]
                }
            },
            "ethos_worker": {
                "class": "EthosWorker",
                "config": {
                    "techniques": self.techniques["ethos"]
                }
            }
        }
        
        # Define the workflow configuration
        self.workflow_config = {
            "nodes": [
                "argument_classifier",
                "debate_history",
                "effectiveness_evaluator",
                "pathos_worker",
                "logos_worker",
                "ethos_worker"
            ],
            "edges": [
                ("argument_classifier", "debate_history"),
                ("debate_history", "effectiveness_evaluator"),
                ("effectiveness_evaluator", "pathos_worker"),
                ("effectiveness_evaluator", "logos_worker"),
                ("effectiveness_evaluator", "ethos_worker")
            ]
        }
        
        # Define the tools configuration
        self.tools_config = {
            "json_tool": {
                "uuid": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
                "name": "json_parsing_tool",
                "path": "arklex/utils/tools/json_parsing_tool.py"
            },
            "error_tool": {
                "uuid": "b2c3d4e5-f6g7-h8i9-j0k1-l2m3n4o5p6q7",
                "name": "error_handling_tool",
                "path": "arklex/utils/tools/error_handling_tool.py"
            },
            "validation_tool": {
                "uuid": "c3d4e5f6-g7h8-i9j0-k1l2-m3n4o5p6q7r8",
                "name": "argument_validation_tool",
                "path": "arklex/utils/tools/argument_validation_tool.py"
            },
            "technique_tool": {
                "uuid": "d4e5f6g7-h8i9-j0k1-l2m3-n4o5p6q7r8s9",
                "name": "technique_formatting_tool",
                "path": "arklex/utils/tools/technique_formatting_tool.py"
            }
        }
}
 
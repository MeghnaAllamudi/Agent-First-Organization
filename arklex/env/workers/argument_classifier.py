import logging
import json
import re
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState, ConvoMessage
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP

# Import argument_classifier_prompt from prompts_for_debate_opp
import importlib.util
import os
import sys

# Get the path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, root_dir)

# Import directly from the root directory
from prompts_for_debate_opp import argument_classifier_prompt

logger = logging.getLogger(__name__)


@register_worker
class ArgumentClassifier(BaseWorker):
    """A worker that classifies user arguments into different types."""
    
    description = "Classifies user arguments into emotional, logical, or ethical categories."
    
    # Class-level counter to track execution calls
    _execution_count = 0
    _last_classified_content = None
    # Track which requests have been processed (keyed by request ID)
    _processed_requests = set()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.stream_response = MODEL.get("stream_response", True)
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        # Initialize the action graph
        self.action_graph = self._create_action_graph()
        
        # Set default categories if none provided
        self.categories = {
            "emotional": {
                "description": "Arguments based on feelings, emotions, and personal experiences",
                "examples": [
                    "This makes me feel very concerned",
                    "I'm worried about the impact",
                    "This is deeply troubling"
                ]
            },
            "logical": {
                "description": "Arguments based on facts, data, and reasoning",
                "examples": [
                    "The data shows a clear trend",
                    "This follows from basic principles",
                    "The evidence supports this conclusion"
                ]
            },
            "ethical": {
                "description": "Arguments based on moral principles and values",
                "examples": [
                    "This raises important ethical concerns",
                    "We have a moral obligation to",
                    "This conflicts with our values"
                ]
            }
        }
        
        self.classification_prompt = PromptTemplate.from_template(
            """Classify the following user argument into one or more categories.
            
            Categories:
            {categories}
            
            User Argument: {user_argument}
            
            Respond in the following JSON format:
            {{
                "dominant_type": "primary category (emotional/logical/ethical)",
                "secondary_types": ["list of secondary categories"],
                "confidence": float (0-1),
                "reasoning": "explanation for classification"
            }}
            
            Classification:"""
        )

    def _validate_classification(self, classification: Dict[str, Any]) -> bool:
        """Validates the classification output format."""
        required_fields = ["dominant_type", "secondary_types", "confidence", "reasoning"]
        
        # Check required fields
        if not all(field in classification for field in required_fields):
            logger.error("Missing required fields in classification")
            return False
            
        # Validate confidence score
        if not 0 <= classification["confidence"] <= 1:
            logger.error("Invalid confidence score in classification")
            return False
            
        # Validate types
        valid_types = list(self.categories.keys())
        if classification["dominant_type"] not in valid_types:
            logger.error("Invalid dominant type in classification")
            return False
            
        if not all(t in valid_types for t in classification["secondary_types"]):
            logger.error("Invalid secondary types in classification")
            return False
            
        return True

    def _classify_argument(self, content):
        """Classify an argument based on its content.
        
        Args:
            content: The text content of the argument
            
        Returns:
            A dictionary containing the classification results
        """
        print(f"\n==== ARGUMENT CLASSIFIER ANALYSIS ====")
        print(f"Analyzing argument: {content[:50]}...")
        
        # Use our helper method to count persuasion indicators
        pathos_count, logos_count, ethos_count = self._count_persuasion_indicators(content)
        
        # Determine the dominant type based on counts
        counts = {
            'emotional': pathos_count,
            'logical': logos_count,
            'ethical': ethos_count
        }
        
        # Find the dominant type
        dominant_type = max(counts, key=counts.get)
        
        # Calculate confidence based on how much the dominant type stands out
        total_count = sum(counts.values())
        if total_count > 0:
            dominant_count = counts[dominant_type]
            confidence = dominant_count / total_count
        else:
            # Default confidence if no indicators found
            dominant_type = 'logical'  # Default to logical if no clear indicators
            confidence = 0.34
        
        # Find secondary types (sorted by count)
        secondary_types = sorted(
            [t for t in counts if t != dominant_type],
            key=lambda t: counts[t],
            reverse=True
        )
        
        # Generate reasoning using helper method
        reasoning = self._generate_classification_reasoning(dominant_type, counts)
        
        # Print analysis for debugging
        print("\n==== PERSUASION ANALYSIS ====")
        print(f"EMOTIONAL (pathos) indicators: {pathos_count}")
        print(f"LOGICAL (logos) indicators: {logos_count}")
        print(f"ETHICAL (ethos) indicators: {ethos_count}")
        print(f"DOMINANT TYPE: {dominant_type.upper()} (confidence: {confidence:.2f})")
        print(f"SECONDARY TYPES: {', '.join(secondary_types)}")
        print(f"REASONING: {reasoning}")
        print("=============================\n")
        
        # Return the classification result
        return {
            'dominant_type': dominant_type,
            'confidence': confidence,
            'secondary_types': secondary_types,
            'reasoning': reasoning,
            'counts': {
                'emotional': pathos_count,
                'logical': logos_count,
                'ethical': ethos_count
            }
        }

    def _create_action_graph(self):
        """Creates the action graph for argument classification."""
        from langgraph.graph import StateGraph, START
        
        workflow = StateGraph(MessageState)
        
        # Add classification node
        workflow.add_node("classifier", self._classify_arguments)
        
        # Add edges
        workflow.add_edge(START, "classifier")
        
        return workflow

    def _classify_arguments(self, state: MessageState) -> MessageState:
        """Classifies all arguments in the state."""
        # Get user message
        user_message = state.get("user_message")
        if not user_message:
            return state
            
        # Access content directly from ConvoMessage object
        user_argument = user_message.content if hasattr(user_message, 'content') else ""
        
        if not user_argument:
            return state
            
        # Classify the argument
        classification = self._classify_argument(user_argument)
        
        if classification:
            state["argument_classification"] = classification
        
        return state

    def execute(self, state: MessageState) -> MessageState:
        """Analyze the user's argument and classify it."""
        print("\n================================================================================")
        print(f"🔍 ARGUMENT CLASSIFIER EXECUTING")
        print(f"================================================================================\n")
        
        # Use the helper method to ensure minimal state exists
        state = self._ensure_minimal_state(state)
        
        # Print available state keys for debugging
        print(f"🔑 AVAILABLE STATE KEYS: {list(state.keys())}")
        print(f"🔑 GLOBAL STATE KEYS: {list(state['metadata']['global_state'].keys())}")
        
        # Get the user message for classification
        message = state.get('user_message', None)
        
        # Handle cases where user_message is None
        if message is None:
            print(f"⚠️ WARNING: user_message is None, setting default values")
            state["argument_classification"] = {"dominant_type": "logical"}
            state["user_persuasion_type"] = "logos"  # Default to logos if no message
            
            # Store in global state too
            state["metadata"]["global_state"]["user_persuasion_type"] = "logos"
            state["metadata"]["global_state"]["counter_persuasion_type"] = "logos"
            print(f"⚠️ DEFAULT: Set persuasion types to 'logos' in global_state")
            
            print(f"🔍 ARGUMENT CLASSIFIER COMPLETED (DEFAULT VALUES)")
            print(f"================================================================================\n")
            return state
        
        # If user_message is a string, convert it to a ConvoMessage object
        if isinstance(message, str):
            print(f"⚠️ WARNING: user_message is a string, converting to ConvoMessage")
            state['user_message'] = ConvoMessage(history=f"User: {message}", content=message)
            message = state['user_message']
        
        # Extract content from message object
        if hasattr(message, 'content'):
            message_content = message.content
        elif isinstance(message, dict) and "content" in message:
            message_content = message["content"]
        elif isinstance(message, str):
            message_content = message
        else:
            try:
                message_content = str(message)
            except Exception as e:
                print(f"⚠️ ERROR: Failed to extract content from message: {e}")
                state["argument_classification"] = {"dominant_type": "logical"}
                state["user_persuasion_type"] = "logos"  # Default to logos if can't analyze
                
                # Store in global state too
                state["metadata"]["global_state"]["user_persuasion_type"] = "logos"
                state["metadata"]["global_state"]["counter_persuasion_type"] = "logos"
                print(f"⚠️ FALLBACK: Set user_persuasion_type and counter_persuasion_type to 'logos' in state and global_state")
                
                return state
        
        # Get request ID from metadata
        request_id = None
        if "metadata" in state and "chat_id" in state["metadata"] and "turn_id" in state["metadata"]:
            # Create a unique request ID combining chat_id and turn_id
            request_id = f"{state['metadata']['chat_id']}_{state['metadata']['turn_id']}"
            print(f"📝 Request ID: {request_id}")
            
            # CRITICAL: Check if this request has already been processed
            if request_id in ArgumentClassifier._processed_requests:
                print(f"🔄 SKIPPING: This request has already been processed (Request ID: {request_id})")
                
                # CRITICAL: Ensure classification results are still available in global state
                if "metadata" in state and "global_state" in state["metadata"] and "user_persuasion_type" in state["metadata"]["global_state"]:
                    user_persuasion_type = state["metadata"]["global_state"]["user_persuasion_type"]
                    # Copy from global state to direct state to ensure it's available
                    state["user_persuasion_type"] = user_persuasion_type
                    print(f"✅ Copied user_persuasion_type from global state to direct state: {user_persuasion_type}")
                    
                print(f"🔍 ARGUMENT CLASSIFIER COMPLETED (SKIPPED)")
                print(f"================================================================================\n")
                return state
            
        # CRITICAL: Skip processing if we've already classified this exact content
        if message_content == ArgumentClassifier._last_classified_content:
            print("🔄 SKIPPING: Content already classified in this session")
            
            # CRITICAL: Ensure classification results are still available in global state
            # First ensure global_state exists
            if "user_persuasion_type" in state["metadata"]["global_state"]:
                user_persuasion_type = state["metadata"]["global_state"]["user_persuasion_type"]
                # Copy from global state to direct state to ensure it's available
                state["user_persuasion_type"] = user_persuasion_type
                print(f"✅ Copied user_persuasion_type from global state to direct state: {user_persuasion_type}")
            else:
                # Default to logos if not found in global state
                state["user_persuasion_type"] = "logos"
                state["metadata"]["global_state"]["user_persuasion_type"] = "logos"
                print("⚠️ user_persuasion_type not found in global state, defaulting to logos")
                
            print(f"🔍 ARGUMENT CLASSIFIER COMPLETED (SKIPPED)")
            print(f"================================================================================\n")
            return state
            
        # CRITICAL: Track number of executions
        ArgumentClassifier._execution_count += 1
        print(f"🧮 ARGUMENT CLASSIFIER EXECUTION COUNT: {ArgumentClassifier._execution_count}")
            
        # Log the message for debugging
        print(f"🔍 Analyzing user argument: {message_content[:150]}...")
        
        try:
            # Use the argument_classifier_prompt from prompts_for_debate_opp
            classify_prompt = PromptTemplate.from_template(argument_classifier_prompt)
            
            # Generate classification prompt
            input_prompt = classify_prompt.invoke({"message": message_content})
            
            # Call LLM to classify the argument
            final_chain = self.llm | StrOutputParser()
            classification_result = final_chain.invoke(input_prompt.text)
            
            try:
                # Parse the classification result to a dictionary
                classification = json.loads(classification_result)
                print(f"✅ Classification results: {json.dumps(classification, indent=2)}")
                
                # Store classification in state
                state["argument_classification"] = classification
                
                # Also store in global state
                state["metadata"]["global_state"]["argument_classification"] = classification
                
                # CRITICAL: Explicitly identify the user's persuasion type and store in state
                dominant_type = classification.get("dominant_type", "logical").lower()
                
                # Map dominant_type to persuasion_type for easier downstream processing
                user_persuasion_type = "logos"  # Default to logos
                
                if dominant_type == "emotional" or dominant_type == "emotion":
                    user_persuasion_type = "pathos"
                    print(f"🔥 DETECTED USER ARGUMENT IS PATHOS (emotional)")
                elif dominant_type == "ethical" or dominant_type == "moral" or dominant_type == "values":
                    user_persuasion_type = "ethos"
                    print(f"🏛️ DETECTED USER ARGUMENT IS ETHOS (ethical/values-based)")
                elif dominant_type == "logical" or dominant_type == "rational":
                    user_persuasion_type = "logos"
                    print(f"🧠 DETECTED USER ARGUMENT IS LOGOS (logical/rational)")
                
                # Store the persuasion type in state and global_state for access by other workers
                state["user_persuasion_type"] = user_persuasion_type
                state["metadata"]["global_state"]["user_persuasion_type"] = user_persuasion_type
                print(f"✅ SET user_persuasion_type in state and global_state: {user_persuasion_type.upper()}")
                
                # REMOVED: Don't set counter_persuasion_type here - let the EffectivenessEvaluator determine that
                # The counter_persuasion_type will be set later based on effectiveness evaluation
                
                print(f"🔍 ARGUMENT CLASSIFIER COMPLETED")
                print(f"🔑 STATE KEYS: {list(state.keys())}")
                print(f"🔑 GLOBAL STATE KEYS: {list(state['metadata']['global_state'].keys())}")
                print(f"================================================================================\n")
                
                return state
            except json.JSONDecodeError:
                print(f"⚠️ ERROR: Failed to parse classification result as JSON")
                print(f"⚠️ Raw result: {classification_result}")
                
                # Fallback classification
                state["argument_classification"] = {"dominant_type": "logical"}
                state["user_persuasion_type"] = "logos"  # Default to logos if parse fails
                
                # Store in global state too
                state["metadata"]["global_state"]["user_persuasion_type"] = "logos"
                state["metadata"]["global_state"]["counter_persuasion_type"] = "logos"
                state["metadata"]["global_state"]["current_persuasion_type"] = "logos"
                print(f"⚠️ FALLBACK: Set all persuasion types to 'logos' in state and global_state due to JSON error")
                
                return state
        except Exception as e:
            print(f"⚠️ ERROR: Exception occurred during classification: {e}")
            print(f"⚠️ Traceback: {traceback.format_exc()}")
            
            # Fallback classification
            state["argument_classification"] = {"dominant_type": "logical"}
            state["user_persuasion_type"] = "logos"  # Default to logos if exception occurs
            state["counter_persuasion_type"] = "logos"
            state["current_persuasion_type"] = "logos"
            
            # Store in global state too
            state["metadata"]["global_state"]["user_persuasion_type"] = "logos"
            state["metadata"]["global_state"]["counter_persuasion_type"] = "logos"
            state["metadata"]["global_state"]["current_persuasion_type"] = "logos"
            print(f"⚠️ FALLBACK: Set all persuasion types to 'logos' in state and global_state due to exception")
            
            return state

    def _count_persuasion_indicators(self, content):
        """Count indicators for each persuasion type in the content.
        
        Args:
            content: The text content to analyze
            
        Returns:
            Tuple of (pathos_count, logos_count, ethos_count)
        """
        # Convert content to lowercase for case-insensitive matching
        content_lower = content.lower()
        
        # Comprehensive keyword sets for each persuasion type
        pathos_indicators = [
            # Basic emotion words
            'feel', 'feeling', 'feelings', 'emotion', 'emotional', 'emotions', 
            'heart', 'heartfelt', 'love', 'hate', 'fear', 'worry', 'anxiety',
            'happy', 'sad', 'angry', 'upset', 'concerned', 'care', 'caring',
            'suffer', 'pain', 'hurt', 'damage', 'harm', 'devastating', 'terrible',
            'awful', 'horrible', 'good', 'bad', 'joy', 'sorrow', 'passionate',
            
            # Strong emotional indicators
            'desire', 'want', 'need', 'hope', 'dream', 'wish', 'please', 'suffering',
            'devastating', 'unfair', 'unjust', 'kind', 'cruel', 'scary', 'frightening',
            'terrifying', 'exciting', 'thrilling', 'wonderful', 'amazing', 'incredible',
            'unbelievable', 'shocking', 'disturbing', 'moving', 'touching',
            
            # Personal experience indicators
            'i feel', 'i felt', 'i was', 'my experience', 'personally', 'for me',
            'in my life', 'what i went through', 'i remember', 'i recall', 'my family',
            'my children', 'my parents', 'my friend', 'my community', 'my generation',
            
            # Emotional descriptors
            'beautiful', 'ugly', 'tragic', 'inspiring', 'hopeful', 'depressing',
            'uplifting', 'distressing', 'outrageous', 'infuriating', 'comforting',
            'reassuring', 'unsettling', 'disturbing', 'delightful', 'horrifying',
            
            # Value-based appeals
            'right', 'wrong', 'moral', 'immoral', 'ethical', 'unethical', 'fair',
            'unfair', 'justice', 'injustice', 'equality', 'freedom', 'liberty',
            'rights', 'responsibility', 'duty', 'values', 'beliefs'
        ]
        
        logos_indicators = [
            # Fact-based language
            'fact', 'facts', 'evidence', 'data', 'statistic', 'statistics', 
            'study', 'studies', 'research', 'analysis', 'analyze', 'logical',
            'logic', 'reason', 'reasoning', 'rational', 'rationally', 'objectively',
            'objective', 'scientifically', 'science', 'scientific', 'proven',
            'proof', 'prove', 'demonstrates', 'demonstrate', 'shows', 'calculations',
            
            # Argumentative structures
            'therefore', 'thus', 'hence', 'consequently', 'as a result', 'due to',
            'because', 'since', 'given that', 'it follows that', 'clearly',
            'obviously', 'evidently', 'demonstrably', 'undeniably', 'must be',
            'necessarily', 'sufficiently', 'adequately', 'conclusively',
            
            # Data references
            'percent', '%', 'rates', 'figures', 'numbers', 'probability', 'likelihood',
            'frequency', 'correlation', 'causation', 'relationship', 'trend',
            'increase', 'decrease', 'growth', 'decline', 'higher', 'lower',
            
            # Expert references
            'according to', 'experts say', 'research shows', 'studies indicate',
            'data suggests', 'evidence demonstrates', 'analysis reveals'
        ]
        
        ethos_indicators = [
            # Authority references
            'expert', 'authority', 'professor', 'doctor', 'scientist', 'researcher',
            'scholar', 'professional', 'specialist', 'leader', 'official', 'credible',
            'reputable', 'respected', 'trusted', 'qualified', 'experienced',
            
            # Credibility language
            'credentials', 'qualification', 'background', 'reputation', 'track record',
            'history', 'established', 'proven', 'verified', 'certified', 'acknowledged',
            'recognized', 'endorsed', 'supported', 'backed', 'authorized',
            
            # Character appeals
            'integrity', 'honest', 'honorable', 'principled', 'ethical', 'moral',
            'virtuous', 'trustworthy', 'reliable', 'dependable', 'consistent',
            'genuine', 'authentic', 'transparent', 'upright', 'decent',
            
            # Institution references
            'university', 'institute', 'organization', 'association', 'foundation',
            'government', 'agency', 'journal', 'publication', 'committee', 'board',
            'council', 'commission', 'department', 'authority', 'administration'
        ]
        
        # Initialize counters for each type
        pathos_count = 0
        logos_count = 0
        ethos_count = 0
        
        # Check for personal stories (strong indicator of pathos)
        personal_story_indicators = [
            'i experienced', 'my own experience', 'happened to me', 
            'i went through', 'in my case', 'my story', 'my journey',
            'my struggle', 'my life', 'personally affected', 'my family'
        ]
        has_personal_story = any(indicator in content_lower for indicator in personal_story_indicators)
        
        # Check for statistical language (strong indicator of logos)
        statistical_indicators = [
            'according to the data', 'statistics show', 'research demonstrates',
            'the evidence indicates', 'studies have found', 'the numbers reveal',
            '% of', 'percent of', 'the rate of', 'significantly increased',
            'statistically significant', 'data analysis shows', 'figures demonstrate'
        ]
        has_statistical_language = any(indicator in content_lower for indicator in statistical_indicators)
        
        # Check for authority appeals (strong indicator of ethos)
        authority_indicators = [
            'as an expert', 'in my professional opinion', 'my credentials',
            'with my background in', 'in my field of expertise', 'my years of experience',
            'based on my qualifications', 'as a professional', 'as a specialist',
            'according to experts', 'authorities agree', 'established institutions'
        ]
        has_authority_appeal = any(indicator in content_lower for indicator in authority_indicators)
        
        # Count occurrences of each type of indicator
        for word in pathos_indicators:
            if word in content_lower:
                pathos_count += 1
                
        for word in logos_indicators:
            if word in content_lower:
                logos_count += 1
                
        for word in ethos_indicators:
            if word in content_lower:
                ethos_count += 1
        
        # Apply adjustments for strong signals
        if has_personal_story:
            pathos_count += 5  # Strong boost for personal narratives
            print("   STRONG EMOTIONAL SIGNAL: Personal story detected")
            
        if has_statistical_language:
            logos_count += 5  # Strong boost for statistical references
            print("   STRONG LOGICAL SIGNAL: Statistical language detected")
            
        if has_authority_appeal:
            ethos_count += 5  # Strong boost for authority appeals
            print("   STRONG ETHICAL SIGNAL: Authority appeal detected")
        
        # Consider sentence structure and length for pathos
        # Emotional content often has shorter sentences, exclamations, questions
        sentences = content.split('.')
        short_emotional_sentences = sum(1 for s in sentences if len(s.strip()) < 15 and ('!' in s or '?' in s))
        if short_emotional_sentences >= 2:
            pathos_count += 3
            print("   EMOTIONAL SIGNAL: Emotional sentence structure detected")
        
        # First-person language is often a pathos signal
        first_person_count = content_lower.count(' i ') + content_lower.count('i\'m') + content_lower.count('i\'ve')
        if first_person_count > 3:
            pathos_count += 2
            print("   EMOTIONAL SIGNAL: Frequent first-person language detected")
        
        return pathos_count, logos_count, ethos_count

    def _generate_classification_reasoning(self, dominant_type, counts):
        """Generate reasoning for the classification based on indicator counts.
        
        Args:
            dominant_type: The dominant persuasion type detected
            counts: Dictionary of counts for each type
            
        Returns:
            Reasoning text explaining the classification
        """
        if dominant_type == 'emotional':
            return (f"The argument uses emotional language and appeals to feelings with {counts['emotional']} emotional indicators "
                    f"compared to {counts['logical']} logical and {counts['ethical']} ethical indicators.")
        elif dominant_type == 'logical':
            return (f"The argument relies on facts, data, and logical reasoning with {counts['logical']} logical indicators "
                    f"compared to {counts['emotional']} emotional and {counts['ethical']} ethical indicators.")
        else:  # ethical
            return (f"The argument appeals to credibility, authority, and ethical principles with {counts['ethical']} ethical indicators "
                    f"compared to {counts['emotional']} emotional and {counts['logical']} logical indicators.") 
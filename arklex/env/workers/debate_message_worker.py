import logging
from typing import Any
import importlib
import traceback

from arklex.env.workers.worker import register_worker
from arklex.env.workers.message_worker import MessageWorker
from arklex.utils.graph_state import MessageState, ConvoMessage
from arklex.env.prompts import load_prompts
from langchain.prompts import PromptTemplate
from arklex.utils.utils import chunk_string
from langchain_core.output_parsers import StrOutputParser
from arklex.utils.model_config import MODEL

logger = logging.getLogger(__name__)

@register_worker
class DebateMessageWorker(MessageWorker):
    """A specialized MessageWorker for debate opponents that incorporates persuasion strategies."""
    
    description = "Delivers debate responses using the most effective persuasion type based on user interactions."

    def __init__(self):
        super().__init__()
        logger.info("DebateMessageWorker initialized")
        
    def _get_persuasion_info(self, persuasion_type: str) -> str:
        """Get formatted information about the persuasion type being used."""
        type_descriptions = {
            "pathos": "💭 *I'm using emotional appeals (pathos) in this response to connect with your values and feelings.*",
            "logos": "💭 *I'm using logical reasoning (logos) in this response to present a fact-based argument.*",
            "ethos": "💭 *I'm using ethical appeals (ethos) in this response to address moral principles and credibility.*"
        }
        
        return type_descriptions.get(persuasion_type, "")

    def _run_database_checks(self, state: MessageState) -> MessageState:
        """Run database checks and updates after every user response.
        
        This method is called in the execute method to ensure database operations
        happen consistently, regardless of task graph structure.
        
        Args:
            state: Current message state
            
        Returns:
            Updated state after database operations
        """
        print("\n================================================================================")
        print(f"💾 RUNNING DATABASE CHECKS (from DebateMessageWorker)")
        print(f"================================================================================\n")
        
        # Add a marker that this state was processed by DebateMessageWorker
        # Do this BEFORE any other operations to ensure consistent state identification
        state["source_worker"] = "DebateMessageWorker"
        
        try:
            # CRITICAL: Ensure all required state variables exist
            print(f"🔑 AVAILABLE STATE KEYS BEFORE CLEANUP: {list(state.keys())}")
            
            # Ensure basic required keys exist
            required_defaults = {
                'user_message': None,
                'message_flow': "",
                'bot_config': {},
                'sys_instruct': ""
            }
            
            # Fill in any missing required keys
            for key, default_value in required_defaults.items():
                if key not in state:
                    print(f"⚠️ Adding missing required key: {key}")
                    state[key] = default_value
            
            # Ensure metadata and global_state are initialized
            if "metadata" not in state:
                print(f"⚠️ Adding missing metadata dict")
                state["metadata"] = {}
                
            if "global_state" not in state["metadata"]:
                print(f"⚠️ Adding missing global_state dict")
                state["metadata"]["global_state"] = {}
            
            # Log global state
            print(f"🔑 GLOBAL STATE KEYS: {list(state['metadata']['global_state'].keys())}")
            
            # Check if we have enough context to run database operations
            if "trajectory" not in state or not state.get("trajectory"):
                print("⏸️ SKIPPING DATABASE CHECKS - No trajectory found")
                return state
                
            # Ensure user has responded at least once
            trajectory = state.get("trajectory", [])
            message_count = len(trajectory)
            if message_count < 3:  # Need at least welcome, user1, bot1
                print(f"⏸️ SKIPPING DATABASE CHECKS - Not enough message history (current count: {message_count})")
                return state
                
            # Lazy-load the DebateDatabaseWorker to avoid circular imports
            try:
                # Try to import the DatabaseWorker dynamically
                module = importlib.import_module('arklex.env.workers.debate_database_worker')
                db_worker_class = getattr(module, 'DebateDatabaseWorker')
                
                # Create a temporary instance just for this operation
                db_worker = db_worker_class()
                
                # Create a deep clone of the state for database operations
                db_state = MessageState()
                for key, value in state.items():
                    db_state[key] = value
                
                # Set the operation to "update" to indicate we want to update effectiveness scores
                db_state["operation"] = "update"
                
                # CRITICAL: Ensure persuasion types are properly set in both direct state and global state
                # Check for user_persuasion_type
                user_persuasion_type = None
                
                # First check global state (higher priority)
                if "metadata" in state and "global_state" in state["metadata"] and "user_persuasion_type" in state["metadata"]["global_state"]:
                    user_persuasion_type = state["metadata"]["global_state"]["user_persuasion_type"]
                    print(f"✅ Found user_persuasion_type in global state: {user_persuasion_type}")
                # Then check direct state
                elif "user_persuasion_type" in state:
                    user_persuasion_type = state["user_persuasion_type"]
                    print(f"✅ Found user_persuasion_type in direct state: {user_persuasion_type}")
                
                # If we found a user_persuasion_type, ensure it's in both places
                if user_persuasion_type:
                    db_state["user_persuasion_type"] = user_persuasion_type
                    db_state["metadata"]["global_state"]["user_persuasion_type"] = user_persuasion_type
                    print(f"✅ Ensured user_persuasion_type is in both direct state and global state: {user_persuasion_type}")
                else:
                    # Default to logos if no persuasion type found
                    user_persuasion_type = "logos"
                    db_state["user_persuasion_type"] = user_persuasion_type
                    db_state["metadata"]["global_state"]["user_persuasion_type"] = user_persuasion_type
                    print(f"⚠️ No user_persuasion_type found. Defaulting to: {user_persuasion_type}")
                
                # Check for counter_persuasion_type
                counter_persuasion_type = None
                
                # First check global state (higher priority)
                if "metadata" in state and "global_state" in state["metadata"] and "counter_persuasion_type" in state["metadata"]["global_state"]:
                    counter_persuasion_type = state["metadata"]["global_state"]["counter_persuasion_type"]
                    print(f"✅ Found counter_persuasion_type in global state: {counter_persuasion_type}")
                # Then check direct state
                elif "counter_persuasion_type" in state:
                    counter_persuasion_type = state["counter_persuasion_type"]
                    print(f"✅ Found counter_persuasion_type in direct state: {counter_persuasion_type}")
                
                # If we found a counter_persuasion_type, ensure it's in both places
                if counter_persuasion_type:
                    db_state["counter_persuasion_type"] = counter_persuasion_type
                    db_state["metadata"]["global_state"]["counter_persuasion_type"] = counter_persuasion_type
                    print(f"✅ Ensured counter_persuasion_type is in both direct state and global state: {counter_persuasion_type}")
                else:
                    # Default to same as user_persuasion_type if no counter type found
                    counter_persuasion_type = user_persuasion_type
                    db_state["counter_persuasion_type"] = counter_persuasion_type
                    db_state["metadata"]["global_state"]["counter_persuasion_type"] = counter_persuasion_type
                    print(f"⚠️ No counter_persuasion_type found. Defaulting to: {counter_persuasion_type}")
                
                # Also set current_persuasion_type for consistency
                db_state["current_persuasion_type"] = counter_persuasion_type
                db_state["metadata"]["global_state"]["current_persuasion_type"] = counter_persuasion_type
                
                # Check if we have an effectiveness score to update
                if "evaluated_effectiveness_score" in state:
                    db_state["evaluated_effectiveness_score"] = state["evaluated_effectiveness_score"]
                    print(f"✅ Found evaluated_effectiveness_score in state: {state['evaluated_effectiveness_score']}")
                    # Also copy to global state
                    db_state["metadata"]["global_state"]["evaluated_effectiveness_score"] = state["evaluated_effectiveness_score"]
                # Also check global state
                elif "metadata" in state and "global_state" in state["metadata"] and "evaluated_effectiveness_score" in state["metadata"]["global_state"]:
                    db_state["evaluated_effectiveness_score"] = state["metadata"]["global_state"]["evaluated_effectiveness_score"]
                    print(f"✅ Found evaluated_effectiveness_score in global state: {state['metadata']['global_state']['evaluated_effectiveness_score']}")
                else:
                    print("⚠️ No evaluated_effectiveness_score found. Database update will be skipped.")
                
                # Execute the database worker with our prepared state
                print("🔄 EXECUTING DATABASE WORKER")
                updated_state = db_worker.execute(db_state)
                
                # CRITICAL: Copy important values back from updated state
                # First, update our global state with any new values from the database worker
                for key, value in updated_state["metadata"]["global_state"].items():
                    state["metadata"]["global_state"][key] = value
                    print(f"✅ Copied {key} from database worker to global state")
                
                # Now check if there are any direct state values to copy back
                important_keys = ["user_persuasion_type", "counter_persuasion_type", "current_persuasion_type", 
                                  "evaluated_effectiveness_score", "best_persuasion_type"]
                
                for key in important_keys:
                    if key in updated_state:
                        state[key] = updated_state[key]
                        print(f"✅ Copied {key} from database worker to direct state: {updated_state[key]}")
                
                print("💾 DATABASE CHECKS COMPLETED SUCCESSFULLY")
                
            except ImportError:
                print("⚠️ ERROR: Could not import DebateDatabaseWorker module")
                print(f"⚠️ Traceback: {traceback.format_exc()}")
            except Exception as e:
                print(f"⚠️ ERROR: Exception during database worker execution: {e}")
                print(f"⚠️ Traceback: {traceback.format_exc()}")
        except Exception as e:
            print(f"⚠️ ERROR: Exception during database checks: {e}")
            print(f"⚠️ Traceback: {traceback.format_exc()}")
            
        # Even if there was an error, return the original state to ensure execution continues
        return state

    def generator(self, state: MessageState) -> MessageState:
        """Generate responses using the best persuasion strategy."""
        print("\n================================================================================")
        print(f"🗣️ DEBATE MESSAGE WORKER EXECUTING")
        print(f"================================================================================\n")
        
        print(f"Available state keys: {list(state.keys())}")
        
        # CRITICAL: Check if counter_argument is directly available from PersuasionWorker
        if "counter_argument" in state:
            counter_argument = state["counter_argument"]
            print(f"✅ FOUND counter_argument DIRECTLY FROM PERSUASION WORKER")
            print(f"✅ USING counter_argument AS DIRECT RESPONSE")
            state["message_flow"] = ""
            state["response"] = counter_argument
            return state
        
        # EXPLICIT CHECK FOR USER_PERSUASION_TYPE
        # If the user used pathos, we should FORCE pathos response
        user_type = None
        
        # Check all possible locations for user_persuasion_type in highest to lowest priority
        if "metadata" in state and "global_state" in state["metadata"] and "user_persuasion_type" in state["metadata"]["global_state"]:
            user_type = state["metadata"]["global_state"]["user_persuasion_type"].lower()
            print(f"🌐 FOUND user_persuasion_type in global state: {user_type.upper()}")
        elif "user_persuasion_type" in state:
            user_type = state["user_persuasion_type"].lower()
            print(f"📄 FOUND user_persuasion_type in direct state: {user_type.upper()}")
        
        # If user used pathos, ALWAYS use pathos response
        if user_type == "pathos":
            print(f"🔥 USER USED PATHOS - FORCING PATHOS RESPONSE MODE")
            
            # Try to find the best pathos response from multiple sources
            # 1. Check for direct pathos_persuasion_response
            if "pathos_persuasion_response" in state:
                pathos_response = state["pathos_persuasion_response"]
                if isinstance(pathos_response, dict) and "counter_argument" in pathos_response:
                    counter_argument = pathos_response["counter_argument"]
                    print(f"⚡⚡⚡ USING PATHOS RESPONSE DIRECTLY FROM PERSUASION WORKER")
                    state["message_flow"] = ""
                    state["response"] = counter_argument
                    
                    # Store the counter_argument as bot_message for future evaluation
                    user_message = state['user_message']
                    user_history = ""
                    if hasattr(user_message, "history"):
                        user_history = user_message.history
                    state["bot_message"] = ConvoMessage(history=user_history, message=counter_argument)
                    print(f"✅ STORED PATHOS RESPONSE AS bot_message FOR FUTURE EVALUATION")
                    
                    return state
                    
            # 2. Check global state for pathos guidance
            if "metadata" in state and "global_state" in state["metadata"] and "persuasion_guidance" in state["metadata"]["global_state"]:
                guidance = state["metadata"]["global_state"]["persuasion_guidance"]
                print(f"🌐 FOUND PATHOS GUIDANCE IN GLOBAL STATE: {guidance[:100]}...")
                
                # Use this guidance to generate a pure pathos response
                print(f"🎭 GENERATING PATHOS RESPONSE USING GUIDANCE")
                pathos_response = self._generate_pure_pathos_response(state, guidance)
                
                state["message_flow"] = ""
                state["response"] = pathos_response
                return state
                    
            # 3. Generate a default emotional response (last resort)
            print(f"⚠️ NO PATHOS RESPONSE FOUND - GENERATING FALLBACK EMOTIONAL RESPONSE")
            user_message = state.get('user_message', {})
            user_content = user_message.content if hasattr(user_message, 'content') else str(user_message)
            
            pathos_fallback = self._generate_fallback_pathos_response(user_content)
            state["message_flow"] = ""
            state["response"] = pathos_fallback
            return state
        
        # CRITICAL FIX: If pathos_persuasion_response exists, use it directly
        if "pathos_persuasion_response" in state:
            pathos_response = state["pathos_persuasion_response"]
            if isinstance(pathos_response, dict) and "counter_argument" in pathos_response:
                counter_argument = pathos_response["counter_argument"]
                print(f"⚡⚡⚡ CRITICAL OVERRIDE: Using pathos response DIRECTLY from persuasion worker")
                print(f"⚡⚡⚡ BYPASSING all other message processing to preserve emotional storytelling")
                state["message_flow"] = ""
                state["response"] = counter_argument
                
                # Store the counter_argument as bot_message for future evaluation
                user_message = state['user_message']
                user_history = ""
                if hasattr(user_message, "history"):
                    user_history = user_message.history
                state["bot_message"] = ConvoMessage(history=user_history, message=counter_argument)
                print(f"✅ STORED PATHOS RESPONSE AS bot_message FOR FUTURE EVALUATION")
                
                return state
        
        # get the input message
        user_message = state['user_message']
        orchestrator_message = state['orchestrator_message']
        message_flow = state.get('response', "") + "\n" + state.get("message_flow", "")
        
        # Check global state for counter_persuasion_type (highest priority)
        persuasion_type = None
        if "metadata" in state and "global_state" in state["metadata"]:
            global_state = state["metadata"]["global_state"]
            if "counter_persuasion_type" in global_state:
                persuasion_type = global_state["counter_persuasion_type"]
                print(f"🌐 FOUND counter_persuasion_type in global state: {persuasion_type.upper()}")
                
                # CRITICAL: Look for the response generated with this persuasion type
                response_key = f"{persuasion_type}_persuasion_response"
                if response_key in state:
                    persuasion_response = state[response_key]
                    if isinstance(persuasion_response, dict) and "counter_argument" in persuasion_response:
                        counter_argument = persuasion_response["counter_argument"]
                        
                        # CRITICAL FIX: For pathos responses, preserve the original storytelling
                        if persuasion_type == "pathos":
                            print(f"✅ PRESERVING PATHOS STORYTELLING RESPONSE DIRECTLY")
                            state["message_flow"] = ""
                            state["response"] = counter_argument
                            
                            # Store the counter_argument as bot_message for future evaluation
                            user_history = ""
                            if hasattr(user_message, "history"):
                                user_history = user_message.history
                            state["bot_message"] = ConvoMessage(history=user_history, message=counter_argument)
                            print(f"✅ STORED PATHOS RESPONSE AS bot_message FOR FUTURE EVALUATION")
                            
                            return state
                            
                        message_flow = counter_argument
                        print(f"✅ USING {persuasion_type.upper()} RESPONSE FROM {response_key}")
                        # Override message_flow with the correct persuasion response
                        state["message_flow"] = message_flow
                
        # If not found in global state, check direct state variables
        if not persuasion_type:
            # Get current persuasion type, prioritizing counter_persuasion_type over others
            counter_persuasion_type = state.get("counter_persuasion_type")
            best_persuasion_type = state.get("best_persuasion_type")
            current_persuasion_type = state.get("current_persuasion_type")
            persuasion_scores = state.get("persuasion_scores", {})
            
            # Determine which persuasion type to use in priority order
            persuasion_type = counter_persuasion_type or best_persuasion_type or current_persuasion_type or "logos"
            
            # Log persuasion scores if available
            if persuasion_scores:
                print("📊 CURRENT EFFECTIVENESS SCORES:")
                for p_type, score in persuasion_scores.items():
                    print(f"  - {p_type.upper()}: {score:.2f} ({int(score * 100)}%)")
            
            # Look for the response generated with this persuasion type
            response_key = f"{persuasion_type}_persuasion_response"
            if response_key in state:
                persuasion_response = state[response_key]
                if isinstance(persuasion_response, dict) and "counter_argument" in persuasion_response:
                    counter_argument = persuasion_response["counter_argument"]
                    
                    # CRITICAL FIX: For pathos responses, preserve the original storytelling
                    if persuasion_type == "pathos":
                        print(f"✅ PRESERVING PATHOS STORYTELLING RESPONSE DIRECTLY")
                        state["message_flow"] = ""
                        state["response"] = counter_argument
                        
                        # Store the counter_argument as bot_message for future evaluation
                        user_history = ""
                        if hasattr(user_message, "history"):
                            user_history = user_message.history
                        state["bot_message"] = ConvoMessage(history=user_history, message=counter_argument)
                        print(f"✅ STORED PATHOS RESPONSE AS bot_message FOR FUTURE EVALUATION")
                        
                        return state
                        
                    message_flow = counter_argument
                    print(f"✅ USING {persuasion_type.upper()} RESPONSE FROM {response_key}")
                    # Override message_flow with the correct persuasion response
                    state["message_flow"] = message_flow
        
        print(f"🎭 USING {persuasion_type.upper()} PERSUASION STRATEGY FOR FINAL RESPONSE")
        persuasion_info = self._get_persuasion_info(persuasion_type)

        # get the orchestrator message content
        orch_msg_content = "None" if not orchestrator_message.message else orchestrator_message.message
        orch_msg_attr = orchestrator_message.attribute
        direct_response = orch_msg_attr.get('direct_response', False)
        if direct_response:
            state["message_flow"] = ""
            state["response"] = orch_msg_content
            return state
        
        prompts = load_prompts(state["bot_config"])
        if message_flow and message_flow != "\n":
            prompt = PromptTemplate.from_template(prompts["message_flow_generator_prompt"])
            # Add persuasion info to the context
            context = message_flow
            if persuasion_info:
                context += f"\n\n{persuasion_info}"
            input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history, "context": context})
        else:
            # If no message flow but we have persuasion info, use the context version of the prompt
            if persuasion_info:
                prompt = PromptTemplate.from_template(prompts["message_flow_generator_prompt"])
                input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history, "context": persuasion_info})
            else:
                prompt = PromptTemplate.from_template(prompts["message_generator_prompt"])
                input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history})
                
        logger.info(f"Prompt: {input_prompt.text}")
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        answer = final_chain.invoke(chunked_prompt)

        state["message_flow"] = ""
        state["response"] = answer
        
        # Store the final answer as bot_message for future evaluation
        user_history = ""
        if hasattr(user_message, "history"):
            user_history = user_message.history
        state["bot_message"] = ConvoMessage(history=user_history, message=answer)
        print(f"✅ STORED FINAL GENERATOR RESPONSE AS bot_message FOR FUTURE EVALUATION")
        
        return state
    
    def _generate_pure_pathos_response(self, state, guidance):
        """Generate a pure pathos response following the guidance."""
        user_message = state.get('user_message', {})
        user_content = user_message.content if hasattr(user_message, 'content') else str(user_message)
        
        prompt = f"""You are an expert in emotional persuasion (pathos). Your ONLY task is to create a highly emotional counter-argument against the user's position using personal stories, emotional appeals, and value-based language.

User's argument: {user_content}

GUIDANCE: {guidance}

CRITICAL INSTRUCTIONS FOR PATHOS RESPONSE:
1. DO NOT USE ANY LOGICAL REASONING OR FACTUAL EVIDENCE - NO STATISTICS, STUDIES, OR EXPERTISE CITATIONS.
2. YOU MUST RESPOND WITH A FIRST-PERSON PERSONAL STORY OR ANECDOTE - THIS IS MANDATORY.
3. Your response MUST appeal ONLY to emotions - use vivid imagery, emotive language, and create a deep emotional connection.
4. COMPLETELY AVOID referencing studies, statistics, abstract principles, systemic analysis, or expert opinions.
5. Focus EXCLUSIVELY on raw emotions and feelings - NOT on analytical reasoning of any kind.
6. Use highly emotional language throughout - words like "devastated," "heartbroken," "overjoyed," "terrified," etc.
7. START WITH ONE OF THESE PHRASES:
   - "Let me share something deeply personal that still brings tears to my eyes..."
   - "I'll never forget the day I witnessed firsthand the devastating impact..."
   - "My heart still aches when I remember what happened to someone I love..."
   - "I was emotionally shattered when I experienced..."

ALWAYS SAY: "I felt," "My heart broke," "I was overcome with emotion," "Tears filled my eyes," "I was devastated"

Create a deeply emotional personal story that opposes the user's view:
"""

        # Generate response
        final_chain = self.llm | StrOutputParser()
        pathos_response = final_chain.invoke(prompt)
        
        print(f"✅ GENERATED PURE PATHOS RESPONSE")
        return pathos_response
    
    def _generate_fallback_pathos_response(self, user_argument):
        """Generate a fallback emotional response when no other pathos response is available."""
        prompt = f"""Create a deeply emotional and personal response to the following argument. You MUST use a first-person emotional story with vivid details.

User's argument: "{user_argument}"

CRITICAL INSTRUCTIONS:
1. START WITH: "Let me share something deeply personal that still brings tears to my eyes..."
2. INCLUDE vivid emotional details and personal experiences
3. USE EMOTIONAL LANGUAGE like: heartbroken, devastated, overcome, tears, anguish, joy
4. AVOID ALL STATISTICS, STUDIES, DATA, or EXPERT OPINIONS
5. RELY ENTIRELY on emotional storytelling and personal connection
6. TELL A PERSONAL STORY that counters the user's argument

Response:"""

        # Generate response
        final_chain = self.llm | StrOutputParser()
        fallback_response = final_chain.invoke(prompt)
        
        print(f"✅ GENERATED FALLBACK PATHOS RESPONSE")
        return fallback_response
    
    def stream_generator(self, state: MessageState) -> MessageState:
        """Stream responses using the best persuasion strategy."""
        # CRITICAL FIX: If pathos_persuasion_response exists, use it directly
        if "pathos_persuasion_response" in state:
            pathos_response = state["pathos_persuasion_response"]
            if isinstance(pathos_response, dict) and "counter_argument" in pathos_response:
                counter_argument = pathos_response["counter_argument"]
                print(f"⚡⚡⚡ CRITICAL OVERRIDE: Using pathos response DIRECTLY from persuasion worker")
                print(f"⚡⚡⚡ BYPASSING all other message processing to preserve emotional storytelling")
                state["message_flow"] = ""
                state["response"] = counter_argument
                return state
        
        # Use the same enhanced persuasion info logic in the streaming version
        user_message = state['user_message']
        orchestrator_message = state['orchestrator_message']
        message_flow = state.get('response', "") + "\n" + state.get("message_flow", "")
        
        # Get current persuasion type, prioritizing best_persuasion_type over current_persuasion_type
        best_persuasion_type = state.get("best_persuasion_type")
        current_persuasion_type = state.get("current_persuasion_type")
        
        # Determine which persuasion type to use
        persuasion_type = best_persuasion_type or current_persuasion_type or ""
        if persuasion_type:
            logger.info(f"Using {persuasion_type} persuasion strategy for streaming response")
            persuasion_info = self._get_persuasion_info(persuasion_type)
        else:
            persuasion_info = ""

        # Process the rest using the streaming implementation
        return super().stream_generator(state)

    def execute(self, state: MessageState) -> MessageState:
        """Generate a debate message based on the determined persuasion type."""
        print("\n================================================================================")
        print(f"🎭 DEBATE MESSAGE WORKER EXECUTING")
        print(f"================================================================================\n")
        
        # CRITICAL: Mark state as being processed by DebateMessageWorker
        # Do this at the very beginning before any other operations
        state["source_worker"] = "DebateMessageWorker"
        
        # CRITICAL: Ensure essential state variables are preserved
        print(f"🔑 AVAILABLE STATE KEYS: {list(state.keys())}")
        required_keys = ['user_message', 'message_flow', 'bot_config', 'sys_instruct']
        for key in required_keys:
            if key not in state:
                print(f"⚠️ WARNING: Required key '{key}' is missing from state!")
                # Don't try to fix here - just log it
        
        # Instead of doing our own classification, respect classification from ArgumentClassifier
        user_persuasion_type = None
        
        # Check for user_persuasion_type in all possible locations
        if "user_persuasion_type" in state:
            user_persuasion_type = state["user_persuasion_type"]
            print(f"📊 FOUND user_persuasion_type IN STATE: {user_persuasion_type.upper()}")
        elif "metadata" in state and "global_state" in state["metadata"] and "user_persuasion_type" in state["metadata"]["global_state"]:
            user_persuasion_type = state["metadata"]["global_state"]["user_persuasion_type"]
            print(f"📊 FOUND user_persuasion_type IN GLOBAL STATE: {user_persuasion_type.upper()}")
        elif "argument_classification" in state:
            # Extract from argument_classification
            arg_class = state["argument_classification"]
            if isinstance(arg_class, dict) and "dominant_type" in arg_class:
                dom_type = arg_class["dominant_type"].lower()
                if dom_type == "emotional" or dom_type == "emotion":
                    user_persuasion_type = "pathos"
                elif dom_type == "logical" or dom_type == "rational":
                    user_persuasion_type = "logos"
                elif dom_type == "ethical" or dom_type == "moral" or dom_type == "values":
                    user_persuasion_type = "ethos"
                else:
                    user_persuasion_type = "logos"  # default
                print(f"📊 EXTRACTED user_persuasion_type FROM argument_classification: {user_persuasion_type.upper()}")
        
        # If we found a user_persuasion_type, use it to determine response type
        if user_persuasion_type:
            # Store it in state for consistent access
            state["user_persuasion_type"] = user_persuasion_type
            if "metadata" not in state:
                state["metadata"] = {}
            if "global_state" not in state["metadata"]:
                state["metadata"]["global_state"] = {}
            state["metadata"]["global_state"]["user_persuasion_type"] = user_persuasion_type
            
            # If it's pathos, force a pathos response
            if user_persuasion_type == "pathos":
                print(f"🔥 USER USED PATHOS ARGUMENTS - GENERATING EMOTIONAL RESPONSE")
                state["counter_persuasion_type"] = "pathos"
                state["metadata"]["global_state"]["counter_persuasion_type"] = "pathos"
                
                # Look for an existing pathos response
                if "pathos_persuasion_response" in state:
                    pathos_response = state["pathos_persuasion_response"]
                    if isinstance(pathos_response, dict) and "counter_argument" in pathos_response:
                        counter_argument = pathos_response["counter_argument"]
                        print(f"⚡⚡⚡ USING PATHOS RESPONSE FROM persuasion_worker")
                        state["response"] = counter_argument
                        
                        # Store for future evaluation
                        user_history = ""
                        if "user_message" in state and hasattr(state["user_message"], "history"):
                            user_history = state["user_message"].history
                        state["bot_message"] = ConvoMessage(history=user_history, message=counter_argument)
                        print(f"✅ STORED PATHOS RESPONSE AS bot_message")
                        
                        # Run database checks before returning
                        state = self._run_database_checks(state)
                        return state
                
                # Generate fallback pathos response if needed
                print(f"⚠️ NO PATHOS RESPONSE FOUND - GENERATING FALLBACK")
                user_content = ""
                if "user_message" in state:
                    user_message = state["user_message"]
                    user_content = user_message.content if hasattr(user_message, "content") else str(user_message)
                
                pathos_response = self._generate_fallback_pathos_response(user_content)
                state["response"] = pathos_response
                
                # Store for future evaluation
                user_history = ""
                if "user_message" in state and hasattr(state["user_message"], "history"):
                    user_history = state["user_message"].history
                state["bot_message"] = ConvoMessage(history=user_history, message=pathos_response)
                print(f"✅ STORED FALLBACK PATHOS RESPONSE AS bot_message")
                
                # Run database checks before returning
                state = self._run_database_checks(state)
                return state
            
            # If it's logos, ensure we use a logical response
            elif user_persuasion_type == "logos":
                print(f"🧠 USER USED LOGOS ARGUMENTS - GENERATING LOGICAL RESPONSE")
                state["counter_persuasion_type"] = "logos"
                state["metadata"]["global_state"]["counter_persuasion_type"] = "logos"
            
            # If it's ethos, ensure we use an ethical response
            elif user_persuasion_type == "ethos":
                print(f"🏛️ USER USED ETHOS ARGUMENTS - GENERATING ETHICAL RESPONSE")
                state["counter_persuasion_type"] = "ethos"
                state["metadata"]["global_state"]["counter_persuasion_type"] = "ethos"
        else:
            # Default to logos if no classification found
            print(f"⚠️ NO ARGUMENT CLASSIFICATION FOUND - DEFAULTING TO LOGOS")
            state["user_persuasion_type"] = "logos"
            state["counter_persuasion_type"] = "logos"
            state["metadata"]["global_state"]["user_persuasion_type"] = "logos"
            state["metadata"]["global_state"]["counter_persuasion_type"] = "logos"
        
        # STEP: Generate response based on current state
        # First check for direct counter_argument from PersuasionWorker
        if "counter_argument" in state:
            print(f"📝 USING counter_argument DIRECTLY FROM PERSUASION WORKER")
            counter_argument = state["counter_argument"]
            state["response"] = counter_argument
            
            # Store the counter_argument as bot_message for future evaluation
            user_history = ""
            if "user_message" in state and hasattr(state["user_message"], "history"):
                user_history = state["user_message"].history
            state["bot_message"] = ConvoMessage(history=user_history, message=counter_argument)
            print(f"✅ STORED counter_argument AS bot_message FOR FUTURE EVALUATION")
            
            # Run database checks before returning
            state = self._run_database_checks(state)
            return state
        
        # Then check for persuasion type-specific responses
        for persuasion_type in ["pathos", "logos", "ethos"]:
            response_key = f"{persuasion_type}_persuasion_response"
            if response_key in state:
                response_obj = state[response_key]
                if isinstance(response_obj, dict) and "counter_argument" in response_obj:
                    print(f"📝 USING {persuasion_type.upper()} PERSUASION RESPONSE")
                    counter_argument = response_obj["counter_argument"]
                    state["response"] = counter_argument
                    
                    # Store the counter_argument as bot_message for future evaluation
                    user_history = ""
                    if "user_message" in state and hasattr(state["user_message"], "history"):
                        user_history = state["user_message"].history
                    state["bot_message"] = ConvoMessage(history=user_history, message=counter_argument)
                    print(f"✅ STORED {persuasion_type.upper()} RESPONSE AS bot_message FOR FUTURE EVALUATION")
                    
                    # Run database checks before returning
                    state = self._run_database_checks(state)
                    return state
        
        # Finally, fall back to standard generator
        print(f"📝 NO DIRECT COUNTER-ARGUMENT FOUND, USING GENERATOR")
        result_state = self.generator(state)
        
        # If generator produced a response, store it as bot_message
        if "response" in result_state:
            response = result_state["response"]
            user_history = ""
            if "user_message" in state and hasattr(state["user_message"], "history"):
                user_history = state["user_message"].history
            result_state["bot_message"] = ConvoMessage(history=user_history, message=response)
            print(f"✅ STORED GENERATOR RESPONSE AS bot_message FOR FUTURE EVALUATION")
            
        # Run database checks before returning
        result_state = self._run_database_checks(result_state)
        return result_state

    def run(self, msg_state: MessageState) -> MessageState:
        """Generate a debate message based on the current context."""
        print("\n================================================================================")
        print(f"💬 DEBATE MESSAGE WORKER - Formatting final response")
        print(f"================================================================================\n")
        
        try:
            # Check if we have a persuasive response from the persuasion worker
            persuasive_response = None
            
            if "persuasive_response" in msg_state:
                persuasive_response = msg_state["persuasive_response"]
                print(f"📄 Found persuasive response in state")
            
            # If no persuasive response is found, check if this is an initial message
            if not persuasive_response:
                # Check if this is the start of a debate (if no messages exist)
                is_initial = True
                if "messages" in msg_state and len(msg_state["messages"]) > 0:
                    is_initial = False
                
                if is_initial:
                    # Generate initial welcome message
                    initial_message = self.welcome_message
                    print(f"👋 INITIAL MESSAGE: Generated welcome message")
                    
                    # Store the message in the state
                    msg_state["assistant_response"] = initial_message
                    return msg_state
                else:
                    # This should not happen - we should have a persuasive response
                    print(f"⚠️ No persuasive response found, and this is not an initial message")
                    msg_state["assistant_response"] = "I apologize, but I couldn't generate a proper debate response. Could you please repeat your position or perhaps try a different approach?"
                    return msg_state
            
            # Get the persuasion type that was used
            persuasion_type = None
            
            # Check global state first (higher priority)
            if "metadata" in msg_state and "global_state" in msg_state["metadata"] and "current_persuasion_type" in msg_state["metadata"]["global_state"]:
                persuasion_type = msg_state["metadata"]["global_state"]["current_persuasion_type"]
                print(f"🌐 Found persuasion_type in global state: {persuasion_type}")
            
            # Check direct state (lower priority)
            elif "current_persuasion_type" in msg_state:
                persuasion_type = msg_state["current_persuasion_type"]
                print(f"📄 Found persuasion_type in direct state: {persuasion_type}")
            
            # Default to logos if we can't determine the type
            if not persuasion_type:
                persuasion_type = "logos"
                print(f"⚠️ Could not determine persuasion type, defaulting to: {persuasion_type}")
            
            # Format the response with persuasion type info if requested
            include_persuasion_info = self.config.get("include_persuasion_info", False)
            
            if include_persuasion_info:
                # Get formatted persuasion info
                persuasion_info = self._get_persuasion_info(persuasion_type)
                
                # Add the persuasion info to the response
                final_response = f"{persuasion_info}\n\n{persuasive_response}"
                print(f"ℹ️ Added persuasion info to response")
            else:
                # Just use the persuasive response as is
                final_response = persuasive_response
            
            # Store the final response in the state
            msg_state["assistant_response"] = final_response
            print(f"✅ Final response prepared using {persuasion_type.upper()} strategy")
            
            # Ensure we have all the necessary state for the next round
            # This method should check that all required state variables are present 
            # and update the database with any new information
            msg_state = self._ensure_state_for_next_round(msg_state, persuasion_type)
            
            return msg_state
            
        except Exception as e:
            print(f"⚠️ ERROR IN DEBATE MESSAGE WORKER: {str(e)}")
            traceback.print_exc()
            
            # Provide a fallback response
            msg_state["assistant_response"] = "I apologize, but I encountered an issue while preparing my response. Could you please repeat your position or perhaps try a different approach?"
            return msg_state

    def _ensure_state_for_next_round(self, msg_state: MessageState, persuasion_type: str) -> MessageState:
        """Ensure all necessary state is present for the next round of debate."""
        print(f"🔄 Ensuring state variables for next round")
        
        # Ensure metadata and global state are initialized
        if "metadata" not in msg_state:
            msg_state["metadata"] = {}
        if "global_state" not in msg_state["metadata"]:
            msg_state["metadata"]["global_state"] = {}
        
        # Make sure current_persuasion_type is set for the next round
        msg_state["current_persuasion_type"] = persuasion_type
        msg_state["metadata"]["global_state"]["current_persuasion_type"] = persuasion_type
        print(f"🌐 SET current_persuasion_type for next round: {persuasion_type}")
        
        # We don't want to overwrite counter_persuasion_type here, as that's determined by 
        # the EffectivenessEvaluator after the user responds
        
        return msg_state 
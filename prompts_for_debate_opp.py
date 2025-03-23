"""
Prompts for the debate opponent chatbot generator.
This module contains the prompts used to generate a debate opponent chatbot.
"""

debate_generate_tasks_sys_prompt = """You are generating tasks for a debate opponent chatbot. 

Your task is to generate detailed tasks that a debate opponent chatbot would need to perform. The tasks should represent a sophisticated debate flow that explicitly tracks persuasion techniques (logos, pathos, ethos) and their effectiveness, with a complete read-evaluate-update cycle for continual improvement.

CRITICAL REQUIREMENTS FOR TASK GENERATION:
1. Each task MUST be highly specific with clear worker responsibilities and state variable references
2. The initial tasks must focus on WAITING for the user to provide both a topic AND their position
3. Counter-arguments should ONLY be generated AFTER the user has shared their position
4. STRICT FLOW CONTROL:
   - Include EXPLICIT tasks to check if user has provided their stance
   - If stance is not provided, ONLY respond with a request for stance - NO counter-arguments
   - Initialize the database with default values ONLY AFTER stance is confirmed
   - Set "stance_confirmed" state variable to true once user position is established
   - Reference "stance_confirmed" in conditional logic for subsequent tasks
5. For each debate interaction, explicitly include the COMPLETE cycle:
   - Identify the user's persuasion strategy (logos, pathos, ethos) with ArgumentClassifier
   - GENERATE counter-arguments using PersuasionWorker with the SAME persuasion type as the user
   - PRESENT arguments to user with MessageWorker
   - EVALUATE effectiveness with EffectivenessEvaluator after user response
   - UPDATE DebateDatabaseWorker with effectiveness scores
6. Explicitly reference state variables in each task description:
   - stance_confirmed
   - user_persuasion_type (the strategy the user is using)
   - current_persuasion_type (the strategy the bot is currently using)
   - evaluated_effectiveness_score (how effective the bot's strategy was)
7. IMPORTANT STRATEGY CHANGE: The bot should MATCH the user's persuasion technique, not try different ones
   - If user uses logos, bot should respond with logos
   - If user uses pathos, bot should respond with pathos
   - If user uses ethos, bot should respond with ethos
8. Include tasks for tracking which persuasion technique the user employs, and adapting to match it
9. TOPIC SELECTION HANDLING:
   - Create a dedicated BOT_CHOOSE_TOPIC intent that activates when user asks the bot to pick a topic
   - Connect this intent directly to a DebateRAGWorker node that selects an interesting debate topic
   - Ensure proper flow back to the main debate cycle after topic selection
   - Include sample utterances like "you can choose", "you decide", "pick a topic", etc.
   - This path should bypass the normal stance confirmation flow and present a topic + request stance

FORMAT REQUIREMENTS:
1. Format as a simple array of JSON objects
2. Each object must have ONLY "intent" and "task" properties
3. No explanations, formatting notes, or wrapper objects like "tasks"
4. Your response must be ONLY the array of tasks, nothing else
5. CRITICAL: Ensure all JSON objects are properly closed with closing braces and all strings have proper closing quotation marks

EXAMPLE TASK STRUCTURES (in valid JSON format):
[
  {{
    "intent": "CLASSIFY_USER_ARGUMENT", 
    "task": "Use ArgumentClassifier to analyze user's response and identify whether they're using logos (logical), pathos (emotional), or ethos (ethical) persuasion techniques. Store result in user_persuasion_type state variable."
  }},
  {{
    "intent": "EVALUATE_EFFECTIVENESS", 
    "task": "Using EffectivenessEvaluator, analyze how effective the bot's current_persuasion_type strategy was based on the user's response. Calculate an effectiveness score (0-100) and store in evaluated_effectiveness_score variable."
  }},
  {{
    "intent": "UPDATE_DATABASE", 
    "task": "Update DebateDatabaseWorker with the evaluated_effectiveness_score for the current_persuasion_type. Then set counter_persuasion_type to MATCH the user_persuasion_type for the next response, implementing a strategy of responding with the same type of argument the user employs."
  }},
  {{
    "intent": "BOT_CHOOSE_TOPIC",
    "task": "When the user requests the bot to choose a topic (with phrases like 'you choose' or 'pick a topic'), use DebateRAGWorker to select an engaging debate topic with clear opposing perspectives, then present it to the user and ask for their stance."
  }}
]

Builder's prompt: {user_objective}
"""

debate_check_best_practice_sys_prompt = """You are a useful assistant to detect if the current debate task needs to be further decomposed if it cannot be solved by the provided resources. Based on the task and the current node level of the task on the tree, please output Yes if it needs to be decomposed; No otherwise. Remember that the debate opponent should always take an adversarial position opposite to the user's stance. Please also provide explanations for your choice.

CRITICAL REQUIREMENTS: Every debate interaction MUST include all of the following:
1. ArgumentClassifier to identify the user's persuasion type (logos, pathos, ethos)
2. PersuasionWorker to generate counter-arguments MATCHING the user's persuasion type
3. Evaluation of counter-arguments using EffectivenessEvaluator AFTER every user response
4. Database updates using DebateDatabaseWorker AFTER every user response to track effectiveness
5. A dedicated path for BOT-INITIATED TOPIC SELECTION using DebateRAGWorker when user requests it

STATE PASSING REQUIREMENTS: All persuasion state variables must be tracked between workers:
1. ArgumentClassifier must set "user_persuasion_type" in global state
2. EffectivenessEvaluator must set "counter_persuasion_type" to match "user_persuasion_type" in global state
3. PersuasionWorker must set "current_persuasion_type" in global state
4. All task graph nodes must include appropriate operations and state variables in their attribute fields
5. Global state must be used to pass variables between workers consistently
6. After bot-selected topics, the "stance_confirmed" variable must be set properly once user responds

Task: The current task is {task}. The current node level of the task is {level}.
Resources: {resources}
Reasoning:
"""

debate_generate_best_practice_sys_prompt = """Given the task for the debate opponent chatbot, your goal is to create a sophisticated, step-by-step implementation plan with precise worker assignments, state tracking, and persuasion technique management.

CRITICAL WORKFLOW REQUIREMENTS:
1. Create a DETAILED step-by-step workflow that maps EXACTLY how each worker will be used
2. For EVERY user interaction, include this COMPLETE cycle:
   - CLASSIFY: ArgumentClassifier must identify user's persuasion technique (logos, pathos, ethos)
   - EVALUATE: EffectivenessEvaluator must measure how effective the bot's previous response was
   - UPDATE: DebateDatabaseWorker must store effectiveness scores and set the next strategy to MATCH user's technique
   - READ: DebateDatabaseWorker must confirm the persuasion technique to use for the next response
   - GENERATE: PersuasionWorker must create counter-arguments using the same technique as the user
   - PRESENT: MessageWorker must deliver arguments to user in an adversarial manner
3. BOT-INITIATED TOPIC SELECTION:
   - Implement a clear BOT_CHOOSE_TOPIC intent that recognizes phrases like "you choose" or "pick a topic"
   - Connect directly to DebateRAGWorker to access knowledge base and select engaging debate topics
   - Select balanced topics with clear opposing viewpoints from the bot's knowledge base
   - Present the selected topic to the user with brief context and request their position
   - Resume normal debate flow with ArgumentClassifier after user responds to the suggested topic

STATE PASSING REQUIREMENTS:
1. Each step must EXPLICITLY reference state variables with these naming conventions:
   - user_persuasion_type: Set by ArgumentClassifier, read by EffectivenessEvaluator
   - current_persuasion_type: The technique the bot is currently using, set by PersuasionWorker
   - counter_persuasion_type: The technique the bot will use next, set to match user_persuasion_type
   - evaluated_effectiveness_score: How effective the bot's last strategy was
   - stance_confirmed: Boolean indicating user has stated a clear position (set after topic selection)

2. Each step must clearly include:
   - Exact worker to use
   - Input state variables (what it reads)
   - Output state variables (what it sets)
   - Specific description of what it does
   - How it handles persuasion technique information

3. EFFECTIVENESS EVALUATION AND ADAPTATION:
   After each user response, include these critical steps:
   - ArgumentClassifier identifies which technique (logos, pathos, ethos) the user is using
   - EffectivenessEvaluator measures how effective the bot's PREVIOUS strategy was
   - EffectivenessEvaluator sets counter_persuasion_type to MATCH user_persuasion_type
   - DebateDatabaseWorker stores the effectiveness score for the previous strategy
   - DebateDatabaseWorker confirms the next strategy (matching the user's technique)
   - PersuasionWorker generates a counter-argument using the same technique as the user

4. ANSWER STEP IMPLEMENTATION:
   Include a dedicated Answer step that explains:
   - Which persuasion technique the user employed in their last response
   - How effective the bot's previous strategy was
   - Why the bot is now using the same technique as the user
   - Example Answer: "User employed ethos-based arguments (ethical appeals). The bot's previous logos-based strategy had 58% effectiveness. The bot will now respond using ethos-based counter-arguments to match the user's persuasion style."

5. Ensure unbroken chain of state passing between workers with no information loss
6. Use global_state consistently to pass variables between workers

RESPONSE FORMAT:
Your response should be structured as a detailed JSON array where each object includes:
- step: Sequential number
- worker: Specific worker to use
- description: Detailed explanation of the step
- state_read: Array of state variables this step reads
- state_set: Array of state variables this step sets
- persuasion_handling: How this step manages persuasion technique information

Background: The builder want to create a chatbot - {role}. {user_objective}
Task: {task}
Resources: {resources}
"""

debate_embed_resources_sys_prompt = """Map the best practice steps to specific resources to create a detailed, robust task graph for the debate opponent chatbot. Your goal is to ensure precise resource allocation, complete state tracking, and sophisticated persuasion technique management throughout the debate flow.

CRITICAL WORKFLOW REQUIREMENTS:
1. Create a comprehensive resource mapping where each step is assigned to the EXACT worker that should execute it
2. STANCE VERIFICATION AND DATABASE INITIALIZATION:
   - Include explicit MessageWorker steps that check for user stance BEFORE any counter-arguments
   - Add conditional logic that ONLY proceeds to counter-arguments if stance_confirmed=true
   - Create a DebateDatabaseWorker initialization step that ONLY executes after stance confirmation
   - Initialize default effectiveness scores for all persuasion types (logos, pathos, ethos)
3. For EVERY debate cycle, include this PRECISE sequence with PROPER resource allocation:
   - ArgumentClassifier to identify user's persuasion strategy
   - EffectivenessEvaluator to measure bot's previous effectiveness
   - DebateDatabaseWorker (update) to store scores and set next strategy
   - DebateDatabaseWorker (read) to retrieve appropriate counter strategy
   - PersuasionWorker to generate counter-arguments using specified technique
   - MessageWorker to deliver the counter-argument
4. TOPIC SELECTION RESOURCE MAPPING:
   - Map BOT_CHOOSE_TOPIC intent directly to DebateRAGWorker for topic selection
   - Assign a high weight (5+) to this intent to prioritize it over generic responses
   - Include comprehensive sample utterances: ["you choose", "you decide", "pick a topic", "choose for me", etc.]
   - Create direct transition from DebateRAGWorker back to ArgumentClassifier
   - Ensure this path bypasses normal stance checking and immediately asks for user's position on selected topic

5. EFFECTIVENESS EVALUATION AND ADAPTATION:
   - When embedding the effectiveness evaluation nodes, ENSURE:
     * EffectivenessEvaluator calculates a numerical score for the bot's prior strategy
     * DebateDatabaseWorker updates this score for historical tracking
     * ArgumentClassifier correctly identifies user's current strategy
     * EffectivenessEvaluator has precise task description setting counter_persuasion_type to MATCH user_persuasion_type
     * PersuasionWorker's task explicitly references using the same persuasion type as the user
     * MessageWorker includes the persuasion type information in its response

6. NODE-TO-NODE CONNECTIONS:
   - Verify that every node has clear outgoing edges for ALL possible user responses
   - Annotate edges with appropriate intents and weights
   - Double-check that initial stance checking edges work correctly
   - Ensure that topic selection has a clear bridge back to main debate flow
   - Verify that the debate cycle has proper circular connections for continued interaction

WORKER INITIALIZATION REQUIREMENTS:
1. Configure DebateRAGWorker with diverse, balanced debate topics
2. Ensure DebateMessageWorker has clear instructions on formatting persuasion types
3. Provide ArgumentClassifier with guidelines for detecting logos, pathos, and ethos
4. Set up EffectivenessEvaluator with criteria for measuring argument impact
5. Initialize DebateDatabaseWorker with proper effectiveness tracking

Resources: {resources}
Best Practice: {best_practice}
"""

debate_generate_start_msg = """The builder plans to create a debate opponent chatbot. Given the role of the chatbot, your task is to generate an engaging starting message that sets the tone for a spirited debate. The starting message should NOT include any counter-arguments or positions yet! It should ONLY:

1. Welcome the user to the debate
2. CLEARLY ask the user to do two things:
   a. Choose a specific debate topic 
   b. Explicitly state THEIR position on that topic
3. Explain that the bot will take the opposing stance AFTER the user has shared their position

The bot should NOT make any assumptions about the user's stance or present counter-arguments in this initial message.

The message should also communicate that:
1. The chatbot will analyze their arguments and adapt its persuasion approach
2. The chatbot tracks which counter-argument types are most effective
3. The debate experience becomes more challenging as the system learns

Return the response in JSON format.

For Example:
Builder's prompt: The builder wants to create a chatbot - Debate Opponent. The debate opponent actively takes opposing positions to the user's stance, vigorously challenging their viewpoints with strong counter-arguments.
Start Message:
```json
{{
    "message": "Welcome to our Debate Arena! I'm your opponent, ready to challenge your viewpoints and help sharpen your arguments. To get started, please:\n\n1. Choose a specific debate topic (e.g., social media, climate policy, AI regulation)\n2. Clearly state YOUR position on that topic\n\nFor example: 'I believe social media has a positive impact on society because it connects people and spreads information.'\n\nOnce you've stated your position, I'll take the opposite stance and challenge your reasoning. As we debate, I'll adapt my approach based on what's most effective at testing your arguments. Let's begin!"
}}
```

Builder's prompt: The builder want to create a chatbot - {role}. {user_objective}
Start Message:
""" 

# Argument classifier prompt to analyze user messages into argument types
argument_classifier_prompt = """Analyze the following message and classify it as emotional (pathos), logical (logos), or ethical (ethos).

User message: {message}

Respond with a valid JSON object in this format:
{{
    "dominant_type": "emotional|logical|ethical",
    "secondary_types": ["list other types that apply"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of classification",
    "persuasion_indicators": {{
        "emotional": 0.0-1.0,
        "logical": 0.0-1.0,
        "ethical": 0.0-1.0
    }}
}}"""

# Add a new prompt for updating worker files with proper logging

debate_worker_logging_template = """
# Worker Header Template - Add to each worker file

def execute(self, state: MessageState) -> MessageState:
    \"\"\"Worker's primary execution function.\"\"\"
    # Clear worker identification in logs
    print("\\n================================================================================")
    print(f"🔄 {self.__class__.__name__.upper()} EXECUTING")
    print(f"================================================================================\\n")
    
    # Worker-specific processing here
    
    # Add completion log
    print(f"\\n================================================================================")
    print(f"✅ {self.__class__.__name__.upper()} COMPLETED")
    print(f"================================================================================\\n")
    
    return state
""" 
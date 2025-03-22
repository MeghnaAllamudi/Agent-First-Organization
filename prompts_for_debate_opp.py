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
   - READ from DebateDatabaseWorker to determine most effective persuasion technique
   - GENERATE counter-arguments using PersuasionWorker with specific persuasion type
   - PRESENT arguments to user with MessageWorker
   - EVALUATE effectiveness with EffectivenessEvaluator after user response
   - UPDATE DebateDatabaseWorker with effectiveness scores
6. Explicitly reference state variables in each task description:
   - stance_confirmed
   - best_persuasion_type
   - current_persuasion_type
   - evaluated_effectiveness_score
7. Each task must clearly specify which persuasion technique (logos, pathos, ethos) is being used
8. Include tasks for adapting strategy based on which persuasion technique is most effective

FORMAT REQUIREMENTS:
1. Format as a simple array of JSON objects
2. Each object must have ONLY "intent" and "task" properties
3. No explanations, formatting notes, or wrapper objects like "tasks"
4. Your response must be ONLY the array of tasks, nothing else
5. CRITICAL: Ensure all JSON objects are properly closed with closing braces and all strings have proper closing quotation marks

EXAMPLE TASK STRUCTURES (in valid JSON format):
[
  {{
    "intent": "READ_DATABASE", 
    "task": "Read database using DebateDatabaseWorker to determine most effective persuasion technique (logos, pathos, or ethos) based on historical effectiveness scores, then store result in best_persuasion_type state variable"
  }},
  {{
    "intent": "EVALUATE_EFFECTIVENESS", 
    "task": "Using EffectivenessEvaluator, analyze user response to the current counter-argument (using current_persuasion_type) to determine how persuasive it was. Calculate an effectiveness score (0-100) based on indicators such as user engagement, concessions made, or defensive reactions. Store result in evaluated_effectiveness_score state variable."
  }},
  {{
    "intent": "UPDATE_DATABASE", 
    "task": "Update DebateDatabaseWorker with the evaluated_effectiveness_score for the current_persuasion_type. Record this score in the appropriate persuasion type's historical effectiveness tracking. Then query database to identify which persuasion type now has the highest cumulative effectiveness score and update best_persuasion_type for next counter-argument generation."
  }}
]

Builder's prompt: {user_objective}
"""

debate_check_best_practice_sys_prompt = """You are a useful assistant to detect if the current debate task needs to be further decomposed if it cannot be solved by the provided resources. Based on the task and the current node level of the task on the tree, please output Yes if it needs to be decomposed; No otherwise. Remember that the debate opponent should always take an adversarial position opposite to the user's stance. Please also provide explanations for your choice.

CRITICAL REQUIREMENTS: Every debate interaction MUST include all of the following:
1. Database reads using DebateDatabaseWorker BEFORE generating counter-arguments to determine most effective persuasion techniques
2. Evaluation of counter-arguments using EffectivenessEvaluator AFTER every user response
3. Database updates using DebateDatabaseWorker AFTER every user response to track effectiveness

STATE PASSING REQUIREMENTS: All persuasion state variables must be tracked between workers:
1. DebateDatabaseWorker must set "best_persuasion_type" in state to be read by PersuasionWorker
2. PersuasionWorker must set "current_persuasion_type" in state to be read by EffectivenessEvaluator
3. All task graph nodes must include "persuasion_type" in their attribute fields
4. The state variables must be explicitly passed through the task graph using variable substitution

Task: The current task is {task}. The current node level of the task is {level}.
Resources: {resources}
Reasoning:
"""

debate_generate_best_practice_sys_prompt = """Given the task for the debate opponent chatbot, your goal is to create a sophisticated, step-by-step implementation plan with precise worker assignments, state tracking, and persuasion technique management.

CRITICAL WORKFLOW REQUIREMENTS:
1. Create a DETAILED step-by-step workflow that maps EXACTLY how each worker will be used
2. For EVERY user interaction, include this COMPLETE cycle:
   - READ: DebateDatabaseWorker must query most effective persuasion technique BEFORE generating arguments
   - GENERATE: PersuasionWorker must create counter-arguments using the recommended technique
   - PRESENT: MessageWorker must deliver arguments to user in an adversarial manner
   - EVALUATE: EffectivenessEvaluator must measure persuasiveness after receiving user response
   - UPDATE: DebateDatabaseWorker must store effectiveness scores to improve future technique selection

STATE PASSING REQUIREMENTS:
1. Each step must EXPLICITLY reference state variables with these naming conventions:
   - best_persuasion_type: Set by DebateDatabaseWorker, read by PersuasionWorker
   - current_persuasion_type: Set by PersuasionWorker, read by MessageWorker and EffectivenessEvaluator
   - evaluated_effectiveness_score: Set by EffectivenessEvaluator, read by DebateDatabaseWorker
   - [persuasion_type]_persuasion_response: Store generated responses for each technique type
   - persuasion_effectiveness_scores: Dictionary tracking cumulative scores for each type

2. Each step must clearly include:
   - Exact worker to use
   - Input state variables (what it reads)
   - Output state variables (what it sets)
   - Specific description of what it does
   - How it handles persuasion technique information

3. EFFECTIVENESS SCORE CALCULATION AND ADAPTATION:
   After each user response, include these critical steps:
   - EffectivenessEvaluator analyzes user response against current_persuasion_type
   - Calculates effectiveness_score (0-100) based on specific metrics:
     * Degree of engagement (did user directly address the argument?)
     * Signs of concession (did user modify their position slightly?)
     * Defensive reactions (did user become more entrenched?)
     * Length and depth of response (did argument provoke substantial thought?)
   - DebateDatabaseWorker updates history for current_persuasion_type with new score
   - DebateDatabaseWorker recalculates which technique has highest cumulative score
   - DebateDatabaseWorker sets best_persuasion_type for next counter-argument

4. ANSWER STEP IMPLEMENTATION:
   Include a dedicated Answer step that:
   - Reads current effectiveness scores for all persuasion types
   - Analyzes which technique has been most effective
   - Explicitly selects the highest-scoring persuasion type for next counter-argument
   - Sets next_persuasion_type state variable
   - Provides clear reasoning about why this technique is most effective
   - Example Answer: "Based on effectiveness metrics, pathos-based arguments (78%) have been most effective at engaging this user compared to logos (65%) and ethos (45%). The user shows more willingness to consider emotional appeals over logical arguments. Next counter-argument will use pathos persuasion technique."

5. Ensure unbroken chain of state passing between workers with no information loss

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
   - INITIAL QUERY: DebateDatabaseWorker MUST be used to determine most effective persuasion technique BEFORE any counter-argument generation
   - GENERATION: PersuasionWorker MUST create the counter-arguments using the specific persuasion type recommended by the database
   - PRESENTATION: MessageWorker MUST deliver these arguments to the user in an adversarial manner
   - EVALUATION: EffectivenessEvaluator MUST measure persuasiveness after user responds
   - DATABASE UPDATE: DebateDatabaseWorker MUST store effectiveness scores for continuous improvement
   - ANSWER: Include reasoning about effectiveness analysis and technique selection

4. EFFECTIVENESS EVALUATION AND ADAPTATION:
   Include these DETAILED steps with SPECIFIC worker configurations:
   - EffectivenessEvaluator step:
     * Must analyze FULL user response text
     * Must explicitly reference current_persuasion_type
     * Must calculate numeric score (0-100) using multiple metrics
     * Must include specific examples of what indicates high/low effectiveness
     * Must set evaluated_effectiveness_score state variable
   
   - DebateDatabaseWorker step (specifically for updating scores):
     * Must read evaluated_effectiveness_score
     * Must update historical record for current_persuasion_type
     * Must recalculate which persuasion type has highest cumulative score
     * Must update best_persuasion_type for next counter-argument
     * Must include example SQL-like operations that would be performed
     
   - Answer step (specifically for effectiveness analysis):
     * Must read all persuasion_effectiveness_scores
     * Must analyze which technique has been most effective with this specific user
     * Must explicitly select highest-scoring persuasion type for next counter-argument 
     * Must provide reasoning about why this technique is most effective with this user
     * Must link effectiveness back to specific user response patterns

ATTRIBUTE STRUCTURE REQUIREMENTS:
1. EVERY step MUST include a detailed attribute object with these properties:
   - task: Clear description of what this step accomplishes
   - value: Example output this step would produce (be specific and realistic)
   - directed: Usually false unless connections need special handling
   - persuasion_type: Reference to the appropriate state variable or literal initial value
   - stance_check: For stance verification steps, set to true

2. WORKER-SPECIFIC ATTRIBUTE REQUIREMENTS:
   - MessageWorker (stance check):
     * operation: "stance_verification"
     * stance_confirmed: false (initial value)
     * action: "request_stance" or "proceed_to_debate"
     
   - DebateDatabaseWorker (initialize):
     * operation: "initialize"
     * requires_stance_confirmation: true
     * initial_values: Default effectiveness scores
     
   - DebateDatabaseWorker (read): 
     * operation: "read"
     * persuasion_type: "logos" (initial) or state reference 
     * purpose: "determine_best_technique"
     * requires_stance_confirmation: true
     
   - PersuasionWorker:
     * persuasion_type: Reference to best_persuasion_type state variable
     * counter_argument_type: Specific description of counter approach
     * adaptation_level: How this adapts based on prior effectiveness
     * requires_stance_confirmation: true
     
   - MessageWorker (debate):
     * persuasion_type: Reference to current_persuasion_type
     * delivery_style: "adversarial"
     * tone: Matches the persuasion type (logical, emotional, authoritative)
     * requires_stance_confirmation: true
     
   - EffectivenessEvaluator:
     * persuasion_type: Reference to current_persuasion_type
     * effectiveness_tracking: true
     * metrics: Array of evaluation criteria ["engagement_level", "concessions_made", "defensive_reactions", "response_depth"]
     * score_calculation: "weighted_average"
     * requires_stance_confirmation: true
     
   - DebateDatabaseWorker (update):
     * operation: "update"
     * persuasion_type: Reference to current_persuasion_type
     * score: Reference to evaluated_effectiveness_score
     * improvement_purpose: "refine_future_technique_selection"
     * update_operations: ["record_score", "recalculate_best_technique", "update_best_type"]
     * requires_stance_confirmation: true
     
   - Answer (effectiveness analysis):
     * operation: "technique_selection_reasoning"
     * all_persuasion_scores: Reference to persuasion_effectiveness_scores
     * selected_technique: Reference to best_persuasion_type
     * reasoning_depth: "detailed"
     * adaptation_strategy: "responsive_to_user_patterns"
     * requires_stance_confirmation: true

3. RESPONSE STRUCTURE:
Include for each step:
   - step: Number
   - task: Detailed description
   - resource: Worker name
   - resource_id: Unique ID
   - example_response: Realistic example of what this worker would output
   - attribute: Complete attribute object with ALL required fields for that worker type
   - conditional_execution: For steps that depend on stance confirmation, specify condition

EXAMPLE EFFECTIVENESS EVALUATION STEP:
```
{{
  "step": 5,
  "task": "Evaluate effectiveness of logos-based counter-argument based on user's response",
  "resource": "EffectivenessEvaluator",
  "resource_id": "effectiveness_evaluator_1",
  "example_response": "Effectiveness analysis complete. The logos-based argument received a score of 75/100 based on: high engagement (user directly addressed logical points), moderate concessions (user acknowledged some factual points), low defensive reactions (user remained calm and rational), high response depth (user provided detailed counter-examples).",
  "attribute": {{
    "task": "Evaluate persuasiveness of current counter-argument",
    "value": "Effectiveness score: 75/100",
    "directed": false,
    "persuasion_type": "$current_persuasion_type",
    "effectiveness_tracking": true,
    "metrics": ["engagement_level", "concessions_made", "defensive_reactions", "response_depth"],
    "score_calculation": "weighted_average",
    "requires_stance_confirmation": true
  }},
  "conditional_execution": "stance_confirmed == true"
}}
```

EXAMPLE DATABASE UPDATE STEP:
```
{{
  "step": 6,
  "task": "Update database with effectiveness score for logos persuasion and determine most effective technique for next argument",
  "resource": "DebateDatabaseWorker",
  "resource_id": "debate_database_worker_3",
  "example_response": "Database updated with effectiveness score 75 for logos persuasion type. Recalculated cumulative scores: logos (75), pathos (50), ethos (40). Best technique for next argument: logos.",
  "attribute": {{
    "task": "Update persuasion type effectiveness scores",
    "value": "Updated scores: logos (75), pathos (50), ethos (40). Best: logos",
    "directed": false,
    "operation": "update",
    "persuasion_type": "$current_persuasion_type",
    "score": "$evaluated_effectiveness_score",
    "improvement_purpose": "refine_future_technique_selection",
    "update_operations": ["record_score", "recalculate_best_technique", "update_best_type"],
    "requires_stance_confirmation": true
  }},
  "conditional_execution": "stance_confirmed == true"
}}
```

EXAMPLE ANSWER STEP:
```
{{
  "step": 7,
  "task": "Analyze effectiveness patterns and provide reasoning for selected persuasion technique",
  "resource": "Answer",
  "resource_id": "answer_1",
  "example_response": "Effectiveness Analysis: Based on user response patterns, logos-based arguments have been most effective (75% effective) compared to pathos (50%) and ethos (40%). The user consistently engages more deeply with logical points, offers counterexamples, and shows willingness to consider factual evidence. Their response length and depth increases significantly when presented with data-driven arguments. Next counter-argument will continue using logos as the primary persuasion technique, with focus on statistical evidence and causal relationships.",
  "attribute": {{
    "task": "Provide reasoning for persuasion technique selection",
    "value": "Selected technique: logos (75% effective) - User shows stronger engagement with logical arguments",
    "directed": false,
    "operation": "technique_selection_reasoning",
    "all_persuasion_scores": "$persuasion_effectiveness_scores",
    "selected_technique": "$best_persuasion_type",
    "reasoning_depth": "detailed",
    "adaptation_strategy": "responsive_to_user_patterns",
    "requires_stance_confirmation": true
  }},
  "conditional_execution": "stance_confirmed == true"
}}
```

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
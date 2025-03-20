"""
Prompts for the debate opponent chatbot generator.
This module contains the prompts used to generate a debate opponent chatbot.
"""

debate_generate_tasks_sys_prompt = """The builder plans to create a debate opponent chatbot. Given the role of the chatbot, along with any introductory information and detailed documentation (if available), your task is to identify the specific, distinct tasks that a debate opponent should handle based on the user's intent. These tasks should not overlap or depend on each other and must address different aspects of debate interaction. Return the response in JSON format.

For Example:
Builder's prompt: The builder wants to create a chatbot - Debate Opponent. The debate opponent engages users in spirited debates by aggressively challenging their viewpoints and providing strong counter-arguments.
Builder's Information: The debate opponent should be able to identify argument types, analyze debate history, generate persuasive counter-arguments, and maintain an engaging debate flow.

Reasoning Process:
Thought 1: Understand the core responsibilities of a debate opponent.
Observation 1: A debate opponent needs to analyze arguments, identify weaknesses, generate counter-arguments, and maintain an engaging debate flow.

Thought 2: Based on these responsibilities, identify specific tasks that don't overlap.
Observation 2: The tasks should cover argument analysis, persuasion strategy selection, counter-argument generation, and debate flow management.

Answer:
```json
[
    {{
        "intent": "User presents an argument to debate",
        "task": "Analyze and counter the user's argument"
    }},
    {{
        "intent": "User challenges the bot's position",
        "task": "Defend position and strengthen counter-arguments"
    }},
    {{
        "intent": "User wants to explore a new debate topic",
        "task": "Initialize debate on new topic"
    }},
    {{
        "intent": "User wants to conclude the debate",
        "task": "Summarize debate points and conclude discussion"
    }}
]
```

Builder's prompt: The builder want to create a chatbot - {role}. {u_objective}
Builder's information: {intro}
Builder's documentations: 
{docs}
Reasoning Process:
"""

debate_check_best_practice_sys_prompt = """You are a useful assistant to detect if the current debate task needs to be further decomposed if it cannot be solved by the provided resources. Based on the task and the current node level of the task on the tree, please output Yes if it needs to be decomposed; No otherwise. Please also provide explanations for your choice.

Here are some examples:
Task: The current task is Analyze and counter the user's argument. The current node level of the task is 1.
Resources:
ArgumentClassifier: Classifies arguments into emotional, logical, or ethical categories
DebateHistoryAnalyzer: Analyzes past debate performance to recommend persuasion types
PersuasionWorker: Generates counter-arguments using appropriate persuasion techniques
MessageWorker: The worker responsible for interacting with the user with predefined responses

Reasoning: This task involves multiple steps: analyzing the argument type, determining the best counter-strategy, generating a counter-argument, and delivering it effectively. Each step requires a different resource. Therefore, it needs to be decomposed into smaller sub-tasks.
Answer:
```json
{{
    "answer": "Yes"
}}
```

Task: The current task is Present final counter-argument to user. The current node level of the task is 2.
Resources:
MessageWorker: The worker responsible for interacting with the user with predefined responses
PersuasionWorker: Generates counter-arguments using appropriate persuasion techniques

Reasoning: This is a specific task that can be handled by the MessageWorker resource. It's a single interaction point that doesn't require further decomposition.
Answer:
```json
{{
    "answer": "No"
}}
```

Task: The current task is {task}. The current node level of the task is {level}.
Resources: {resources}
Reasoning:
"""

debate_generate_best_practice_sys_prompt = """Given the background information about the debate opponent chatbot, the task it needs to handle, and the available resources, your task is to generate a step-by-step best practice for addressing this task. Each step should represent a distinct interaction in the debate flow. Return the answer in JSON format.

For example:
Background: The builder wants to create a chatbot - Debate Opponent. The debate opponent engages users in spirited debates by aggressively challenging their viewpoints and providing strong counter-arguments.

Task: Analyze and counter the user's argument

Resources:
ArgumentClassifier: Classifies arguments into emotional, logical, or ethical categories
DebateHistoryAnalyzer: Analyzes past debate performance to recommend persuasion types
PersuasionWorker: Generates counter-arguments using appropriate persuasion techniques
MessageWorker: The worker responsible for interacting with the user with predefined responses

Thought: To effectively counter a user's argument, the bot should first analyze the argument type and identify weaknesses. Then, it should determine the most effective persuasion strategy based on debate history. Next, it should generate a strong counter-argument using appropriate persuasion techniques. Finally, it should present the counter-argument in a challenging but engaging way.

Answer:
```json
[
    {{
        "step": 1,
        "task": "Analyze the type and weaknesses of the user's argument"
    }},
    {{
        "step": 2,
        "task": "Determine the most effective persuasion strategy based on debate history"
    }},
    {{
        "step": 3,
        "task": "Generate a strong counter-argument using appropriate techniques"
    }},
    {{
        "step": 4,
        "task": "Present the counter-argument in a challenging way"
    }}
]
```

Background: The builder want to create a chatbot - {role}. {u_objective}
Task: {task}
Resources: {resources}
Thought:
"""

debate_embed_resources_sys_prompt = """Given the best practice steps and available resources for the debate opponent, map each step to the appropriate resource and ensure the steps are properly connected. Each step MUST include an example_response field that shows what the resource would output. Return the answer in JSON format.

For example:
Best Practice:
[
    {{
        "step": 1,
        "task": "Analyze the type and weaknesses of the user's argument",
        "resource": "ArgumentClassifier",
        "example_response": "I've analyzed your argument and identified it as primarily emotional, based on personal anecdotes rather than empirical evidence. Let me challenge this perspective.",
        "resource_id": "argument_classifier_id"
    }},
    {{
        "step": 2,
        "task": "Determine the most effective persuasion strategy based on debate history",
        "resource": "DebateHistoryAnalyzer",
        "example_response": "Based on our debate history, presenting statistical evidence has been most effective in countering emotional arguments. I'll focus on data-driven counter-arguments.",
        "resource_id": "debate_history_analyzer_id"
    }},
    {{
        "step": 3,
        "task": "Generate a strong counter-argument using appropriate techniques",
        "resource": "PersuasionWorker",
        "example_response": "While your personal experience is valid, let me present some contradicting statistics: According to recent studies...",
        "resource_id": "persuasion_worker_id"
    }},
    {{
        "step": 4,
        "task": "Present the counter-argument in a challenging way",
        "resource": "MessageWorker",
        "example_response": "I understand your emotional connection to this issue, but the data tells a different story. How do you reconcile your personal experience with these broader statistics?",
        "resource_id": "message_worker_id"
    }}
]

Resources: {resources}

Best Practice: {best_practice}

Thought: To properly map the steps to resources and ensure they are connected, we need to:
1. Verify that each step uses an available resource
2. Ensure the steps flow logically from one to another
3. Make sure each step has a descriptive example_response that shows what the resource would output
4. Include resource_id for each step to properly identify the resource
5. Validate that the final step completes the task

Answer:
[
    {{
        "step": 1,
        "task": "Analyze the type and weaknesses of the user's argument",
        "resource": "ArgumentClassifier",
        "example_response": "I've analyzed your argument and identified it as primarily emotional, based on personal anecdotes rather than empirical evidence. Let me challenge this perspective.",
        "resource_id": "argument_classifier_id"
    }},
    {{
        "step": 2,
        "task": "Determine the most effective persuasion strategy based on debate history",
        "resource": "DebateHistoryAnalyzer",
        "example_response": "Based on our debate history, presenting statistical evidence has been most effective in countering emotional arguments. I'll focus on data-driven counter-arguments.",
        "resource_id": "debate_history_analyzer_id"
    }},
    {{
        "step": 3,
        "task": "Generate a strong counter-argument using appropriate techniques",
        "resource": "PersuasionWorker",
        "example_response": "While your personal experience is valid, let me present some contradicting statistics: According to recent studies...",
        "resource_id": "persuasion_worker_id"
    }},
    {{
        "step": 4,
        "task": "Present the counter-argument in a challenging way",
        "resource": "MessageWorker",
        "example_response": "I understand your emotional connection to this issue, but the data tells a different story. How do you reconcile your personal experience with these broader statistics?",
        "resource_id": "message_worker_id"
    }}
]
"""

debate_generate_start_msg = """The builder plans to create a debate opponent chatbot. Given the role of the chatbot, your task is to generate an engaging starting message that sets the tone for a spirited debate. Return the response in JSON format.

For Example:
Builder's prompt: The builder wants to create a chatbot - Debate Opponent. The debate opponent engages users in spirited debates by aggressively challenging their viewpoints and providing strong counter-arguments.
Start Message:
```json
{{
    "message": "Ready to engage in a spirited debate! Present your argument, and I'll challenge your perspective with compelling counter-points. What topic shall we debate today?"
}}
```

Builder's prompt: The builder want to create a chatbot - {role}. {u_objective}
Start Message:
""" 
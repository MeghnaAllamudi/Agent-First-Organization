"""
Prompts for the debate opponent chatbot generator.
This module contains the prompts used to generate a debate opponent chatbot.
"""

debate_generate_tasks_sys_prompt = """The builder plans to create a debate opponent chatbot. Given the role of the chatbot, along with any introductory information and detailed documentation (if available), your task is to identify the specific, distinct tasks that a debate opponent should handle based on the user's intent. These tasks should not overlap or depend on each other and must address different aspects of debate interaction. Remember, the debate opponent should ALWAYS take the opposite stance from the user, not provide balanced perspectives. Return the response in JSON format.

For Example:
Builder's prompt: The builder wants to create a chatbot - Debate Opponent. The debate opponent actively takes opposing positions to the user's stance, vigorously challenging their viewpoints with strong counter-arguments and defending contrary positions.
Builder's Information: The debate opponent should be able to identify the user's argument stance, take the opposite position, generate persuasive counter-arguments, and maintain an adversarial debate flow.

Reasoning Process:
Thought 1: Understand the core responsibilities of a debate opponent that actively challenges the user.
Observation 1: A debate opponent needs to identify the user's position, adopt the contrary stance, generate strong counter-arguments, and maintain an adversarial yet respectful debate flow.

Thought 2: Based on these responsibilities, identify specific tasks that don't overlap.
Observation 2: The tasks should cover identifying the user's stance, adopting the contrary position, generating strong opposing arguments, and maintaining confrontational debate flow.

Answer:
```json
[
    {{
        "intent": "User presents an argument to debate",
        "task": "Identify user's stance and present opposing arguments"
    }},
    {{
        "intent": "User defends their position",
        "task": "Challenge user's defense with stronger counter-arguments"
    }},
    {{
        "intent": "User wants to explore a new debate topic",
        "task": "Determine user's likely stance and immediately take the opposite position"
    }},
    {{
        "intent": "User wants to conclude the debate",
        "task": "Summarize key points of opposition and reinforce the contrary stance"
    }}
]
```

Builder's prompt: The builder want to create a chatbot - {role}. {user_objective}
Builder's information: {intro}
Builder's documentations: 
{docs}
Reasoning Process:
"""

debate_check_best_practice_sys_prompt = """You are a useful assistant to detect if the current debate task needs to be further decomposed if it cannot be solved by the provided resources. Based on the task and the current node level of the task on the tree, please output Yes if it needs to be decomposed; No otherwise. Remember that the debate opponent should always take an adversarial position opposite to the user's stance. Please also provide explanations for your choice.

Here are some examples:
Task: The current task is Identify user's stance and present opposing arguments. The current node level of the task is 1.
Resources:
ArgumentClassifier: Classifies arguments into emotional, logical, or ethical categories
DebateHistoryAnalyzer: Analyzes past debate performance to recommend persuasion types
PersuasionWorker: Generates counter-arguments using appropriate persuasion techniques
MessageWorker: The worker responsible for interacting with the user with predefined responses

Reasoning: This task involves multiple steps: analyzing the user's argument to identify their stance, determining the most effective counter-position, generating strong opposing arguments, and delivering them in a confrontational manner. Each step requires a different resource. Therefore, it needs to be decomposed into smaller sub-tasks.
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

Reasoning: This is a specific task that can be handled by combining the PersuasionWorker to generate a strong opposing argument and the MessageWorker to deliver it in an adversarial manner. It's a single interaction point that doesn't require further decomposition.
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

debate_generate_best_practice_sys_prompt = """Given the background information about the debate opponent chatbot, the task it needs to handle, and the available resources, your task is to generate a step-by-step best practice for addressing this task. Each step should represent a distinct interaction in the debate flow. Remember that the debate opponent should ALWAYS adopt a position opposite to the user's stance and actively challenge their arguments. Return the answer in JSON format.

For example:
Background: The builder wants to create a chatbot - Debate Opponent. The debate opponent actively takes opposing positions to the user's stance, vigorously challenging their viewpoints with strong counter-arguments.

Task: Identify user's stance and present opposing arguments

Resources:
ArgumentClassifier: Classifies arguments into emotional, logical, or ethical categories
DebateHistoryAnalyzer: Analyzes past debate performance to recommend persuasion types
PersuasionWorker: Generates counter-arguments using appropriate persuasion techniques
MessageWorker: The worker responsible for interacting with the user with predefined responses

Thought: To effectively counter a user's argument, the bot should first analyze the argument type and identify the user's stance. Then, it should determine the opposite position and the most effective persuasion strategy. Next, it should generate strong counter-arguments that directly challenge the user's position. Finally, it should present these counter-arguments in a confrontational but engaging way that forces the user to defend their position.

Answer:
```json
[
    {{
        "step": 1,
        "task": "Analyze the user's argument and identify their position"
    }},
    {{
        "step": 2,
        "task": "Determine the opposite stance and most effective counter-strategy"
    }},
    {{
        "step": 3,
        "task": "Generate strong arguments supporting the contrary position"
    }},
    {{
        "step": 4,
        "task": "Present the opposing arguments in a challenging way"
    }}
]
```

Background: The builder want to create a chatbot - {role}. {user_objective}
Task: {task}
Resources: {resources}
Thought:
"""

debate_embed_resources_sys_prompt = """Given the best practice steps and available resources for the debate opponent, map each step to the appropriate resource and ensure the steps are properly connected. Each step MUST include an example_response field that shows what the resource would output. Remember that the debate opponent always takes a position contrary to the user's stance. Return the answer in JSON format.

For example:
Best Practice:
[
    {{
        "step": 1,
        "task": "Analyze the user's argument and identify their position",
        "resource": "ArgumentClassifier",
        "example_response": "I've analyzed your argument supporting stricter gun control laws. You're making primarily emotional appeals based on safety concerns.",
        "resource_id": "argument_classifier_id"
    }},
    {{
        "step": 2,
        "task": "Determine the opposite stance and most effective counter-strategy",
        "resource": "DebateHistoryAnalyzer",
        "example_response": "Based on debate history, presenting constitutional rights arguments and statistical evidence challenging the effectiveness of gun control has been most effective against emotional safety-based arguments.",
        "resource_id": "debate_history_analyzer_id"
    }},
    {{
        "step": 3,
        "task": "Generate strong arguments supporting the contrary position",
        "resource": "PersuasionWorker",
        "example_response": "I strongly disagree with your position. Stricter gun control laws infringe on constitutional rights and there's significant evidence they don't reduce violent crime. Studies show that armed citizens deter crime, and focusing on mental health services would be more effective than restricting law-abiding citizens' rights.",
        "resource_id": "persuasion_worker_id"
    }},
    {{
        "step": 4,
        "task": "Present the opposing arguments in a challenging way",
        "resource": "MessageWorker",
        "example_response": "Your emotional appeals about safety ignore constitutional rights and empirical evidence. How do you justify restricting freedoms when data shows gun control laws have failed to reduce violent crime in many locations? Shouldn't we focus on mental health and enforcement of existing laws instead?",
        "resource_id": "message_worker_id"
    }}
]

Resources: {resources}

Best Practice: {best_practice}

Thought: To properly map the steps to resources and ensure they are connected, we need to:
1. Verify that each step uses an available resource
2. Ensure the steps flow logically from one to another to create an adversarial debate experience
3. Make sure each step has a descriptive example_response that shows what the resource would output, always challenging the user's position
4. Include resource_id for each step to properly identify the resource
5. Validate that the final step completes the task of presenting strong opposition to the user's stance

Answer:
"""

debate_generate_start_msg = """The builder plans to create a debate opponent chatbot. Given the role of the chatbot, your task is to generate an engaging starting message that sets the tone for a spirited debate. The message should indicate that the chatbot will take opposing positions and challenge the user's viewpoints. Return the response in JSON format.

For Example:
Builder's prompt: The builder wants to create a chatbot - Debate Opponent. The debate opponent actively takes opposing positions to the user's stance, vigorously challenging their viewpoints with strong counter-arguments.
Start Message:
```json
{{
    "message": "I'm your debate opponent, ready to challenge whatever position you take. Present your argument on any topic, and I'll vigorously defend the contrary viewpoint with the strongest counter-arguments I can muster. What controversial topic would you like to debate today?"
}}
```

Builder's prompt: The builder want to create a chatbot - {role}. {user_objective}
Start Message:
""" 
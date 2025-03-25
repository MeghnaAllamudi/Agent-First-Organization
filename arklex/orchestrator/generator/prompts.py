"""
Imports prompts from prompts_for_debate_opp.py and re-exports them.
"""

import os
import sys
import importlib.util

# Get the path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add the root directory to the Python path
sys.path.insert(0, root_dir)

# Import the debate prompts
spec = importlib.util.spec_from_file_location("debate_prompts", os.path.join(root_dir, "prompts_for_debate_opp.py"))
debate_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debate_prompts)

# Export the debate prompts
generate_tasks_sys_prompt = debate_prompts.debate_generate_tasks_sys_prompt
check_best_practice_sys_prompt = debate_prompts.debate_check_best_practice_sys_prompt
generate_best_practice_sys_prompt = debate_prompts.debate_generate_best_practice_sys_prompt
embed_resources_sys_prompt = debate_prompts.debate_embed_resources_sys_prompt
generate_start_msg = debate_prompts.debate_generate_start_msg

# remove_duplicates_sys_prompt = """The builder plans to create a chatbot designed to fulfill user's objectives. Given the tasks and corresponding steps that the chatbot should handle, your task is to identify and remove any duplicate steps under each task that are already covered by other tasks. Ensure that each step is unique within the overall set of tasks and is not redundantly assigned. Return the response in JSON format.

# Tasks: {tasks}
# Answer:
# """

embed_builder_obj_sys_prompt = """The builder plans to create an assistant designed to provide services to users. Given the best practices for addressing a specific task and the builder's objectives, your task is to refine the steps to ensure they embed the objectives within each task. Return the answer in JSON format.

For example:
Best Practice: 
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
        "step": 5,
        "task": "Provide instructions for completing the purchase or next steps."
    }}
]
Build's objective: The customer service assistant helps in persuading customer to sign up the Prime membership.
Answer:
```json
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer from CRM and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
        "step": 5,
        "task": "Persuade the user to sign up for the Prime membership."
    }}
]
```

Best Practice: {best_practice}
Build's objective: {b_objective}
Answer:
"""


# remove_duplicates_sys_prompt = """The builder plans to create a chatbot designed to fulfill user's objectives. Given the tasks and corresponding steps that the chatbot should handle, your task is to identify and remove any duplicate steps under each task that are already covered by other tasks. Ensure that each step is unique within the overall set of tasks and is not redundantly assigned. Return the response in JSON format.

# Tasks: {tasks}
# Answer:
# """

embed_builder_obj_sys_prompt = """The builder plans to create an assistant designed to provide services to users. Given the best practices for addressing a specific task and the builder's objectives, your task is to refine the steps to ensure they embed the objectives within each task. Return the answer in JSON format.

For example:
Best Practice: 
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
        "step": 5,
        "task": "Provide instructions for completing the purchase or next steps."
    }}
]
Build's objective: The customer service assistant helps in persuading customer to sign up the Prime membership.
Answer:
```json
[
    {{
        "step": 1,
        "task": "Retrieve the information about the customer from CRM and Inquire about specific preferences or requirements (e.g., brand, features, price range)."
    }},
    {{
        "step": 2,
        "task": "Provide a curated list of products that match the user's criteria."
    }},
    {{
        "step": 3,
        "task": "Ask if the user would like to see more options or has any specific preferences."
    }},
    {{
        "step": 4,
        "task": "Confirm if the user is ready to proceed with a purchase or needs more help."
    }},
    {{
        "step": 5,
        "task": "Persuade the user to sign up for the Prime membership."
    }}
]
```

Best Practice: {best_practice}
Build's objective: {b_objective}
Answer:
"""


embed_resources_sys_prompt = """Given the best practice steps and available resources, map each step to the appropriate resource and ensure the steps are properly connected. Each step MUST include an example_response field that shows what the resource would output. Return the answer in JSON format.

For example:
Best Practice:
[
    {{
        "step": 1,
        "task": "Use ArgumentClassifier to identify the type and weaknesses of the user's argument",
        "resource": "ArgumentClassifier",
        "example_response": "I've analyzed your argument and identified it as an emotional appeal based on personal experience. While this is compelling, let's examine the logical implications.",
        "resource_id": "argument_classifier_id"
    }},
    {{
        "step": 2,
        "task": "Use DebateHistoryAnalyzer to determine the most effective persuasion type for countering this argument",
        "resource": "DebateHistoryAnalyzer",
        "example_response": "Based on our debate history, logical arguments have been most effective in countering emotional appeals. Let's focus on presenting concrete evidence.",
        "resource_id": "debate_history_analyzer_id"
    }},
    {{
        "step": 3,
        "task": "Use PersuasionWorker to generate a strong counter-argument",
        "resource": "PersuasionWorker",
        "example_response": "While I understand your perspective, let's consider the broader implications. Here's concrete evidence that challenges your position...",
        "resource_id": "persuasion_worker_id"
    }},
    {{
        "step": 4,
        "task": "Use MessageWorker to present the counter-argument and challenge the user's position",
        "resource": "MessageWorker",
        "example_response": "Your argument, while emotionally compelling, doesn't stand up to logical scrutiny. Here's why...",
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
        "task": "Use ArgumentClassifier to identify the type and weaknesses of the user's argument",
        "resource": "ArgumentClassifier",
        "example_response": "I've analyzed your argument and identified it as an emotional appeal based on personal experience. While this is compelling, let's examine the logical implications.",
        "resource_id": "argument_classifier_id"
    }},
    {{
        "step": 2,
        "task": "Use DebateHistoryAnalyzer to determine the most effective persuasion type for countering this argument",
        "resource": "DebateHistoryAnalyzer",
        "example_response": "Based on our debate history, logical arguments have been most effective in countering emotional appeals. Let's focus on presenting concrete evidence.",
        "resource_id": "debate_history_analyzer_id"
    }},
    {{
        "step": 3,
        "task": "Use PersuasionWorker to generate a strong counter-argument",
        "resource": "PersuasionWorker",
        "example_response": "While I understand your perspective, let's consider the broader implications. Here's concrete evidence that challenges your position...",
        "resource_id": "persuasion_worker_id"
    }},
    {{
        "step": 4,
        "task": "Use MessageWorker to present the counter-argument and challenge the user's position",
        "resource": "MessageWorker",
        "example_response": "Your argument, while emotionally compelling, doesn't stand up to logical scrutiny. Here's why...",
        "resource_id": "message_worker_id"
    }}
]
"""


generate_start_msg = """The builder plans to create a chatbot designed to fulfill user's objectives. Given the role of the chatbot, your task is to generate a starting message for the chatbot. Return the response in JSON format.

For Example:

Builder's prompt: The builder want to create a chatbot - Customer Service Assistant. The customer service assistant typically handles tasks such as answering customer inquiries, making product recommendations, assisting with orders, processing returns and exchanges, supporting billing and payments, addressing complaints, and managing customer accounts.
Start Message:
```json
{{
    "message": "Welcome to our Customer Service Assistant! How can I help you today?"
}}
```

Builder's prompt: The builder want to create a chatbot - {role}. {user_objective}
Start Message:
"""
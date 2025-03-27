import json
import random
from arklex.evaluation.get_documents import load_docs
from arklex.evaluation.chatgpt_utils import (chatgpt_chatbot, query_chatbot, filter_convo, adjust_goal,
                                               flip_hist, generate_goals, format_chat_history_str, flip_hist_content_only)


def check_goal_completion(goal, convo):
    convo_str = format_chat_history_str(flip_hist_content_only(convo[2:]))
    #DEBATE OPPONENT PROMPT
    prompt = f"Here is a conversation between a user and a debate agent:\n{convo_str}\n\nThe user's goal is the following: {goal}\nOutput False if the debate opponent is not engaging in a back and forth debate with the user. Output True otherwise. Only output True or False."
    #CUSTOMER SERVICE PROMPT
    #prompt = f"Here is a conversation between a user and a customer service chatbot assistant:\n{convo_str}\n\nThe user's goal is the following: {goal}\nOutput False if the user needs to learn more information regarding their goal. Output True otherwise. Only onput True or False and nothing else."
    output = chatgpt_chatbot([{'role': 'user', 'content': prompt}])
    print(f"CHECK GOAL OUTPUT {output}")
    return output == "True"

def conversation(model_api, goal, summary, model_params, synthetic_data_params, env_config):
    history = []
    #DEBATE OPPONENT PROMPT
    instructional_prompt = f'Replicate the writing behavior of a debate opponent. You are interacting with a debate opponent chatbot that will have a stance that is opposite of yours for a specific topic of their choosing. \nYou have the following goal when interacting with this chatbot:\n{goal}\n Have a conversation with the chatbot while trying to achieve this goal. Make sure the conversation is natural. For example, if the chatbot brings up a counter argument, you can choose to agreew with some of the points but try to defend your stance.'
    start_text = "Humans write short questions with typos and a neutral sentiment. Here are some examples of what a human customer would type: [I agree with your point, but.., I disagree completely, Wouldn't you agree that this...]. Replicate the writing behavior of a human debate opponent and let the chatbot choose the debate topic for the conversation."
    #CUSTOMER SERVICE PROMPT
    #instructional_prompt = f'Replicate the writing behavior of a human customer. You are interacting with a customer service chatbot for the following company: {summary}\nYou have the following goal when interacting with this chatbot:\n{goal}\n Have a conversation with the chatbot while trying to achieve this goal. Make sure the conversation is natural. For example, if the chatbot asks you a question you should answer it.'
    #start_text = "Humans write short questions with typos and a neutral sentiment. Here are some examples of what a human customer would type: [how much is it?, Can you send info to my email, yes I need a job, want to check both proposals to rent and buy, How much does it cost a [PRODUCT_HERE], Im interested in [PRODUCT_HERE], hi i would like to rent out [PRODUCT_HERE] but im wondering which countries are available for rental]. Replicate the writing behavior of a human customer and begin the conversation with a question to achieve your goal."
    history.append({'role': 'system','content': instructional_prompt})
    history.append({'role': 'user', 'content': start_text})
    chatbot_history = []
    print(f"LENGTH OF MAX_TURNS {synthetic_data_params['max_turns']}")
    for i in range(synthetic_data_params['max_turns']):
        output = chatgpt_chatbot(history) 
        history.append({'role': 'assistant', 'content': output})
        chatbot_history.append({'role': 'assistant', 'content': output})
        response_data = query_chatbot(model_api, chatbot_history, model_params, env_config)
        answer = response_data["answer"]
        answer = answer.replace('\n', ' ')
        model_params = response_data["parameters"]
        pred_intent = response_data['parameters']['nlu_records'][-1]['pred_intent']
        history[-1]['intent'] = pred_intent

        history.append({'role': 'user', 'content': answer})
        chatbot_history.append({'role': 'user', 'content': answer})
        is_goal_complete = check_goal_completion(goal, history.copy())
        #print(f"IS GOAL COMPLETE {is_goal_complete} AND {i}") 
        if i > 2 and is_goal_complete:
            history.append({'goal_completetion': True})
            break
    
    if not history[-1].get('goal_completetion', False):
        history.append({'goal_completetion': False})
    history.append({'trajectory': model_params["history"]})
    #print(f"HISTORY {history}")
    return history

def generate_conversations(model_api, goals, summary, model_params, synthetic_data_params, env_config):
    convos = []
    # for i in range(synthetic_data_params['num_convos']):
    for goal in goals:
        # goal = random.choice(goals)
        convo = conversation(model_api, goal, summary, model_params, synthetic_data_params, env_config)
        convos.append(flip_hist(filter_convo(convo, filter_turns=False)))
    return convos

def simulate_conversations(model_api, model_params, synthetic_data_params, config):
    documents = load_docs(config['documents_dir'], config, synthetic_data_params['num_goals'] * 2)
    summary = config['intro']
    env_config = {
        "workers": config['workers'],
        "tools": config["tools"]
    }
    
    final_goals = []
    if synthetic_data_params.get('goals', None):
        raw_goals = []
        cases = synthetic_data_params['goals']
        for stage, categories in cases.items():
            for first_level, second_levels in categories.items():
                for second_level, goals in second_levels.items():
                    raw_goal = goals[0]
                    raw_goals.append(raw_goal)
        
        # goal adaptation
        final_goals = []
        for goal in raw_goals:
            doc = random.choice(documents)
            new_goal = adjust_goal(doc, goal)
            final_goals.append(new_goal)

    else:
        final_goals = generate_goals(documents, synthetic_data_params)
    try:
        conversations = generate_conversations(
            model_api,
            final_goals,
            summary,
            model_params,
            synthetic_data_params,
            env_config,
        )
    except Exception as e:
        print("Generate conversations failed")
        print("Error: ", e)
        conversations = []
    return conversations, final_goals

if __name__ == "__main__":
    model_api = "http://adaptation.cs.columbia.edu:55231/qa/richtech/v1alpha1"
    synthetic_data_params = {'num_convos': 5, 'num_goals': 3, 'max_turns': 10}
    model_params = {}
    convos  = simulate_conversations(model_api, model_params, synthetic_data_params)
    with open('p1_sample_convos.json', 'w') as f:
        json.dump(convos, f, indent=5)
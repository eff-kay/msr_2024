
import json
import pandas as pd
import os
import re
import requests


ENDPOINT = "https://api.openai.com/v1/chat/completions"

# sogols api key
ACCESS_KEY ="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"

# faizank api key
ACCESS_KEY ="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"



def prep_data_for_pipeline():
    # this creates files with the relevant issues, and discussions
    data_files = []
    for file in os.listdir('snapshot_20231012'):
        json_file = re.findall(r'^(2023.+)', file)
        data_files.append(json_file[0]) if len(json_file)>0 else None
    
    # save file
    for data_file in data_files:
        df_ret = save_file(data_file)
        df_ret.to_json(open(f'new_artifacts/{data_file.split(".")[0]}_multiple.json', 'w'))
    
    return True

def save_file(starting_file):
    x = Path('snapshot_20231012', starting_file)
    df_chat1 = pd.DataFrame(pd.read_json(x)["Sources"].tolist())
    df_chat1.columns

    chat_gpt_sharings = []
    for row in df_chat1['ChatgptSharing']:
        for chat in row:
            chat_gpt_sharings.append(chat)
    
    df_chat1_inverted = pd.DataFrame(chat_gpt_sharings)
    print(df_chat1_inverted.shape)

    # value_counts = df_chat1['ChatgptSharing'].apply(lambda x:len(x)).value_counts()
    # index = df_chat1['ChatgptSharing'].apply(lambda x:len(x)).value_counts().index

    # (value_counts*index).sum() == df_chat1_inverted.shape[0]

    # collect all sharing with valid conversations
    df_chat1_inverted_with_conversation = df_chat1_inverted[~df_chat1_inverted['Conversations'].isna()]
    print(df_chat1_inverted_with_conversation.shape)


    # with multiple conversations
    df_chat1_multiple = df_chat1_inverted_with_conversation[df_chat1_inverted_with_conversation['Conversations'].apply(lambda x:len(x)>1)]

    print(df_chat1_multiple.shape)

    print(df_chat1_multiple["Model"].value_counts())

    # remove the models
    df_chat1_multiple_possible = df_chat1_multiple[df_chat1_multiple['Model'].apply(lambda x: x not in ['Advanced Data Analysis', 'Web Browsing', 'Plugins'])]

    print(df_chat1_multiple_possible.shape)

    # create the prompt
    df_chat1_multiple_possible[["PromptForExperiment", "RespForExperiment"]] = df_chat1_multiple_possible.apply(lambda x: create_one_prompt(x.Conversations), axis='columns', result_type='expand')

    # create msg array
    df_chat1_multiple_possible[["PromptForExperiment2", "ObjectForExp2"]] = df_chat1_multiple_possible.apply(lambda x: create_message_arr(x.Conversations), axis='columns', result_type='expand')

    df_chat1_multiple_possible["normalized_models"] = df_chat1_multiple_possible["Model"].apply(lambda x: model_mapping(x))

    return df_chat1_multiple_possible

def normalize_resp(conversation):

    answer = None
    if len(conversation['ListOfCode'])>0:
        # normalize the answser text
        answer = conversation['Answer']
        for ob in conversation['ListOfCode']:
            answer = answer.replace(ob['ReplaceString'], f"\n```{ob.get('Type')}\n{ob['Content']}```\n")
    else:
        answer = conversation["Answer"]
    
    return answer

def create_message_arr(conversations):
    messages = []
    context = conversations[:-1]

    # the last one is the one that we want to consider
    candidate = conversations[-1]

    for c in context:
        if len(c['ListOfCode'])>0:
            # normalize the answser text
            answer = c['Answer']
            for ob in c['ListOfCode']:
                answer = answer.replace(ob['ReplaceString'], f"\n```{ob.get('Type')}\n{ob['Content']}```\n")
        else:
            answer = c["Answer"]

        # replace ChatGPT at the start of every answer
        answer = re.sub(r"^ChatGPT", "", answer)

        user_dict = {
            "role": "user",
            "content": c['Prompt']
        }

        assistant_dict = {
            "role": "assistant",
            "content": answer
        }

        messages.append(user_dict)
        messages.append(assistant_dict)

    prompt_str = {candidate['Prompt']}

    return prompt_str, messages


def create_one_prompt(conversations):
    # all of this needs to be context
    context = conversations[:-1]

    # the last one is the one that we want to consider
    candidate = conversations[-1]

    context_str = ""
    for c in context:
        if len(c['ListOfCode'])>0:
            # normalize the answser text
            answer = c['Answer']
            for ob in c['ListOfCode']:
                answer = answer.replace(ob['ReplaceString'], f"\n```{ob.get('Type')}\n{ob['Content']}```\n")
        else:
            answer = c["Answer"]

        # replace ChatGPT at the start of every answer
        answer = re.sub(r"^ChatGPT", "You responded with: ", answer)
        context_str += f"I asked: {c['Prompt']}" + "\n\n" + answer + "\n\n"
    
    prompt_str = context_str + "\n" + f"Now my question is: {candidate['Prompt']}"
    
    # normalize the answser text
    expected_resp = candidate['Answer']
    if len(candidate['ListOfCode'])>0:
        # normalize the answser text
        expected_resp = candidate['Answer']
        for ob in candidate['ListOfCode']:
            expected_resp = expected_resp.replace(ob['ReplaceString'], f"\n```{ob.get('Type')}\n{ob['Content']}```\n")

    # replace ChatGPT at the start of every answer
    expected_resp = re.sub(r"^ChatGPT", "", expected_resp)

    return prompt_str, expected_resp

def gpt_prompt_automation(model, prompt):
    """Automating the prompt requet to open ai api: first method"""

    headers = {
        "Authorization": "Bearer {}".format(ACCESS_KEY),
        "Content-Type": "application/json",
        "User-Agent": "OpenAI-Python"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        # For now, we will not use seed and temp. together.
        # "seed": 42
        # "temperature": 0
    }
    response = requests.post(ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        try:
            error_message = response.json().get("error", {}).get("message", "Unknown error")
            print(f"Error {response.status_code}: {error_message}")
            return f"Error {response.status_code}: {error_message}"
        # try to catch any type of exception
        except Exception as e:
            return "Exception: {}".format(e)

def gpt_embeding_automation(model, prompt_arr, prompt):
    """Automating the prompt request to open ai api: second method"""
    headers = {
            "Authorization": "Bearer {}".format(ACCESS_KEY),
            "Content-Type": "application/json",
            "User-Agent": "OpenAI-Python"}
    data = {
                "model": model,
                "messages":[
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
            }

    data["messages"].extend(prompt_arr)
    prompt_dict = {"role": "user", "content": prompt}

    data["messages"].append(prompt_dict)
    assert len(data["messages"]) == (len(prompt_arr) + 2)

    response = requests.post(ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        try:
            error_message = response.json().get("error", {}).get("message", "Unknown error")
            print(f"Error {response.status_code}: {error_message}")
            return response.status_code
        # try to catch any type of exception
        except:
            return "Exception"

MODEL_MAP = {
    'Default (GPT-3.5)': 'gpt-3.5-turbo',
    'GPT-4': 'gpt-4',
    'Default': 'gpt-3.5-turbo',
    'Model: Default':'gpt-3.5-turbo',
}

def model_mapping(model):
    return MODEL_MAP[model]

def run_pipeline(file_name):

    # read the files
    df = pd.read_json(open(f'new_artifacts/{file_name}.json', 'r'))

    df = df
    print(df.shape)

    starting_point = 0
    df = df[starting_point:2]

    with open(f'results/results_{file_name}_exp1.json', 'a') as write_file:
        result_dict = {}
        for i, (index, row) in enumerate(df.iterrows(), starting_point):
            print(row["URL"], row["normalized_models"])
            result_dict['URL'] = row['URL']
            result_dict['NormalizedModel'] = row['normalized_models']
            result_dict['PromptForExperiment'] = row['PromptForExperiment']
            result_dict['RespForExperiment'] = row['RespForExperiment']

            print('evaluating row: ', i, index)

            resp = gpt_prompt_automation(row["normalized_models"], row["PromptForExperiment"])
            result_dict['FirstEmbedResp'] = resp
        
            write_file.write("\t"+ json.dumps(result_dict))
            write_file.write(",\n")
    
    print('Done')

def run_pipeline_2(file_name):

    # read the files
    df = pd.read_json(open(f'new_artifacts/{file_name}.json', 'r'))

    df = df
    print(df.shape)

    starting_point = 0
    df = df[starting_point:2]

    with open(f'results_2/results_{file_name}_exp2.json', 'a') as write_file:
        result_dict = {}
        for i, (index, row) in enumerate(df.iterrows(), starting_point):
            print(row["URL"], row["normalized_models"])
            result_dict['URL'] = row['URL']
            result_dict['NormalizedModel'] = row['normalized_models']
            result_dict['PromptForExperiment'] = row['PromptForExperiment']
            result_dict['RespForExperiment'] = row['RespForExperiment']

            print('evaluating row: ', i, index)

            resp = gpt_embeding_automation(row["normalized_models"], row['ObjectForExp2'], row["PromptForExperiment2"])
            result_dict['FirstEmbedResp'] = resp
        
            write_file.write("\t"+ json.dumps(result_dict))
            write_file.write(",\n")
    
    print('Done')

from pathlib import Path
if __name__ == "__main__":

    # resp = prep_data_for_pipeline()
    # done = ['pr_sharings_multiple']
    # prep_data_for_pipeline()

    # artifact_names = ['commit_sharings_multiple', 'discussion_sharings_multiple', 'issue_sharings_multiple', 'file_sharings_multiple', 'hn_sharings_multiple', 'issue_sharings_multiple']

    # # commit sharing pipeline: next_no 100

    run_pipeline('issue_sharings_multiple')
    # run_pipeline_2('commit_sharings_multiple')

    print('DONE')





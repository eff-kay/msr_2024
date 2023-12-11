import pandas as pd
import os
import re
from pathlib import Path

def print_total_issues_in_raw_source_files():
    # this creates files with the relevant issues, and discussions
    data_files = []
    for file in os.listdir('snapshot_20231012'):
        json_file = re.findall(r'^(2023.+)', file)
        data_files.append(json_file[0]) if len(json_file)>0 else None

    fig1_dict = {} 
    # save file
    total = 0
    total_b = 0
    for data_file in data_files:
        x = Path('snapshot_20231012', data_file)
        df_chat1 = pd.DataFrame(pd.read_json(x)["Sources"].tolist())
        total+=df_chat1.shape[0]
        print('total issues', data_file, df_chat1.shape)

        chat_gpt_sharings = []
        for row in df_chat1['ChatgptSharing']:
            for chat in row:
                chat_gpt_sharings.append(chat)
        
        df_chat1_inverted = pd.DataFrame(chat_gpt_sharings)
        print('total_chats', df_chat1_inverted.shape)

        # collect all sharing with valid conversations
        df_chat1_inverted_with_conversation = df_chat1_inverted[~df_chat1_inverted['Conversations'].isna()]
        print(df_chat1_inverted_with_conversation.shape)


        # with multiple conversations
        df_chat1_multiple = df_chat1_inverted_with_conversation[df_chat1_inverted_with_conversation['Conversations'].apply(lambda x:len(x)>1)]

        total_b += df_chat1_multiple.shape[0]
        print('multiple conv', data_file, df_chat1_multiple.shape)

        print(df_chat1_multiple["Model"].value_counts())

        # remove the models
        df_chat1_multiple_possible = df_chat1_multiple[df_chat1_multiple['Model'].apply(lambda x: x not in ['Advanced Data Analysis', 'Web Browsing', 'Plugins'])]

        print('removed models', df_chat1_multiple_possible.shape, data_file)
    
    print(total, total_b)

def print_multi_prompt_conversations_in_artifacts():
    artifact_files = [x for x in os.listdir('new_artifacts')]

    total = 0
    for artifact_file in artifact_files:
        df = pd.read_json(f'new_artifacts/{artifact_file}')
        import pdb
        pdb.set_trace()
        print(artifact_file, df.shape)
        total+=df.shape[0]
    
    print(total)

if __name__=='__main__':
    print_multi_prompt_conversations_in_artifacts()

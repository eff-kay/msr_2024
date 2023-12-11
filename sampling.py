import pandas as pd
import os


def sample_df(artifact_location):
    artifact_files = [x for x in os.listdir(artifact_location)]
    dfs = [pd.read_json(f'{artifact_location}/{x}') for x in artifact_files]

    combined_df = pd.concat(dfs)
    # drop the index column
    sample = combined_df.sample(311, random_state=1).reset_index()

    return sample

def next_sample(artifact_location, previous_samples):
    artifact_files = [x for x in os.listdir(artifact_location)]
    dfs = [pd.read_json(f'{artifact_location}/{x}') for x in artifact_files]

    combined_df = pd.concat(dfs)
    # drop the index column
    # this is 2 for the second sample
    state = len(previous_samples)+1
    sample = combined_df.sample(311, random_state=state).reset_index()

    # make sure that the sample is not in the previous samples
    previous_samples = [pd.read_json(x) for x in previous_samples]
    for previous_sample in previous_samples:

        print(sample.shape)
        sample = sample[~sample['URL'].isin(previous_sample['URL'])]
        # shape after filtering
        print(sample.shape)

    return sample

def filter_for_400(result_location)->int:
    exp1 = pd.read_json(f'{result_location}/results_sample_exp1.json')
    exp2 = pd.read_json(f'{result_location}/results_sample_exp2.json')

    err_exp1 = exp1[exp1['FirstEmbedResp'].apply(lambda x: x.startswith('Error'))]
    err_exp2 = exp2[exp2['SecondEmbedResp'].apply(lambda x: isinstance(x, int) and x==400)]

    total_err = pd.concat([err_exp1, err_exp2], axis=1)
    total_err = total_err.reset_index()
    print(total_err.shape)

    exp1 = exp1[~exp1['FirstEmbedResp'].apply(lambda x: x.startswith('Error'))]
    exp2 = exp2[exp2['SecondEmbedResp'].apply(lambda x: isinstance(x, str))]

    exp1.to_json(f'{result_location}/results_sample_exp1_without_err.json')
    exp2.to_json(f'{result_location}/results_sample_exp2_without_err.json')

def filter_for_400_2(result_location)->int:
    exp1 = pd.read_json(f'{result_location}/results_sample2_exp1.json')
    exp2 = pd.read_json(f'{result_location}/results_sample2_exp2.json')

    err_exp1 = exp1[exp1['FirstEmbedResp'].apply(lambda x: x.startswith('Error'))]
    err_exp2 = exp2[exp2['SecondEmbedResp'].apply(lambda x: isinstance(x, int) and x==400)]

    total_err = pd.concat([err_exp1, err_exp2], axis=1)
    total_err = total_err.reset_index()
    print(total_err.shape)

    exp1 = exp1[~exp1['FirstEmbedResp'].apply(lambda x: x.startswith('Error'))]
    exp2 = exp2[exp2['SecondEmbedResp'].apply(lambda x: isinstance(x, str))]

    exp1.to_json(f'{result_location}/results_sample2_exp1_without_err.json')
    exp2.to_json(f'{result_location}/results_sample2_exp2_without_err.json')
    print(exp1.shape)

if __name__=="__main__":
    # artifact_names = ['commit_sharings_multiple', 'discussion_sharings_multiple', 'issue_sharings_multiple', 'file_sharings_multiple', 'hn_sharings_multiple', 'issue_sharings_multiple']

    # sample = sample_df('new_artifacts')
    # # save the sample
    # sample.to_json('artifact_for_rq1/sample.json')
    filter_for_400('results_for_rq1')
    filter_for_400_2('results_for_rq1')
    # sample = next_sample('new_artifacts', ['artifact_for_rq1/sample.json'])
    # sample.to_json('artifact_for_rq1/sample2.json')
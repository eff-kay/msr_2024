import pandas as pd


def calculate_code_column():
    # then read the data from the final_result
    exp_1 = pd.read_json('results_for_rq1/final_exp1_with_scores.json')
    exp_2 = pd.read_json('results_for_rq1/final_exp2_with_scores.json')

    exp_1['code_snippet'] = exp_1.apply(lambda x: True if '```' in x['RespForExperiment'] else False, axis=1)
    exp_2['code_snippet'] = exp_2.apply(lambda x: True if '```' in x['RespForExperiment'] else False, axis=1)

    exp_1.to_json('artifact_for_rq2/final_exp1_with_scores.json')
    exp_2.to_json('artifact_for_rq2/final_exp2_with_scores.json')

def calculate_code_table():

    exp_1 = pd.read_json('artifact_for_rq2/final_exp1_with_scores.json')
    exp_2 = pd.read_json('artifact_for_rq2/final_exp2_with_scores.json')

    exp_1_filt = exp_1[exp_1['code_snippet']==True]

    exp_2_filt = exp_2[exp_2['code_snippet']==True]
    print(len(exp_1_filt))
    print(len(exp_2_filt))
    # calculate a different score for code snippets and non-code snippets
    scores = {}

    scores['avg_resp_size'] = [exp_1_filt.RespForExperiment.apply(lambda x: len(x.split())).mean(), exp_2_filt.RespForExperiment.apply(lambda x: len(x.split())).mean()]

    scores['f1'] = [exp_1_filt.f1_score.mean(), exp_2_filt.f1_score.mean()]

    scores['levenstien_raw'] = [exp_1_filt.lavenstien.mean(), exp_2_filt.lavenstien.mean()]

    scores['levenstien_norm'] = [exp_1_filt.norm_lavenstien.mean(), exp_2_filt.norm_lavenstien.mean()]

    scores_df = pd.DataFrame.from_dict(scores, orient='index')

    scores_df.columns = ['exp1', 'exp2']
    print(scores_df.to_latex())

def table_without_code_snippets():

    exp_1 = pd.read_json('artifact_for_rq2/final_exp1_with_scores.json')
    exp_2 = pd.read_json('artifact_for_rq2/final_exp2_with_scores.json')

    # calculate a different score for code snippets and non-code snippets
    exp_1_filt = exp_1[exp_1['code_snippet']==False]
    exp_2_file = exp_2[exp_2['code_snippet']==False]

    print(len(exp_1_filt))
    print(len(exp_2_file))

    # import pdb
    # pdb.set_trace()

    scores = {}

    # mean median and mode in one cell

    scores['avg_resp_size'] = [exp_1_filt.RespForExperiment.apply(lambda x: len(x.split())).mean(), exp_2_file.RespForExperiment.apply(lambda x: len(x.split())).mean()]

    scores['f1'] = [exp_1_filt.f1_score.mean(), exp_2_file.f1_score.mean()]

    scores['levenstien_raw'] = [exp_1_filt.lavenstien.mean(), exp_2_file.lavenstien.mean()]

    scores['levenstien_norm'] = [exp_1_filt.norm_lavenstien.mean(), exp_2_file.norm_lavenstien.mean()]

    scores_df = pd.DataFrame.from_dict(scores, orient='index')

    scores_df.columns = ['exp1', 'exp2']
    print(scores_df.to_latex())

if __name__=="__main__":
   calculate_code_table()
   table_without_code_snippets()
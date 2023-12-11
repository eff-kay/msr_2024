import re
import string
import collections
import pandas as pd

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()


import jieba
def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)

  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  if len(common)==0:
    gold_toks = jieba.lcut(a_gold)
    pred_toks = jieba.lcut(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)

  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

import pylcs
def compute_levenstien(a_gold, a_pred):
    return pylcs.edit_distance(a_gold, a_pred)

def calculate_final_dfs_with_scores():
    exp1 = 'results_for_rq1/results_sample_exp1_without_err'
    exp2 = 'results_for_rq1/results_sample_exp2_without_err'

    exp1_s2 = 'results_for_rq1/results_sample2_exp1_without_err'
    exp2_s2 = 'results_for_rq1/results_sample2_exp2_without_err'

    df_exp1 = pd.read_json(open(f'{exp1}.json'))
    df_exp2 = pd.read_json(open(f'{exp2}.json'))

    df_exp1_s2 = pd.read_json(open(f'{exp1_s2}.json'))
    df_exp2_s2 = pd.read_json(open(f'{exp2_s2}.json'))

    combined_exp1 = pd.concat([df_exp1, df_exp1_s2]).drop_duplicates(subset=['URL'], keep='first')[:311]
    combined_exp2 = pd.concat([df_exp2, df_exp2_s2]).drop_duplicates(subset=['URL'], keep='first')[:311]

    combined_exp1['f1_score'] = combined_exp1.apply(lambda x: compute_f1(x['RespForExperiment'], x['FirstEmbedResp']), axis=1, result_type='expand')

    combined_exp2['f1_score'] = combined_exp2.apply(lambda x: compute_f1(x['RespForExperiment'], x['SecondEmbedResp']), axis=1, result_type='expand')

    combined_exp1['lavenstien'] = combined_exp1.apply(lambda x: compute_levenstien(x['RespForExperiment'], x['FirstEmbedResp']), axis=1, result_type='expand')

    # min-max normalization of levenstien
    combined_exp1['norm_lavenstien'] = (combined_exp1['lavenstien'] - combined_exp1['lavenstien'].min()) / (combined_exp1['lavenstien'].max() - combined_exp1['lavenstien'].min())

    combined_exp2['lavenstien'] = combined_exp2.apply(lambda x: compute_levenstien(x['RespForExperiment'], x['SecondEmbedResp']), axis=1, result_type='expand')

    # min-max normalization of levenstien
    combined_exp2['norm_lavenstien'] = (combined_exp2['lavenstien'] - combined_exp2['lavenstien'].min()) / (combined_exp2['lavenstien'].max() - combined_exp2['lavenstien'].min())


    combined_exp1 = combined_exp1.reset_index()
    combined_exp2 = combined_exp2.reset_index()
    combined_exp1.to_json(f'results_for_rq1/final_exp1_with_scores.json')
    combined_exp2.to_json(f'results_for_rq1/final_exp2_with_scores.json')


def calculate_scores_from_df():
    df_exp1 = pd.read_json(open(f'results_for_rq1/final_exp1_with_scores.json'))
    df_exp2 = pd.read_json(open(f'results_for_rq1/final_exp2_with_scores.json'))
    scores = {'f1_score': [df_exp1.f1_score.mean(), df_exp2.f1_score.mean()]}

    scores['levenstien_raw'] = [df_exp1.lavenstien.mean(), df_exp2.lavenstien.mean()]
    scores['levenstien_norm'] = [df_exp1.norm_lavenstien.mean(), df_exp2.norm_lavenstien.mean()]

    scores_df = pd.DataFrame.from_dict(scores, orient='index')
    scores_df.columns = ['exp1', 'exp2']
    print(scores_df.to_latex())

if __name__=="__main__":
    calculate_scores_from_df()
    # calculate_final_dfs_with_scores()
    # df_exp1 = pd.read_json(open(f'results_for_rq1/final_exp1_with_scores.json'))
    # df_exp2 = pd.read_json(open(f'results_for_rq1/final_exp2_with_scores.json'))
    # scores = {'exp1_score': df_exp1.f1_score.mean()}
    # scores['exp2_score'] = df_exp2.f1_score.mean()

    # scores['exp1_lavenstien'] = df_exp1.lavenstien.mean()
    # scores['exp2_lavenstien'] = df_exp2.lavenstien.mean()

    # scores['exp1_norm_lavenstien'] = df_exp1.norm_lavenstien.mean()
    # scores['exp2_norm_lavenstien'] = df_exp2.norm_lavenstien.mean()

    # scores_df = pd.DataFrame.from_dict(scores, orient='index').to_latex()
    # print(scores_df)



import pandas as pd
import numpy as np

from tqdm import tqdm

import argparse
import math
import yaml
import glob

from src.fit_survival_function import fit_per_length_survival_function
from src.HC_survival_function import get_HC_survival_function
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareArticles import PrepareArticles
from src.DetectLM import DetectLM

from sklearn.metrics import confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'device: {device}')

def get_survival_function(df, G=101):
    """
    Returns a survival function for every sentence length in tokens.

    Args:
    :df:  data frame with columns 'response' and 'length'
    :G:   number of interpolation points
    
    Return:
        bivariate function (length, responce) -> (0,1)

    """
    assert not df.empty
    value_name = "response" if "response" in df.columns else "logloss"

    df1 = df[~df[value_name].isna()]
    ll = df1['length']
    xx1 = df1[value_name]
    return fit_per_length_survival_function(ll, xx1, log_space=True, G=G)

def get_threshold(df, statistics_type, threhsold_jumps=0.1):
    # Get thresholds options
    min_stats = math.floor(df[statistics_type].min())
    max_stats = math.ceil(df[statistics_type].max())
    thresholds = [i for i in np.arange(min_stats, max_stats, threhsold_jumps)]
    
    # For each threshold
    for threshold in thresholds:
        pred = df[statistics_type].apply(lambda x: 1 if x > threshold else 0) # Use it

        # Calculate FPR
        FP = len(pred[pred==1])
        TN = len(pred[pred==0])
        FPR = FP / (FP + TN)
        
        # If FP @ 0.05
        if FPR <= 0.05:
            return threshold
        
    return -1

def calc_metrics_power_analysis(pred):
    TP = len(pred[pred==1])
    FN = len(pred[pred==0])
    FP = 0
    TN = 0

    TPR = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TPR, accuracy

# Create argparse
parser = argparse.ArgumentParser(description="Read args to parse article")
parser.add_argument('-conf', type=str, help='configurations file', default="simulate_replication_conf_power_analysis.yml")

# Read conf.yaml
args = parser.parse_args()
with open(args.conf, "r", encoding='utf8') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Init parameters
k_folds = params['k-folds']                                 # Number of folds
train_split= params['train-split']                          # Split the train set into 80/20 (Example train-split=0.8). Using the 0.2 for calibration
window_size = params['window_size']                         # Determine the window size, make it easy to adjust the train/test ratio. Example 2 => 80-20 when k-fold=10
edit_ratios = params['edit-ratios']                         # Edit ratio
topics = params['topics']                                   # Topic to run on
number_sentences = params['number-sentences']               # Number Of sentence in article
files_path = params['files-path']                           # Path to the folder containing all the topics folders 
output_path = params['output-path']                         # Output path for the results     
lm_name = params['language-model-name']                     # Language model name (Example "gpt2-xl"/"microsoft/phi-2")

min_tokens_per_sentence = params['min-tokens-per-sentence'] # Min token to consider
max_tokens_per_sentence = params['max-tokens-per-sentence'] # Max token to consider 

sentences_logloss = params['sentences-logloss']             # If using cache, the path for the json cache file

# Loading model
print(f"Loading Language model {lm_name}...")
tokenizer = AutoTokenizer.from_pretrained(lm_name)
model = AutoModelForCausalLM.from_pretrained(lm_name)
model.to(device)

# Loading LPPT evaluator
print(f"Loading LPPT evaluator...")
sentence_detector = PerplexityEvaluator(model, tokenizer, cache_logloss_path=sentences_logloss)

# Save the results to output later. For each topic + edit_ratio we save the accuracy, FPR, accuracy@FPR@0.05, std_accuracy, std_accuracy_005
lst_df_global_results = []
data_length = 0
# Run over the data
for topic in topics:
    for number_sentence in number_sentences:
        for edit_ratio in edit_ratios:
            INPUT_FILE = f'{files_path}/{topic}/{number_sentence}_sentences/{edit_ratio}/{topic}.csv' # Get the data file for the topic + edit ratio

            # Read input file
            input_file_df = pd.read_csv(INPUT_FILE)
            data_length = len(input_file_df) 
            # Get splits
            splits = [i for i in range(0, len(input_file_df), len(input_file_df) // k_folds)]

            # Save results for parsed articles
            lst_df_results = []

            # Do it k_folds times
            for fold in range(k_folds):
                print(f'Test fold: {fold} out of {k_folds} folds')
                
                # Create the split
                df_null = None
                df_test = None
                current_window_size = window_size

                for i in range(len(splits) - 1):
                    temp_df = input_file_df.iloc[splits[i]:splits[i+1]]
                    if i == fold: # Take test
                        if df_test is None:
                            df_test = temp_df
                        else:
                            df_test = pd.concat([df_test, temp_df])
                    else: 
                        if current_window_size > 1: # Take test Take test until reaching the window size
                            if df_test is None:
                                df_test = temp_df
                            else:
                                df_test = pd.concat([df_test, temp_df]) 
                            current_window_size -= 1
                            continue
                        
                        # Take null
                        if df_null is None:
                            df_null = temp_df
                        else:
                            df_null = pd.concat([df_null, temp_df])
        
                # Use the train split
                df_test = pd.concat([df_test, df_null[int(len(df_null) * train_split):]], ignore_index=True)
                df_null = df_null[:int(len(df_null) * train_split)]

                # Create null distribution
                print('Create null distribution')
                parse_chunks = PrepareArticles(input_file='', df_input=df_null)
                chunks = parse_chunks()
                responses = []
                
                for sent in tqdm(chunks['text']):
                    responses.append(sentence_detector(sent))

                # Create null dataframe
                df_null = pd.DataFrame()
                df_null['text'] = chunks['text']
                df_null['response'] = np.array(responses)
                df_null['length'] = chunks['length']
                
                if params['ignore-first-sentence'] and 'num' in df_null.columns:
                    df_null = df_null[df_null.num > 1]
                    
                # Fitting null log-loss survival function
                print(f"Fitting null log-loss survival function using data from splits.")
                print(f"Found {len(df_null)} log-loss values of text atoms in splits.")
                pval_functions = get_survival_function(df_null, G=params['number-of-interpolation-points'])

                # Initializing detector
                print("Initializing detector...")
                detector = DetectLM(sentence_detector, pval_functions,
                                    min_len=min_tokens_per_sentence,
                                    max_len=max_tokens_per_sentence,
                                    length_limit_policy='max_available',
                                    HC_type=params['hc-type'],
                                    gamma=params['gamma'],
                                    ignore_first_sentence=False,
                                    cache_logloss_path=sentences_logloss
                                    )

                HC_pval_func = get_HC_survival_function(gamma=params['gamma'], stbl=params['hc-type'])
                print("Iterating over the list of articles: ")
                for get_edits in [False, True]:
                    suffix_file_name = 'edited' if get_edits else 'original'
                    parser = PrepareArticles('', get_edits=get_edits, edit_ratio=edit_ratio, df_input=df_test)

                    print("Parsing document...")
                    chunks = None
                    try:
                        chunks = parser(combined=False)
                    except:
                        continue

                    print("Testing parsed document")

                    # Go over all the documents
                    for i in range(len(chunks['text'])):
                        results = {}
                        # Parse sample
                        res = detector(chunks['text'][i], chunks['context'][i], dashboard=False)

                        # Calculate variables
                        df = res['sentences']
                        df['tag'] = chunks['tag'][i]
                        df.loc[df.tag.isna(), 'tag'] = 'no edits'

                        len_valid = len(df[~df.pvalue.isna()])
                        edit_rate = np.mean(df['tag'] == '<edit>')
                        HC = res['HC']
                        HC_pvalue = HC_pval_func(len_valid, HC)[0][0]
                        dfr = df[df['mask']]
                        precision = np.mean(dfr['tag'] == '<edit>')
                        recall = np.sum((df['mask'] == True) & (df['tag'] == '<edit>')) / np.sum(df['tag'] == '<edit>')

                        # Save the results
                        name = f"article_{i}_sentences_{suffix_file_name}"
                        results[name] = dict(length=len_valid, edit_rate=edit_ratio, HC=res['HC'],
                                                HC_pvalue=HC_pvalue, precision=precision, recall=recall,
                                                fisher=res['fisher'], fisher_pvalue=res['fisher_pvalue'],
                                                minP=res['minP'], name=name)

                        print("Saving results")
                        dfr = pd.DataFrame.from_dict(results).T
                        dfr['fold'] = fold
                        dfr['edited'] = 1 if get_edits else 0
                        lst_df_results.append(dfr)

            print("Start analyzing results")
            # Combine the results from all the folds
            df_results_folds = pd.concat(lst_df_results, ignore_index=True)
            lst_folds_analyze = []
            for fold in tqdm(range(k_folds)):
                # Get results for current fold
                df_fold_results = df_results_folds[df_results_folds['fold']==fold].copy()
                df_fold_results.sort_values(by='name', inplace=True)

                # Split type of article
                df_original = df_fold_results[df_fold_results['edited'] == 0]
                df_edited = df_fold_results[df_fold_results['edited'] == 1]

                # Testing HC and minP
                for dist_type in ['HC', 'minP']:
                    results_obj = {
                        'topic': [topic],
                        'model': [lm_name],
                        'edit_ratio': [edit_ratio],
                        'number_sentence': [number_sentence],
                        'fold': [fold],
                        'statistics': [dist_type]
                    }

                    # Calculate thresholds and predictions
                    threshold = get_threshold(df_original, dist_type)
                    pred = df_edited[dist_type].apply(lambda x: 1 if x > threshold else 0)
                    TPR, accuracy = calc_metrics_power_analysis(pred)
                    
                    # Save it
                    results_obj['accuracy_at_FPR_005'] = accuracy
                    results_obj['TPR_at_FPR_005'] = TPR

                    lst_folds_analyze.append(pd.DataFrame(results_obj))

            df_folds_analyze = pd.concat(lst_folds_analyze, ignore_index=True)

            print("Save analyze results")
            # Analyze the results for the current topic/edit ratio
            # Testing HC and minP
            for dist_type in ['HC', 'minP']:
                df_current = df_folds_analyze[df_folds_analyze['statistics']==dist_type]

                results_obj = {
                    'topic': [topic],
                    'model': [lm_name],
                    'statistics': [dist_type],
                    'edit_ratio': [edit_ratio],
                    'number_sentence': [number_sentence],
                    'accuracy_at_FPR_005': [df_current['accuracy_at_FPR_005'].mean()],
                    'accuracy_at_FPR_005_std': [df_current['accuracy_at_FPR_005'].std()],
                    'TPR_at_FPR_005': [df_current['TPR_at_FPR_005'].mean()], 
                    'TPR_at_FPR_005_std': [df_current['TPR_at_FPR_005'].std()]
                }
                lst_df_global_results.append(pd.DataFrame(results_obj))

df_results = pd.concat(lst_df_global_results, ignore_index=True)
output_path = 'results_power_analysis.csv' if output_path == '' else output_path
df_results.to_csv(output_path, index=False)

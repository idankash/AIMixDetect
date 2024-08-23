import pandas as pd
import numpy as np

from tqdm import tqdm

import argparse
import yaml

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

def find_threhsold_at_FPR(df, dist_type, threshold_FPR_at, label='edited', steps=0.01):
    """
    Returns the best threshold with the highest accuracy on the calibration set under restricted FPR

    Args:
    :df :              Data frame with the data
    :dist_type:        The column to test
    :threshold_FPR_at: We want the FPR to be less than threshold_FPR_at on the calibration set 
    :label:            The true label of the sample
    :steps:            Steps to take for the threshold search

    Return:
        Returns the best threshold

    """
    # Find the threshold for FPR = threshold_FPR_at
    min_dist_type = df[dist_type].min()
    max_dist_type = df[dist_type].max()
    best_threshold = max_dist_type
    best_acc = 0
    
    # Go over the thresholds
    for threshold in np.arange(min_dist_type, max_dist_type, steps):
        y_score = df[dist_type].apply(lambda x: 1 if x >= threshold else 0) # Apply thresholds
        y_true = df[label]
        tn, fp, fn, tp = confusion_matrix(y_true, y_score).ravel() # Get confusion matrix
        TPR = fp / (fp + tn) # Calc FPR

        # FPR need to be less than threshold_FPR_at
        if TPR <= threshold_FPR_at: 
            # Accuracy formula
            acc = (tp + tn) / (tp + tn + fp + fn) 
            # Save the best param
            if acc > best_acc: 
                best_acc = acc
                best_threshold = threshold 
            
    return best_threshold

def find_threhsold_best_accuracy(df, dist_type, label='edited', steps=0.01):
    """
    Returns the best threshold with the highest accuracy on the calibration set

    Args:
    :df :              Data frame with the data
    :dist_type:        The column to test
    :label:            The true label of the sample
    :steps:            Steps to take for the threshold search

    Return:
        Returns the best threshold
    """

    min_dist_type = df[dist_type].min()
    max_dist_type = df[dist_type].max()
    best_threshold = max_dist_type
    best_acc = 0
    
    # Go over the thresholds
    for threshold in np.arange(min_dist_type, max_dist_type, steps):
        y_score = df[dist_type].apply(lambda x: 1 if x >= threshold else 0) # Apply thresholds
        y_true = df[label]
        tn, fp, fn, tp = confusion_matrix(y_true, y_score).ravel() # Get confusion matrix
        acc = (tp + tn) / (tp + tn + fp + fn) # Accuracy formula
        
        # Save the best param
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold 
            
    return best_threshold

def check_AUC_ROC(df, dist_type='HC', threshold_FPR_at=None, label='edited', steps=0.01):
    """
    Returns the best threshold with the highest accuracy on the calibration set

    Args:
    :df:               Data frame with the data
    :dist_type:        The column to test
    :threshold_FPR_at: We want the FPR to be less than threshold_FPR_at on the calibration set 
    :label:            The true label of the sample
    :steps:            Steps to take for the threshold search

    Return:
        Returns the best threshold
    """

    # Get dist_type values fot the edited/not_edited samples
    HC_edits = df[df[label] == 1][dist_type].values
    HC_original = df[df[label] == 0][dist_type].values
    
    # Create label
    y_test = [1 for _ in range(len(HC_edits))] + [0 for _ in range(len(HC_original))]
    y_prob = list(HC_edits) + list(HC_original)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Calculate the ROC curve
    if threshold_FPR_at == None:
        # Find the index of the point on the ROC curve closest to (0, 1)
        best_threshold_index = np.argmax(tpr - fpr)
        # Get the threshold corresponding to the point on the ROC curve
        best_threshold = thresholds[best_threshold_index]
    elif threshold_FPR_at == -1:
        best_threshold = find_threhsold_best_accuracy(df, dist_type, label=label, steps=steps)
    else:
        best_threshold = find_threhsold_at_FPR(df, dist_type, threshold_FPR_at, label=label, steps=steps)
    
    # Calculate the AUC-ROC score
    auc_roc = auc(fpr, tpr)
    return best_threshold, auc_roc

def calc_metrics(df, true_label, pred_label):
    """
    Calculate and return TPR, FPR, accuracy

    Args:
    :df:         Data frame with the data
    :true_label: The column to test
    :pred_label: The true label of the sample

    Return:
        Returns TPR, FPR, accuracy
    """

    TP = len(df[(df[pred_label]==1) & (df[true_label]==1)])
    FN = len(df[(df[pred_label]==0) & (df[true_label]==1)])
    FP = len(df[(df[pred_label]==1) & (df[true_label]==0)])
    TN = len(df[(df[pred_label]==0) & (df[true_label]==0)])

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TPR, FPR, accuracy

# Create argparse
parser = argparse.ArgumentParser(description="Read args to parse article")
parser.add_argument('-conf', type=str, help='configurations file', default="simulate_replication_conf.yml")

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
    for edit_ratio in edit_ratios:
        INPUT_FILE = f'{files_path}/{topic}/edit_ratio_{edit_ratio}/{topic}.csv' # Get the data file for the topic + edit ratio

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
        df_results_folds.to_csv('test.csv')
        lst_folds_analyze = []
        for fold in tqdm(range(k_folds)):
            # Get results for current fold
            df_fold_results = df_results_folds[df_results_folds['fold']==fold].copy()
            df_fold_results.sort_values(by='name', inplace=True)
            df_test = df_fold_results[:data_length // k_folds].copy() # The first data_length // k_folds are test data
            df = df_fold_results[data_length // k_folds:].copy() # The rest is for the calibration 

            # Testing HC and minP
            for dist_type in ['HC', 'minP']:
                results_obj = {
                    'topic': [topic],
                    'model': [lm_name],
                    'edit_ratio': [edit_ratio],
                    'fold': [fold],
                    'statistics': [dist_type]
                }

                best_Threshold_HC, auc_roc = check_AUC_ROC(df, dist_type=dist_type, threshold_FPR_at=None)
                best_Threshold_HC_005, auc_roc = check_AUC_ROC(df, dist_type=dist_type, threshold_FPR_at=0.05)
                best_Threshold_HC_optimal_acc, auc_roc = check_AUC_ROC(df, dist_type=dist_type, threshold_FPR_at=-1)

                # Use threshold
                df_test['pred'] = df_test[dist_type].apply(lambda x: 1 if x >= best_Threshold_HC else 0)
                df_test['pred_FPR_005'] = df_test[dist_type].apply(lambda x: 1 if x >= best_Threshold_HC_005 else 0)
                df_test['pred_optimal_acc'] = df_test[dist_type].apply(lambda x: 1 if x >= best_Threshold_HC_optimal_acc else 0)

                # Get results
                TPR, FPR, accuracy = calc_metrics(df_test, 'edited', 'pred')
                results_obj['TPR'] = [TPR]
                results_obj['FPR'] = [FPR]
                results_obj['accuracy'] = [accuracy]

                TPR, FPR, accuracy = calc_metrics(df_test, 'edited', 'pred_FPR_005')
                results_obj['TPR_005'] = [TPR]
                results_obj['FPR_005'] = [FPR]
                results_obj['accuracy_005'] = [accuracy]

                TPR, FPR, accuracy = calc_metrics(df_test, 'edited', 'pred_optimal_acc')
                results_obj['TPR_optimal_acc'] = [TPR]
                results_obj['FPR_optimal_acc'] = [FPR]
                results_obj['accuracy_optimal_acc'] = [accuracy]

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
                'accuracy': [df_current['accuracy_optimal_acc'].mean()],
                'accuracy_std': [df_current['accuracy_optimal_acc'].std()],
                'FPR': [df_current['FPR_optimal_acc'].mean()], 
                'FPR_std': [df_current['FPR_optimal_acc'].std()], 
                'accuracy_005': [df_current['accuracy_005'].mean()], 
                'accuracy_005_std': [df_current['accuracy_005'].std()]
            }
            lst_df_global_results.append(pd.DataFrame(results_obj))

df_results = pd.concat(lst_df_global_results, ignore_index=True)
output_path = 'results.csv' if output_path == '' else output_path
df_results.to_csv(output_path, index=False)

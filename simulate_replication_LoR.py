import pandas as pd
import numpy as np

from tqdm import tqdm

import argparse
import yaml

from sklearn.metrics import roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression

def calculate_fpr(y, pred):
    # Calculate True Negatives (TN) and False Positives (FP)
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    
    # Calculate False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    return fpr

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

SEED = 555

# Save the results to output later. For each topic + edit_ratio we save the accuracy, FPR, accuracy@FPR@0.05, std_accuracy, std_accuracy_005
lst_df_global_results = []
data_length = 0

# Run over the data
for topic in topics:
    for edit_ratio in edit_ratios:
        print(f'Start Testing topic: {topic} edit_ratio: {edit_ratio}')
        file_edited_path = f'{files_path}/{topic}/edit_ratio_{edit_ratio}/{topic}_embedded_edited.csv'
        file_not_edited_path = f'{files_path}//{topic}/edit_ratio_{edit_ratio}/{topic}_embedded_not_edited.csv'
        
        df_edited = pd.read_csv(file_edited_path)
        df_not_edited = pd.read_csv(file_not_edited_path)
        
        # Add label
        df_edited['edited'] = 1
        df_not_edited['edited'] = 0

        dim_columns = [column for column in df_edited.columns if 'dim_' in column]
        results_df = pd.DataFrame(columns=['topic', 'model', 'edit_ratio', 'fold', 'accuracy', 'accuracy_at_FPR_005'])

        # Get splits
        splits = [i for i in range(0, len(df_edited), len(df_edited) // k_folds)]

        # Do it k_folds times
        for fold in tqdm(range(k_folds)):
            # Create the split
            df_null = None
            df_test = None
            current_window_size = window_size

            for i in range(len(splits) - 1):
                temp_df = pd.concat([df_edited.iloc[splits[i]:splits[i+1]], df_not_edited.iloc[splits[i]:splits[i+1]]], ignore_index=True)
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

            if df_test is None:
                continue

            # Use the train split
            df_null = df_null.sample(frac=1, random_state=SEED)
            df_test = df_test.sample(frac=1, random_state=SEED)

            df_val = df_null[int(len(df_null) * train_split):]                
            df_null = df_null[:int(len(df_null) * train_split)]

            best_parameters = {
                'topic': [topic], 
                'model': ['logistic_regression'],
                'edit_ratio': [edit_ratio],
                'fold': [fold],
                'accuracy': [0]
            }

            model = LogisticRegression(random_state=SEED)
            model.fit(df_null[dim_columns], df_null['edited'].values)

            pred = model.predict_proba(df_val[dim_columns])[:, 1]

            # Compute the ROC curve
            fpr, tpr, thresholds = roc_curve(df_val['edited'].values, pred)

            # Find the threshold that corresponds to the desired FPR (0.05)
            desired_fpr = 0.05
            threshold = thresholds[np.where(fpr <= desired_fpr)[0][-1]]

            pred = model.predict_proba(df_test[dim_columns])[:, 1]

            # Apply the threshold to get final predictions
            pred_adjusted = (pred >= threshold).astype(int)

            acc_at_fpr_005 = accuracy_score(df_test['edited'].values, pred_adjusted)

            pred = model.predict(df_test[dim_columns])
            acc = accuracy_score(df_test['edited'].values, pred)
            fpr = calculate_fpr(df_test['edited'].values, pred)

            if acc > best_parameters['accuracy'][0]:
                best_parameters['accuracy'] = [acc]
                best_parameters['FPR'] = [fpr]
                best_parameters['accuracy_at_FPR_005'] = [acc_at_fpr_005]

            results_df = pd.concat([results_df, pd.DataFrame(best_parameters)])

        results_obj = {
            'topic': [topic],
            'model': ['logistic_regression'],
            'statistics': ['embedding'],
            'edit_ratio': [edit_ratio],
            'accuracy': [results_df['accuracy'].mean()],
            'accuracy_std': [results_df['accuracy'].std()], 
            'FPR': [results_df['FPR'].mean()],
            'FPR_std': [results_df['FPR'].std()],
            'accuracy_005': [results_df['accuracy_at_FPR_005'].mean()],
            'accuracy_005_std': [results_df['accuracy_at_FPR_005'].std()]
        }
        lst_df_global_results.append(pd.DataFrame(results_obj))

df_results = pd.concat(lst_df_global_results, ignore_index=True)
output_path = 'results_LoR.csv' if output_path == '' else output_path
df_results.to_csv(output_path, index=False)

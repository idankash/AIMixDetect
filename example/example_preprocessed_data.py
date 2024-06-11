
"""
This script is an example of how to use the DetectLM class.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM    

import sys
import numpy as np
sys.path.append('./')
from src.PerplexityEvaluator import PerplexityEvaluator
from src.DetectLM import DetectLM
from src.PrepareSentenceContext import PrepareSentenceContext
from src.PrepareArticles import PrepareArticles
from src.fit_survival_function import fit_per_length_survival_function
import pickle
from tqdm import tqdm

INPUT_FILE = 'D:\\.Idan\\תואר שני\\תזה\\detectLM\\example\\articles_dataset_null_data.csv'
INPUT_FILE_FIT_SURVIVAL = 'D:\\.Idan\\תואר שני\\תזה\\detectLM\\example\\articles_dataset_null_data_fit_survival_function.csv'
LOGLOSS_PVAL_FUNC_FILE = 'D:\.Idan\תואר שני\תזה\detectLM\example\logloss_pval_function.pkl'
HC_PVAL_FUNC_FILE = 'D:\.Idan\תואר שני\תזה\detectLM\example\HC_pval_function.pkl'
OUTPUT_FILE = 'article_null_small.csv'

# Load the logloss p-value function. Ususally one must fit this function using triaining data
# from the null class and ``fit_survival_function``.
# Here we load a pre-fitted function for the GPT-2 language model under Wikipedia-Introduction 
# dataset and no context.

# with open(LOGLOSS_PVAL_FUNC_FILE, 'rb') as f:
#     pval_function = pickle.load(f)


# Initialize LoglossEvaluator with a language model and a tokenizer
lm_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(lm_name)

sentence_detector = PerplexityEvaluator(AutoModelForCausalLM.from_pretrained(lm_name),
                    AutoTokenizer.from_pretrained(lm_name))

parse_chunks = PrepareArticles(INPUT_FILE_FIT_SURVIVAL)
chunks = parse_chunks()
responses = []
for sent in tqdm(chunks['text']):
    responses.append(sentence_detector(sent))

pval_function = fit_per_length_survival_function(chunks['length'], np.array(responses))

# initialize the detector...
detector = DetectLM(sentence_detector, pval_function,
                    min_len=8, max_len=50, length_limit_policy='truncate')

# Read preprocessed data
parse_chunks = PrepareArticles(INPUT_FILE)
chunks = parse_chunks()

# Test document
res = detector(chunks['text'], chunks['context'])
res['sentences']['length'] = chunks['length'] # Adding length
print(res['sentences'])

print("Save res['sentences']")
res['sentences'].to_csv(OUTPUT_FILE)

with open(HC_PVAL_FUNC_FILE, "rb") as f:
    HC_survival_function = pickle.load(f)

num_valid_sentences = res['sentences']['pvalue'].dropna().shape[0]
print(f"HC P-value = {HC_survival_function(num_valid_sentences, res['HC'])[0][0]}")

"""
'HC': 1.3396600337668725, 'fisher': 20.49921190930749, 'fisher_pvalue': 0.02486927492187124}
HC P-value = 0.1065159
"""

"""
This script is an example of how to use the DetectLM class.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM    

import sys
sys.path.append('./')
from src.PerplexityEvaluator import PerplexityEvaluator
from src.DetectLM import DetectLM
from src.PrepareSentenceContext import PrepareSentenceContext
import pickle

INPUT_FILE = 'D:\.Idan\תואר שני\תזה\detectLM\example\example_text.txt'
LOGLOSS_PVAL_FUNC_FILE = 'D:\.Idan\תואר שני\תזה\detectLM\example\logloss_pval_function.pkl'
HC_PVAL_FUNC_FILE = 'D:\.Idan\תואר שני\תזה\detectLM\example\HC_pval_function.pkl'

# Load the logloss p-value function. Ususally one must fit this function using triaining data
# from the null class and ``fit_survival_function``.
# Here we load a pre-fitted function for the GPT-2 language model under Wikipedia-Introduction 
# dataset and no context.
with open(LOGLOSS_PVAL_FUNC_FILE, 'rb') as f:
    pval_function = pickle.load(f)

# Initialize LoglossEvaluator with a language model and a tokenizer
lm_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(lm_name)

sentence_detector = PerplexityEvaluator(AutoModelForCausalLM.from_pretrained(lm_name),
                    AutoTokenizer.from_pretrained(lm_name))

# initialize the detector...
detector = DetectLM(sentence_detector, pval_function,
                    min_len=8, max_len=50, length_limit_policy='truncate')

# parse text from an input file 
with open(INPUT_FILE, 'rt') as f:
    text = f.read()
parse_chunks = PrepareSentenceContext(context_policy=None)
chunks = parse_chunks(text)

# Test document
res = detector(chunks['text'], chunks['context'])
print(res['sentences'])

with open(HC_PVAL_FUNC_FILE, "rb") as f:
    HC_survival_function = pickle.load(f)

num_valid_sentences = res['sentences']['pvalue'].dropna().shape[0]
print(f"HC P-value = {HC_survival_function(num_valid_sentences, res['HC'])[0][0]}")

"""
'HC': 1.3396600337668725, 'fisher': 20.49921190930749, 'fisher_pvalue': 0.02486927492187124}
HC P-value = 0.1065159
"""
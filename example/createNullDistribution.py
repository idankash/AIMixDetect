"""
This script is designed to create the null distribution from a given csv file containing articles.
Where each article in the csv looks like this object. 
"sentence" is the GLM sentence and "alternative" is the human sentence/edited sentence:

{
	"title": ""Aldous Huxley", 
	"sub_titles"": [{
		"sub_title": "Early life", 
		"sentences": [{
			"sentence": "Aldous Leonard Huxley was born on July 26, 1894, in Godalming, a picturesque market town in Surrey, England."", 
			"alternative": "As a child, Huxley's nickname was \""Ogie\"", short for \""Ogre\"".""
		}]
	}]
}

"""

from transformers import AutoTokenizer, AutoModelForCausalLM    
import torch

import sys
import numpy as np
import pandas as pd
sys.path.append('./')
from src.PerplexityEvaluator import PerplexityEvaluator
from src.DetectLM import DetectLM
from src.PrepareSentenceContext import PrepareSentenceContext
from src.PrepareArticles import PrepareArticles
from tqdm import tqdm

INPUT_FILE = '/home/treuser1/Idan-detectLM/SecondDataset/WikiDataset/200_sentences/20/model_name_wiki_intro_null.csv' 
OUTPUT_FILE = '/home/treuser1/Idan-detectLM/SecondDataset/WikiDataset/200_sentences/20/model_name_wiki_intro_null_sentences_phi2.csv' # _phi2

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
        
print(device)

# Initialize LoglossEvaluator with a language model and a tokenizer
lm_name = "microsoft/phi-2" #"gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(lm_name, force_download=True)
model = AutoModelForCausalLM.from_pretrained(lm_name).to(device)

sentence_detector = PerplexityEvaluator(model,tokenizer)

parse_chunks = PrepareArticles(INPUT_FILE)
chunks = parse_chunks()
responses = []
for sent in tqdm(chunks['text']):
    responses.append(sentence_detector(sent))

df = pd.DataFrame()
df['text'] = chunks['text']
df['response'] = np.array(responses)
df['length'] = chunks['length']
df.to_csv(OUTPUT_FILE)
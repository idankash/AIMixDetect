import torch
import pandas as pd
import json
import re

class PerplexityEvaluator(object):
    def __init__(self, model, tokenizer, ignore_index=-1, cache_logloss_path=''):
        self.model = model
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        
        # Idan 26/05/204
        self.cache_logloss_path = cache_logloss_path 
        try:
            # Load the dictionary from the file
            with open(self.cache_logloss_path, 'r') as file:
                self.cache_logloss = json.load(file)
        except:
            print('Could not find cache file')
            self.cache_logloss = None
            
    def clean_string(self, s):
        # Remove escape characters
        s = re.sub(r'\\[nrt]', '', s)
        # Strip leading and trailing spaces and quotes
        s = s.strip().strip("'")
        # Convert to lower case
        return s.lower()

    def __call__(self, text, context=None):
        # Try getting logloss from cache
        sentence_response = self._get_logloss_cache(self.clean_string(text))
        if sentence_response != None:
            return sentence_response
        
        sentence_response = self.log_perplexity(text, context)
        return sentence_response
    
    def _get_logloss_cache(self, sent: str) -> float:
        sent = sent.strip()
        if self.cache_logloss is None: return None
        if sent not in self.cache_logloss: return None
        return self.cache_logloss[sent]

    def log_perplexity(self, text, context=None):
        """
        Evaluate log perplexity of text with respect to the language model
        based on the context

        :param text:
        :param context:
        :return:
        """
        device = self.model.device
        text_ids = self.tokenizer(text, return_tensors='pt')
        if context:
            context_ids = self.tokenizer(context, return_tensors='pt')
            input_ids = torch.concatenate([context_ids['input_ids'], text_ids['input_ids']], axis=1)
            labels = torch.concatenate([torch.ones_like(context_ids['input_ids']) * self.ignore_index,
                                        text_ids['input_ids']], axis=1)
            print("Warning, need to remove context length when reporting lppx")
        else:
            input_ids = text_ids['input_ids']
            labels = input_ids

        loss = self.model(input_ids=input_ids.to(device), labels=labels.to(device)).loss
        return loss.cpu().detach().numpy()
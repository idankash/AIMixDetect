import pandas as pd
import numpy as np
import json

class PrepareArticles(object):
    """
    Parse preprocessed data from csv
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

    This information is needed for evaluating log-perplexity of the text with respect to a language model
    and later on to test the likelihood that the sentence was sampled from the model with the relevant context.
    """
    def __init__(self, input_file, get_edits=False, edit_ratio=None, min_tokens=10, max_tokens=100, max_sentences=None, df_input=None):
        self.input_file = input_file
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.get_edits = get_edits
        self.max_sentences = max_sentences
        self.df_input = df_input
        self.edit_ratio = edit_ratio

    def __call__(self, combined=True):
        return self.parse_dataset(combined)
    
    def parse_dataset(self, combined=True):
        if self.df_input is not None:
            articles_dataset = self.df_input
        else:
            articles_dataset = pd.read_csv(self.input_file)

        texts = []
        lengths = []
        contexts = []
        tags = []

        # For each article
        for i in range(articles_dataset.shape[0]):
            article_str = articles_dataset.iloc[i]['article_json']
            article_obj = json.loads(article_str) # Load article json
            
            number_of_sentences = article_str.count('sentence":')
            if self.edit_ratio is not None:
                num_of_edits = number_of_sentences * self.edit_ratio // 1
            else:
                num_of_edits = article_str.count('alternative":')
            
            current_texts = []
            current_lengths = []
            current_contexts = []
            current_tags = []
            exceeded_max_sentences = False
            
            for sub_title in article_obj['sub_titles']: # For each sub title
                for sentence in sub_title['sentences']: # Go over each sentence
                    sentence_size = len(sentence['sentence'].split())
                    if sentence_size >= self.min_tokens and sentence_size <= self.max_tokens:
                        current_texts.append(sentence['sentence'])
                        current_lengths.append(len(sentence['sentence'].split())) # Number of tokens
                        current_contexts.append(sentence['context'] if 'context' in sentence else None)
                        current_tags.append('no edits')

                    # If get_edits and has edited sentence save it
                    if self.get_edits and num_of_edits > 0 and 'alternative' in sentence and len(sentence['alternative'].split()) >= self.min_tokens and len(sentence['alternative'].split()) <= self.max_tokens:
                        current_texts.append(sentence['alternative'])
                        current_lengths.append(len(sentence['alternative'].split()))
                        current_contexts.append(sentence['alternative_context'] if 'alternative_context' in sentence else None)
                        current_tags.append('<edit>')
                        num_of_edits -= 1
                    if self.max_sentences and len(current_texts) >= self.max_sentences:
                        exceeded_max_sentences = True
                        break
                        # return {'text': np.array(texts, dtype=object), 'length': np.array(lengths, dtype=object), 'context': np.array(contexts, dtype=object), 'tag': np.array(tags, dtype=object),
                        #             'number_in_par': np.arange(1,1+len(texts))}
                if exceeded_max_sentences:
                    break
            
            # If exceede max sentences only if self.max_sentences is not None
            if (self.max_sentences and exceeded_max_sentences) or (not self.max_sentences):
                # If combined, combine the data
                if combined:
                    texts = texts + current_texts
                    lengths = lengths + current_lengths
                    contexts = contexts + current_contexts
                    tags = tags + current_tags
                else:
                    texts.append(np.array(current_texts))
                    lengths.append(np.array(current_lengths))
                    contexts.append(np.array(current_contexts))
                    tags.append(np.array(current_tags))
            
        return {'text': np.array(texts, dtype=object), 'length': np.array(lengths, dtype=object), 'context': np.array(contexts, dtype=object), 'tag': np.array(tags, dtype=object),
                    'number_in_par': np.arange(1,1+len(texts))}


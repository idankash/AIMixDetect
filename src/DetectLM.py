import numpy as np
import pandas as pd
from multitest import MultiTest
from tqdm import tqdm
import logging
import json
import re


def truncae_to_max_no_tokens(text, max_no_tokens):
    return " ".join(text.split()[:max_no_tokens])


class DetectLM(object):
    def __init__(self, sentence_detection_function, survival_function_per_length,
                 min_len=4, max_len=100, HC_type="stbl",
                 length_limit_policy='truncate', ignore_first_sentence=False, cache_logloss_path=''):
        """
        Test for the presence of sentences of irregular origin as reflected by the
        sentence_detection_function. The test is based on the sentence detection function
        and the P-values obtained from the survival function of the detector's responses.

        Args:
        ----
            :sentence_detection_function:  a function returning the response of the text 
            under the detector. Typically, the response is a logloss value under some language model.
            :survival_function_per_length:  survival_function_per_length(l, x) is the probability of the language
            model to produce a sentence value as extreme as x or more when the sentence s is the input to
            the detector. The function is defined for every sentence length l.
            The detector can also recieve a context c, in which case the input is the pair (s, c).
            :length_limit_policy: When a sentence exceeds ``max_len``, we can:
                'truncate':  truncate sentence to the maximal length :max_len
                'ignore':  do not evaluate the response and P-value for this sentence
                'max_available':  use the logloss function of the maximal available length
            :ignore_first_sentence:  whether to ignore the first sentence in the document or not. Useful when assuming
                context of the form previous sentence.
            :cache_logloss_path: cache dict to restore the logloss faster
        """

        self.survival_function_per_length = survival_function_per_length
        self.sentence_detector = sentence_detection_function
        self.min_len = min_len
        self.max_len = max_len
        self.length_limit_policy = length_limit_policy
        self.ignore_first_sentence = ignore_first_sentence
        self.HC_stbl = True if HC_type == 'stbl' else False
        
        # Idan 26/05/204
        self.cache_logloss_path = cache_logloss_path
        try:
            # Load the dictionary from the file
            with open(self.cache_logloss_path, 'r') as file:
                self.cache_logloss = json.load(file)
        except:
            print('Could not find cache file')
            self.cache_logloss = None

    def _logperp(self, sent: str, context=None) -> float:
        return float(self.sentence_detector(sent, context))

    def _test_sentence(self, sentence: str, context=None):
        return self._logperp(sentence, context)
    
    def _get_length(self, sentence: str):
        return len(sentence.split())

    def _test_response(self, response: float, length: int):
        """
        Args:
            response:  sentence logloss
            length:    sentence length in tokens

        Returns:
          pvals:    P-value of the logloss of the sentence
          comments: comment on the P-value
        """
        if self.min_len <= length:
            comment = "OK"
            if length > self.max_len:  # in case length exceeds specifications...
                if self.length_limit_policy == 'truncate':
                    length = self.max_len
                    comment = f"truncated to {self.max_len} tokens"
                elif self.length_limit_policy == 'ignore':
                    comment = "ignored (above maximum limit)"
                    return np.nan, np.nan, comment
                elif self.length_limit_policy == 'max_available':
                    comment = "exceeding length limit; resorting to max-available length"
                    length = self.max_len
            pval = self.survival_function_per_length(length, response)
            assert pval >= 0, "Negative P-value. Something is wrong."
            return dict(response=response, 
                        pvalue=pval, 
                        length=length,
                        comment=comment)
        else:
            comment = "ignored (below minimal length)"
            return dict(response=response, 
                        pvalue=np.nan, 
                        length=length,
                        comment=comment)

    def _get_pvals(self, responses: list, lengths: list) -> tuple:
        pvals = []
        comments = []
        for response, length in zip(responses, lengths):
            r = self._test_response(response, length)
            pvals.append(float(r['pvalue']))
            comments.append(r['comment'])
        return pvals, comments
    
    def clean_string(self, s):
        # Remove escape characters
        s = re.sub(r'\\[nrt]', '', s)
        # Strip leading and trailing spaces and quotes
        s = s.strip().strip("'")
        # Convert to lower case
        return s.lower()
    
    def _get_logloss_cache(self, sent: str) -> float:
        sent = sent.strip()
        if self.cache_logloss is None: return None
        if sent not in self.cache_logloss: return None
        return self.cache_logloss[sent]

    def _get_responses(self, sentences: list, contexts: list) -> list:
        """
        Compute response and length of a text sentence 
        """
        assert len(sentences) == len(contexts)

        responses = []
        lengths = []
        for sent, ctx in tqdm(zip(sentences, contexts)):
            logging.debug(f"Testing sentence: {sent} | context: {ctx}")
            length = self._get_length(sent)
            if self.length_limit_policy == 'truncate':
                sent = truncae_to_max_no_tokens(sent, self.max_len)
            if length == 1:
                logging.warning(f"Sentence {sent} is too short. Skipping.")
                responses.append(np.nan)
                continue
            try:
                # Try getting logloss from cache
                sentence_response = self._get_logloss_cache(self.clean_string(sent))
                if sentence_response != None:
                    responses.append(sentence_response)
                else: # If sentence not found
                    current_response = self._test_sentence(sent, ctx)
                    responses.append(current_response)
            except:
                # something unusual happened...
                import pdb; pdb.set_trace()
            lengths.append(length)
            
        return responses, lengths

    def get_pvals(self, sentences: list, contexts: list) -> tuple:
        """
        logloss test of every (sentence, context) pair
        """
        assert len(sentences) == len(contexts)

        responses, lengths = self._get_responses(sentences, contexts)
        pvals, comments = self._get_pvals(responses, lengths)
        
        return pvals, responses, comments


    def testHC(self, sentences: list) -> float:
        pvals = np.array(self.get_pvals(sentences)[1])
        mt = MultiTest(pvals, stbl=self.HC_stbl)
        return mt.hc(gamma=0.4)[0]

    def testFisher(self, sentences: list) -> dict:
        pvals = np.array(self.get_pvals(sentences)[1])
        print(pvals)
        mt = MultiTest(pvals, stbl=self.HC_stbl)
        return dict(zip(['Fn', 'pvalue'], mt.fisher()))

    def _test_chunked_doc(self, lo_chunks: list, lo_contexts: list) -> tuple:
        pvals, responses, comments = self.get_pvals(lo_chunks, lo_contexts)
        if self.ignore_first_sentence:
            pvals[0] = np.nan
            logging.info('Ignoring the first sentence.')
            comments[0] = "ignored (first sentence)"
        
        df = pd.DataFrame({'sentence': lo_chunks, 'response': responses, 'pvalue': pvals,
                           'context': lo_contexts, 'comment': comments},
                          index=range(len(lo_chunks)))
        df_test = df[~df.pvalue.isna()]
        if df_test.empty:
            logging.warning('No valid chunks to test.')
            return None, df
        return MultiTest(df_test.pvalue, stbl=self.HC_stbl), df

    def test_chunked_doc(self, lo_chunks: list, lo_contexts: list, dashboard=False) -> dict:
        mt, df = self._test_chunked_doc(lo_chunks, lo_contexts)
        if mt is None:
            hc = np.nan
            fisher = (np.nan, np.nan)
            berk_jones = (np.nan, np.nan)
            df['mask'] = pd.NA
        else:
            hc, hct = mt.hc(gamma=0.4)
            fisher = mt.fisher()
            berk_jones = mt.berk_jones()
            print("berk_jones")
            print(berk_jones)
            df['mask'] = df['pvalue'] <= hct
        if dashboard:
            mt.hc_dashboard(gamma=0.4)
        return dict(sentences=df, HC=hc, fisher=fisher[0], fisher_pvalue=fisher[1]
                    , berk_jones=berk_jones)

    def __call__(self, lo_chunks: list, lo_contexts: list, dashboard=False) -> dict:
        return self.test_chunked_doc(lo_chunks, lo_contexts, dashboard=dashboard)
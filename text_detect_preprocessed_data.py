"""
This script is designed to take a precomputed null distribution and run the pipeline.

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
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import numpy as np
import argparse
from src.DetectLM import DetectLM
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from src.PrepareArticles import PrepareArticles
from src.fit_survival_function import fit_per_length_survival_function
from src.HC_survival_function import get_HC_survival_function
from glob import glob
import pathlib
import yaml
from pathlib import Path
import re


logging.basicConfig(level=logging.INFO)


def read_all_csv_files(pattern):
    df = pd.DataFrame()
    print(pattern)
    for f in glob(pattern):
        df = pd.concat([df, pd.read_csv(f)])
    return df


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


def mark_edits_remove_tags(chunks, tag="edit"):
    text_chunks = chunks['text']
    edits = []
    for i,text in enumerate(text_chunks):
        chunk_text = re.findall(rf"<{tag}>(.+)</{tag}>", text)
        if len(chunk_text) > 0:
            import pdb; pdb.set_trace()
            chunks['text'][i] = chunk_text[0]
            chunks['length'][i] -= 2
            edits.append(True)
        else:
            edits.append(False)

    return chunks, edits

def main():
    # Set inputs
    parser = argparse.ArgumentParser(description="Apply detector of non-GLM text to a text file or several text files (based on an input pattern)")
    parser.add_argument('-i', type=str, help='input regex', default="Data/ChatGPT/*.txt")
    parser.add_argument('-o', type=str, help='output folder', default="results/")
    parser.add_argument('-result-file', type=str, help='where to write results', default="out.csv")
    parser.add_argument('-conf', type=str, help='configurations file', default="conf.yml")
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--leave-out', action='store_true')
    
    args = parser.parse_args()
    
    # Read conf.yaml
    with open(args.conf, "r", encoding='utf8') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print("context = ", args.context)

    if args.context:
        null_data_file = params['context-null-data-file']
    else:
        null_data_file = params['no-context-null-data-file']
    lm_name = params['language-model-name']

    if not args.leave_out:
        logging.info(f"Fitting null log-loss survival function using data from {null_data_file}.")
        logging.info(f"Please verify that the data was obtained under the same context policy.")

        df_null = read_all_csv_files(null_data_file)
        print(df_null)
        if params['ignore-first-sentence'] and 'num' in df_null.columns:
            df_null = df_null[df_null.num > 1]
        logging.info(f"Found {len(df_null)} log-loss values of text atoms in {null_data_file}.")
        pval_functions = get_survival_function(df_null, G=params['number-of-interpolation-points'])

    max_tokens_per_sentence = params['max-tokens-per-sentence']
    min_tokens_per_sentence = params['min-tokens-per-sentence']
    SENTENCES_LOGLOSS = params['sentences-logloss']

    logging.info(f"Loading Language model {lm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModelForCausalLM.from_pretrained(lm_name)
    logging.info(f"Loading LPPT evaluator...")
    sentence_detector = PerplexityEvaluator(model, tokenizer)

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    print(device)
    model.to(device)

    if args.context:
        context_policy = 'previous_sentence'
    else:
        context_policy = None

    if not args.leave_out:
        
        logging.debug("Initializing detector...")
        detector = DetectLM(sentence_detector, pval_functions,
                            min_len=min_tokens_per_sentence,
                            max_len=max_tokens_per_sentence,
                            length_limit_policy='truncate',
                            HC_type=params['hc-type'],
                            ignore_first_sentence=
                            True if context_policy == 'previous_sentence' else False,
                            cache_logloss_path=SENTENCES_LOGLOSS
                            )

    HC_pval_func = get_HC_survival_function(HC_null_sim_file=None)
    pattern = args.i
    output_folder = args.o

    print("Iterating over the list of articles: ")
    
    INPUT_FILE = params['input-file']
    OUTPUT_PATH = params['output-path']
    logging.info(f"Parsing document {INPUT_FILE}...")

#     get_edits = True
    for get_edits in [False, True]:
        suffix_file_name = 'edited' if get_edits else 'original'
        parser = PrepareArticles(INPUT_FILE, get_edits=get_edits)

        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Parsing document...")
        chunks = parser(combined=False)

        logging.info("Testing parsed document")
        # Go over all the documents
        for i in range(len(chunks['text'])):
            results = {}
            res = detector(chunks['text'][i], chunks['context'][i], dashboard=args.dashboard)
            logging.basicConfig(level=logging.INFO)

            df = res['sentences']
            df['tag'] = chunks['tag'][i]
            df.loc[df.tag.isna(), 'tag'] = 'no edits'

            name = f"article_{i}_sentences_{suffix_file_name}"#Path(INPUT_FILE).stem
            output_file = f"{OUTPUT_PATH}{name}.csv"
            print("Saving sentences to ", output_file)
            df.to_csv(output_file)
            df['pvalue'].hist

            print(df.groupby('tag').response.mean())
            print(df[df['mask']])
            len_valid = len(df[~df.pvalue.isna()])
            print("Length valid: ", len_valid)
            edit_rate = np.mean(df['tag'] == '<edit>')
            print(f"Num of Edits (rate) = {np.sum(df['tag'] == '<edit>')} ({edit_rate})")
            HC = res['HC']
            print(f"HC = {res['HC']}")
            HC_pvalue = HC_pval_func(len_valid, HC)[0][0]
            print(f"Pvalue (HC) = {HC_pvalue}")
            print(f"Fisher = {res['fisher']}")
            print(f"Fisher (chisquared pvalue) = {res['fisher_pvalue']}")
            dfr = df[df['mask']]
            precision = np.mean(dfr['tag'] == '<edit>')
            recall = np.sum((df['mask'] == True) & (df['tag'] == '<edit>')) / np.sum(df['tag'] == '<edit>')
            print("Precision = ", precision)
            print("recall = ", recall)
            print("F1 = ", 2 * precision*recall / (precision + recall))

            results[name] = dict(length=len_valid, edit_rate=edit_rate, HC=res['HC'],
                                    HC_pvalue=HC_pvalue, precision=precision, recall=recall,
                                    fisher=res['fisher'], fisher_pvalue=res['fisher_pvalue'],
                                    berk_jones=res['berk_jones'])

            #plt.title("Hisogram of P-values")
            #plt.savefig("pvalue_hist.png")
            #plt.show()

            results_filename = f"{OUTPUT_PATH}article_{i}_{suffix_file_name}_results.csv"#args.result_file
            print(results)
            print(f"Saving results to {results_filename}")
            dfr = pd.DataFrame.from_dict(results).T
            dfr.to_csv(results_filename)


if __name__ == '__main__':
    main()
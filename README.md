# AIMixDetect: detect mixed authorship of a language model (LM) and humans

## Pipeline

### Initialization
In initialization, we tune the detector to a specific language model of known weights, specific texts generated by AI 
(preferably the actual model) considered as null data, and a specific context policy (what is used as a context for generation). 
The main ingredient we tune is the function with which we get P-values, i.e. the survival function of log-perplexity 
of AI-generated texts, we assume to have. For better results, also tune the survival function of Higher Criticism and
Fisher's combination statistic associated with full documents.        

#### The ingredients:
1. Causal language model. A mapping from context text to next tokens probabilities.
2. "Null data". Texts generated by the model or a related language model (You can create it on the fly using ``text_detect_create_null_and_run.py``).
3. CSV file containing the articles for the "Null data" and for testing. You can find an example in ``example\example_null_article.csv``.
4. If it exists, a dictionary of sentences and logloss to use as a cache to make the process faster.
 
#### The steps:
1. Evaluate the log-perplexity of individual sentences (or larger chunks) given a context policy. Here we want to record 
the length of every sentence. Use ``many_atomic_detections.py`` for this step. 
2. Fit the survival function to the resulting log-perplexity, while taking into account the length (different function to 
each length). Check ``fit_pvalue_function.py`` to understand this step.   

### Inference
Here we apply the detector to a new document. The detector assigns a P-value to every sentence, returning the
statistics: Higher Criticism (HC) and Fisher's combination test. Under the null, the distribution of HC can be simulated
from the null data or via sampling P-values from the uniform distribution. For Fisher's combination statistic, you can 
use the chi-squared distribution or characterize the null by simulating from the null data. 

#### Ingredients:
1. Causal language model. A mapping from context text to next tokens probabilities.
2. Survival function of log-perplexity to evaluate per-sentence P-values. This function must match the context policy.
3. If it exists, a dictionary of sentences and logloss to use as a cache to make the process faster.

#### The steps:
1. Initialize a detector (``DetectLM``) with a log-perplexity function (``PerplexityEvaluator``) and a corresponding 
P-value function. 
2. Given a document, convert it to a list of (sentence, context) pairs (``PrepareArticles``) 
3. Pass these pairs to the detector, which will return:
    (a) a list of sentences and their P-values.
    (b) HC test statistic
    (c) Fisher's test statistic
    (d) Chi-squared test P-value for Fisher's statistic 
4. If HC or Fisher's tests are too large, report "some sentences are not model-generated" and identify those sentences by
the ``mask`` column. Additional calibration may be needed to determine what is "too large".

## Modules
- ``PerplexityEvaluator``. Evaluate the log-perplexity (aka log-loss) of a (text, context) pair with respect to a given 
language model. For initialization, you'll need to provide a ``Huggingface`` tokenizer and model like 
``AutoTokenizer`` and ``AutoModelForCausalLM``.
- ``PrepareSentenceContext``. This module breaks a document into sentences and assigns a context to every sentence. For 
example, the context can be ``None`` or the previous sentence or the title.
- ``PrepareArticles``. This module is designed to read a CSV file with articles, parse it, and return a list of sentences, corresponding lengths, context, and tags. 
- ``DetectLM``. Given a list of sentences and possible contexts, returns HC test and Fisher test statistics indicating
whether the document contains parts written not by the model (large values of HC or Fisher indicate the involvement of a 
human). The result is obtained by applying ``PerplexityEvaluator`` to every (sentence, context) pair.
To initialize ``DetectLM``, you will need to provide a function to evaluate P-values. Typically, this function is 
evaluated by fitting a curve to the empirical survival function of the perplexity of sentences under the model with a given 
context policy. It is also a good idea to take into account the length of the sentence because longer sentences tend to 
have smaller perplexity.    

## Scripts
- ``many_atomic_detections.py``. Evaluate the log perplexity of many sentences given a specific policy. This script can 
be useful to characterize P-value function or to analyze the power of the pipeline against a mixture from a specific 
domain.
- ``text_detect_preprocessed_data.py``. Apply the full testing pipeline to an input text file. This script loads "null data" and fits a function to evaluate P-values. To obtain reliable detection, the null data must be obtained under the same context policy of the test.
- ``text_detect_create_null_and_run.py``. Apply the full testing pipeline to an input text file. This script creates the "null data" and fits a function to evaluate P-values. To obtain reliable detection.
- ``text_detect_cross_validation.py``. Apply the full testing pipeline with a cross-validation method. This script creates the "null data" and fits a function to evaluate P-values for each fold. To obtain reliable detection.
- ``fit_survival_function.py``. Report on the histogram of simulated null logloss (log-perplexities) and the dependency of the logloss in the sentence's length. 
- ``HC_survival_function.py``. Computes the survival function of Higher Criticism of uniformly distributed P-values for several numbers of P-values. The computation is based
on stored simulated values. If stored values are not found in the specified location, the 
script simulates the values from scratch and stores the results. 

## Example
``
    from text_detect import get_survival_function
    from transformers import AutoTokenizer, AutoModelForCausalLM    
    from src.PerplexityEvaluator import PerplexityEvaluator
    from src.DetectLM import DetectLM
    from src.PrepareSentenceContext import PrepareSentenceContext
    import pandas as pd

    INPUT_FILE = 'example_text.txt'
    NULL_DATA_FILE = "results/gpt2_no_context_wiki_machine.csv"

    # Reading null data and fit p-value function for every sentence length
    pval_function = get_survival_function(pd.read_csv(NULL_DATA_FILE), G=45)

    # Initialize PerplexityEvaluator with a language model and a tokenizer
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
    print(res)
``

import logging
import spacy
import re
import numpy as np
from src.SentenceParser import SentenceParser

class PrepareSentenceContext(object):
    """
    Parse text and extract length and context information

    This information is needed for evaluating log-perplexity of the text with respect to a language model
    and later on to test the likelihood that the sentence was sampled from the model with the relevant context.
    """

    def __init__(self, sentence_parser='spacy', context_policy=None, context=None):
        if sentence_parser == 'spacy':
            self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer", "ner"])
        if sentence_parser == 'regex':
            logging.warning("Regex-based parser is not good at breaking sentences like 'Dr. Stone', etc.")
            self.nlp = SentenceParser()
            
        self.sentence_parser_name = sentence_parser

        self.context_policy = context_policy
        self.context = context

    def __call__(self, text):
        return self.parse_sentences(text)

    def parse_sentences(self, text):
        pattern_close = r"(.*?)</edit>"
        pattern_open = r"<edit>(.*?)"
        MIN_TOKEN_LEN = 3

        texts = []
        tags = []
        lengths = []
        contexts = []

        def update_sent(sent_text, tag, sent_length):
            texts.append(sent_text)
            tags.append(tag)
            lengths.append(sent_length)
            if self.context is not None:
                context = self.context
            elif self.context_policy is None:
                context = None
            elif self.context_policy == 'previous_sentence' and len(texts) > 0:
                context = texts[-1]
            else:
                context = None
            contexts.append(context)

        curr_tag = None
        parsed = self.nlp(text)
        for s in parsed.sents:
            prev_tag = curr_tag
            matches_close = re.findall(pattern_close, s.text)
            matches_open = re.findall(pattern_open, s.text)
            matches_between = re.findall(r"<edit>(.*?)</edit>", s.text)
            
            logging.debug(f"Current sentence: {s.text}")
            logging.debug(f"Matches open: {matches_open}")
            logging.debug(f"Matches close: {matches_close}")
            logging.debug(f"Matches between: {matches_between}")
            if len(matches_close)>0 and len(matches_open)>0: 
                logging.debug("Found an opening and a closing tag in the same sentence.")
                if prev_tag is None and len(matches_open[0]) >= MIN_TOKEN_LEN:
                    logging.debug("Openning followed by closing with some text in between.")
                    update_sent(matches_open[0], "<edit>", len(s)-2)
                    curr_tag = None
                if prev_tag == "<edit>" and len(matches_close[0]) >= MIN_TOKEN_LEN:
                    logging.warning(f"Wierd case: closing/openning followed by openning in sentence {len(texts)}")
                    update_sent(matches_close[0], prev_tag, len(s)-1)
                    curr_tag = None
                if prev_tag == "</edit>":
                    logging.debug("Closing followed by openning.")
                    curr_tag = "<edit>"
                    if len(matches_between[0]) > MIN_TOKEN_LEN:
                        update_sent(matches_between[0], None, len(s)-2)
            elif len(matches_open) > 0:
                curr_tag = "<edit>"
                assert prev_tag is None, f"Found an opening tag without a closing tag in sentence num. {len(texts)}"
                if len(matches_open[0]) >= MIN_TOKEN_LEN:
                    # text and tag are in the same sentence
                    sent_text = matches_open[0]
                    update_sent(sent_text, curr_tag, len(s)-1)      
            elif len(matches_close) > 0:
                curr_tag = "</edit>"
                assert prev_tag == "<edit>", f"Found a closing tag without an opening tag in sentence num. {len(texts)}"
                if len(matches_close[0]) >= MIN_TOKEN_LEN:
                    # text and tag are in the same sentence
                    update_sent(matches_close[0], prev_tag, len(s)-1)
                curr_tag = None
            else:
                #if len(matches_close)==0 and len(matches_open)==0: 
                # no tag
                update_sent(s.text, curr_tag, len(s))
        return {'text': texts, 'length': lengths, 'context': contexts, 'tag': tags,
                    'number_in_par': np.arange(1,1+len(texts))}

    def REMOVE_parse_sentences(self, text):
        texts = []
        contexts = []
        lengths = []
        tags = []
        num_in_par = []
        previous = None

        text = re.sub("(</?[a-zA-Z0-9 ]+>\.?)\s+", r"\1.\n", text)  # to make sure that tags are in separate sentences
        #text = re.sub("(</[a-zA-Z0-9 ]+>\.?)\s+", r"\n\1.\n", text)  # to make sure that tags are in separate sentences

        parsed = self.nlp(text)

        running_sent_num = 0
        curr_tag = None
        for i, sent in enumerate(parsed.sents):
            # Here we try to track HTML-like tags. There might be
            # some issues because spacy sentence parser has unexpected behavior when it comes to newlines
            all_tags = re.findall(r"(</?[a-zA-Z0-9 ]+>)", str(sent))
            if len(all_tags) > 1:
                    logging.error(f"More than one tag in sentence {i}: {all_tags}")
                    exit(1)
            if len(all_tags) == 1:
                tag = all_tags[0]
                if tag[:2] == '</': # a closing tag
                    if curr_tag is None:
                        logging.warning(f"Closing tag without an opening tag in sentence {i}: {sent}")
                    else:
                        curr_tag = None
                else:
                    if curr_tag is not None:
                        logging.warning(f"Opening tag without a closing tag in sentence {i}: {sent}")
                    else:
                        curr_tag = tag
            else:  # if text is not a tag
                sent_text = str(sent)
                sent_length = len(sent)

                texts.append(sent_text)
                running_sent_num += 1
                num_in_par.append(running_sent_num)
                tags.append(curr_tag)
                lengths.append(sent_length)

                if self.context is not None:
                    context = self.context
                elif self.context_policy is None:
                    context = None
                elif self.context_policy == 'previous_sentence':
                    context = previous
                    previous = sent_text
                else:
                    context = None

                contexts.append(context)
        return {'text': texts, 'length': lengths, 'context': contexts, 'tag': tags,
                'number_in_par': num_in_par}
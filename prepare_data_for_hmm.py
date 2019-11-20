import json
import os
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import pandas as pd
import argparse

from util import files_in_dir
from ArticleDataSample import CommonSectionNames


class DataCreator:
    """
    A class for converting json files of scientific papers into text files containing each sentence in a separate line.
    The json files are assumed to be the output of Allen-AI's science-parse which converts pdf into a json file.
    During conversion, we try to remove some noise (e.g. footnotes), however not all noise is removed.
    For sentence splitting we use NLTK's sent_tokenize which is not perfect as well.
    In addition to creating a paper's text file, we also create files with some information about the paper's
    sections - it is used by HmmArticle class.
    """
    def __init__(self,
                 json_folder,
                 out_folder,
                 glove_fname: str,
                 vocab_size=100000):

        self.vocab = self.get_vocab(glove_fname, vocab_size)

        # workaround for some strings which are not handled correctly by sent_tokenize
        self.substitutions = [
            ("e.g.", "<EG>"),
            ("i.e.", "<IE>"),
            ("et al.", "<ET AL>")
        ]

        self.forbidden_prefixes = ("proceedings", "copyright 201", "correspondence to", 'c©')

        # the footnote-related regular expressions below were defined assuming the papers were converted from pdf to
        # text using science-parse. in this case, there is no space between the footnote number and the following text

        # regular expression for capturing footnote at beginning of sentence: 1 or 2 digits followed by capital letter or "http"
        self.regexp_footnote_at_beginning = re.compile('\d{1,2}([A-Z]|http)')

        # the following 2 regular expressions are for capturing footnote in the middle of a sentence (it may happen,
        # depending on the sentence-splitter's decision)

        # capture 1 or 2 digits followed by a capital letter.
        # the letter should not be K as it is often used after digits to denote Kilo
        # \D*: to capture the latest possible digit followed by a letter
        # then capture "code" or "available".
        # then capture last occurrence of "http"
        self.regexp_footnote_url1 = re.compile('\d{1,2}[A-JL-Z]\D*( code | available ).*(http)(?!.*(http))')

        # 1 or 2 digits and "http" right afterwards - a footnote starting with url
        self.regexp_footnote_url2 = re.compile('\d{1,2}(http)')

        # characters that appear in equations
        self.regexp_equation = re.compile('[\+=≡{}\[\]<>≤≥|∑∇∆‖√∈∠×εσλ∂αβδθµ]')

        # asterisk (or alike), then zero or more characters, then "equal contribution", optional 's' at the end
        self.regexp_equal_contribution = re.compile('(\*|∗|⇤|†).*equal contributions?')

        self.regexp_footnote_misc = re.compile('(\*|∗|⇤|†)(this|work|these|corresponding|the|available|code|http|both)')

        # input & output folders
        self.json_folder = json_folder
        self.out_folder = out_folder
        self.out_text_path = os.path.join(self.out_folder, 'text')
        os.makedirs(self.out_text_path, mode=0o775, exist_ok=False)
        self.out_sections_info_path = os.path.join(self.out_folder, 'sections_info')
        os.makedirs(self.out_sections_info_path, mode=0o775, exist_ok=False)
        self.out_section_per_sent_path = os.path.join(self.out_folder, 'section_per_sent')
        os.makedirs(self.out_section_per_sent_path, mode=0o775, exist_ok=False)

        self.failed_papers = []
        self.footnotes = []

    # extracts the vocabulary from a GloVe embedding file, and remove stop words
    @staticmethod
    def get_vocab(glove_fname, vocab_size):
        print("reading file: {}".format(glove_fname))
        w2vec = pd.read_csv(glove_fname, header=None, sep=' ', quoting=3, encoding="ISO-8859-1")
        print("done")
        vocab = w2vec.ix[:, 0].values
        vocab = set(vocab[:vocab_size])

        stop_words = set(stopwords.words("english"))
        stop_words.update({'min', 'max', 'argmin', 'agrmax', 's.t.', 'w.r.t.'})

        for stop_word in stop_words:
            if stop_word in vocab:
                vocab.remove(stop_word)

        return vocab

    @staticmethod
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def is_footnote_at_beginning(self, sent):
        # match() -> find at the beginning
        return self.regexp_footnote_at_beginning.match(sent) is not None

    def is_equation(self, sent, num_words_thresh):

        # lower() because we use a lowercase vocab
        sent_tokens = word_tokenize(sent.lower())

        num_words = 0
        for token in sent_tokens:
            if len(token) > 1 and token in self.vocab:
                num_words += 1
                if num_words >= num_words_thresh:
                    # if there are enough words, the sentence will not be omitted
                    return False

        # we omit the sentence if it has too few words AND it contains at least one "equation character"
        return self.regexp_equation.search(sent) is not None

    def remove_footnote_with_url_within_sent(self, sent):
        r = self.regexp_footnote_url1.search(sent)

        if r is not None:
            start_i, end_i = r.span()
        else:
            r = self.regexp_footnote_url2.search(sent)
            if r is None:
                # nothing found - return the sentence as is
                return sent
            else:
                start_i = r.span()[0]
                http_str = 'http'
                # find last occurrence of "http" (sometimes there are several)
                last_http_idx = sent.rfind(http_str)
                # last_http_idx is necessarily non-negative as we know 'http' occurs in the string
                end_i = last_http_idx + len(http_str)

        # set end_i to be the index of first space after the url, or end-of-string if space is not found

        # find the last slash (as sometimes a space appears within the url)
        slash_idx = sent[end_i:].rfind('/')
        end_i += slash_idx

        space_idx = sent[end_i:].find(' ')
        if space_idx >= 0:
            end_i += space_idx
        else:
            end_i = len(sent)

        # remove the captured footnote
        new_sent = sent.replace(sent[start_i:end_i], '')

        self.footnotes.append(f"{sent} !@!@ {new_sent}")

        return new_sent

    def has_forbidden_prefix(self, sent):
        return sent.lower().startswith(self.forbidden_prefixes)

    # if regexp matches, the start index is returned. otherwise, -1 is returned
    def search_footnotes(self, sent):
        if self.is_footnote_at_beginning(sent) or self.has_forbidden_prefix(sent):
            return 0

        r = self.regexp_equal_contribution.search(sent.lower())
        if r is not None:
            return r.span()[0]

        r = self.regexp_footnote_misc.search(sent.lower())
        if r is not None:
            return r.span()[0]

        return -1

    def filter_sentences(self, paper_sents, minimal_num_chars=30):
        valid_sents = []

        for sent in paper_sents:
            if self.is_footnote_at_beginning(sent) or self.has_forbidden_prefix(sent):
                self.footnotes.append(sent)
                continue

            sent = self.remove_footnote_with_url_within_sent(sent)

            if len(sent) < minimal_num_chars:
                continue
            if self.is_equation(sent, num_words_thresh=3):
                continue

            valid_sents.append(sent)

        return valid_sents

    # returns True on success, False otherwise
    def prepare_textual_data(self, json_fname):
        with open(json_fname, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        success = self.process_paper(json_data)
        return success

    def process_section_text(self,sec_text):
        if len(sec_text) == 0:
            return sec_text

        # split by newline as it makes it easier to capture some unwanted text
        lines = sec_text.split('\n')
        new_lines = []
        footnote_removed = False

        for cur_line in lines:
            if len(cur_line) == 0:
                continue

            footnote_i = self.search_footnotes(cur_line)
            if footnote_i >= 0:
                footnote_removed = True

                # check if footnote index is positive, i.e. it was found within the line (rather than at the beginning)
                # in this case we don't omit the entire line
                if footnote_i > 0:
                    footnote_text = cur_line[footnote_i:]
                    cur_line = cur_line[:footnote_i]
                    new_lines.append(cur_line)
                else:
                    # entire line is omitted
                    footnote_text = cur_line

                self.footnotes.append(footnote_text)
                continue

            if footnote_removed:
                footnote_removed = False

                if len(new_lines) > 0:
                    # handle the case where the last word of the line is split, but ScienceParse didn't recover it due to
                    # existence of footnotes between the two parts of the split word
                    last_line = new_lines[-1]
                    if last_line[-1] == '-':
                        last_line_tokens = word_tokenize(last_line.lower())
                        cur_line_tokens = word_tokenize(cur_line.lower())
                        # remove the dash from last token of last line, and try to append it to first token of current line
                        candidate_word = last_line_tokens[-1][:-1] + cur_line_tokens[0]
                        if candidate_word in self.vocab:
                            # update the last line of new_lines
                            updated_line = last_line[:-1] + cur_line
                            new_lines[-1] = updated_line
                            continue

            new_lines.append(cur_line)

        # concatenate all lines - the nltk sentence splitter will split this text into sentences
        updated_sec_text = ' '.join(new_lines)
        return updated_sec_text

    # returns True on success, False otherwise
    def process_paper(self, json_data):
        j_title = json_data['title']
        if j_title is None:
            return False

        indices_dict = {CommonSectionNames.INTRO.value: [-1, -1],
                        CommonSectionNames.RELATED.value: [-1, -1],
                        CommonSectionNames.ACK.value: [-1, -1]}  # 'section_name': [sec_start, sec_end]
        # print(j_title)
        sections = json_data['sections']
        section_per_sent = []

        paper_content = []
        end_id = ''
        flag_end = False
        intro_flag_end = False
        intro_passed = False  # did we pass the introduction section yet
        ack_section_found = False
        for sec in sections:
            if not sec['heading']:
                # print('myspecialtoken! ' + sec['text'])
                continue

            sec_text = sec['text']

            # workaround for some strings which are not handled correctly by sent_tokenize
            for subs_tuple in self.substitutions:
                # replace the problematic tokens with temporary substitution
                sec_text = sec_text.replace(subs_tuple[0], subs_tuple[1])

            sec_text = self.process_section_text(sec_text)

            sents = sent_tokenize(sec_text)
            sents = self.filter_sentences(sents)

            section_per_sent += [sec['heading'] + '\n'] * len(sents)

            # parse "related work" section and subsections
            if "Related" in sec['heading']:
                indices_dict[CommonSectionNames.RELATED.value][0] = len(paper_content)
                indices_dict[CommonSectionNames.RELATED.value][1] = len(sents)
                if indices_dict[CommonSectionNames.RELATED.value][1] == 0:  # means that there are sub sections
                    flag_end = True
                    related_sec = sec['heading'].strip().split(" ")
                    end_id = related_sec[0]
                    if end_id.endswith("."):
                        end_id = end_id[:-1]
                    # print("related sub sections: " + end_id)
            if flag_end:
                related_sec = sec['heading'].strip().split(" ")
                id_related = related_sec[0]
                if id_related.endswith("."):
                    id_related = id_related[:-1]
                if self.isfloat(id_related):
                    if float(id_related) < int(end_id) + 1:
                        indices_dict[CommonSectionNames.RELATED.value][1] += len(sents)

            # parse "introduction" section and subsections
            if not intro_passed and ("Introduction" in sec['heading'] or sec['heading'].startswith(("1. ", "1 "))):
                # print(sec['heading'])
                indices_dict[CommonSectionNames.INTRO.value][0] = len(paper_content)
                indices_dict[CommonSectionNames.INTRO.value][1] = len(sents)
                intro_passed = True
                if indices_dict[CommonSectionNames.INTRO.value][1] == 0:  # means that there are sub sections
                    intro_flag_end = True
                    intro_sec = sec['heading'].strip().split(" ")
                    intro_end_id = intro_sec[0]
                    if intro_end_id.endswith("."):
                        intro_end_id = intro_end_id[:-1]
                    # print("related sub sections: " + intro_end_id)
            if intro_flag_end:
                intro_sec = sec['heading'].strip().split(" ")
                id_intro = intro_sec[0]
                if id_intro.endswith("."):
                    id_intro = id_intro[:-1]
                if self.isfloat(id_intro):
                    if float(id_intro) < int(intro_end_id) + 1:
                        indices_dict[CommonSectionNames.INTRO.value][1] += len(sents)

            # parse "acknowledgment" section and subsections
            if "Acknowledgment" in sec['heading'] or "Acknowledgement" in sec['heading']:
                ack_section_found = True
                indices_dict[CommonSectionNames.ACK.value][0] = len(paper_content)
                indices_dict[CommonSectionNames.ACK.value][1] = len(sents)

            for sent_i, sent in enumerate(sents):
                line = sent + '\n'
                paper_content.append(line)

        if len(paper_content) == 0:
            print("something is wrong with paper: {}".format(j_title))
            return False

        if not ack_section_found:
            # in some cases, acknowledgment sentences are not in a dedicated section, but it's easy to capture them
            # we go back few sentences and look for a sentence starting with "Acknowledgment"
            total_num_sents = len(paper_content)
            start_idx = max(total_num_sents - 10, 0)
            for sent_i in range(start_idx, total_num_sents):
                if paper_content[sent_i].startswith("Acknowledgment") or paper_content[sent_i].startswith(
                        "Acknowledgement"):
                    indices_dict[CommonSectionNames.ACK.value][0] = sent_i
                    indices_dict[CommonSectionNames.ACK.value][1] = total_num_sents - sent_i
                    break

        # if there is intersection between Introduction and Related Work, we will not use the Related Work indices
        # (it happens in rare cases where Related Work is a sub-section of Introduction
        intro_last_idx = indices_dict[CommonSectionNames.INTRO.value][0] + indices_dict[CommonSectionNames.INTRO.value][
            1] - 1
        related_start_idx = indices_dict[CommonSectionNames.RELATED.value][0]
        if related_start_idx <= intro_last_idx:
            indices_dict[CommonSectionNames.RELATED.value] = [-1, -1]

        # update section_per_sent according to indices_dict for better section titles for the common sections
        for section_title in indices_dict:
            start_idx = indices_dict[section_title][0]
            num_sents = indices_dict[section_title][1]

            if start_idx >= 0:
                for idx in range(start_idx, start_idx + num_sents):
                    section_per_sent[idx] = section_title + '\n'

        with open(os.path.join(self.out_sections_info_path, j_title + ".txt"),
                  "w", encoding='utf-8') as related_out_file:
            def write_section(sec_name, sec_start, sec_end):
                if sec_start > -1:
                    related_out_file.write("{0}\t{1}\t{2}\n".format(sec_name, sec_start, sec_end))

            for section_name in CommonSectionNames:
                write_section(section_name.value, indices_dict[section_name.value][0],
                              indices_dict[section_name.value][1])

        with open(os.path.join(self.out_text_path, j_title + ".txt"),
                  "w", encoding='utf-8') as out_file:
            out_str = ''.join(paper_content)

            # workaround for some strings which are not handled correctly by sent_tokenize
            for subs_tuple in self.substitutions:
                # replace the temporary substitutions back to original tokens
                out_str = out_str.replace(subs_tuple[1], subs_tuple[0])

            out_file.write(out_str)

        with open(os.path.join(self.out_section_per_sent_path, j_title + ".txt"),
                  "w", encoding='utf-8') as out_file:
            out_str = ''.join(section_per_sent)
            out_file.write(out_str)

        return True

    def save_captured_footnotes(self):
        out_fname = os.path.join(self.out_folder, "footnotes_log.txt")
        with open(out_fname, 'w') as out_file:
            for sent in self.footnotes:
                out_file.write(sent + '\n')

    def run(self):
        json_filenames = files_in_dir(self.json_folder)
        print("number of papers: {}".format(len(json_filenames)))

        for fname_i, fname in enumerate(json_filenames):
            print("--- paper {}: {}".format(fname_i, fname))
            fname = os.path.join(self.json_folder, fname)
            success = self.prepare_textual_data(fname)
            if not success:
                print("FAILED: {}".format(fname))
                self.failed_papers.append(fname)

        if len(self.failed_papers) > 0:
            print("FAILURE with the following papers:")
            for paper_fname in self.failed_papers:
                print(paper_fname)


def main(args):
    data_creator = DataCreator(args.json_folder, args.out_folder, args.glove_path)
    data_creator.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Given json files of scientific papers, this script creates the data which the HMM model expects'
                    'as input'
    )
    parser.add_argument('--json_folder', help='folder of the json files of the papers')
    parser.add_argument('--out_folder', help='output folder')
    parser.add_argument('--glove_path', help='path to GloVe embedding file (GloVe format is assumed')

    args = parser.parse_args()
    main(args)

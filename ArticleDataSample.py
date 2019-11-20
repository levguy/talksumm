import json
import os
from enum import Enum


class CommonSectionNames(Enum):
    """
    Notice - these string values should match the strings in the sections_info files (created by prepare_data_for_hmm)
    """
    INTRO = "Introduction"
    RELATED = "Related Work"
    ACK = "Acknowledgments"


class ArticleDataSample:
    """
    A class for reading the files of a single data sample
    """
    def __init__(self,
                 transcript_fname,
                 paper_text_fname,
                 sections_info_fname,
                 section_per_sent_fname,
                 transcript_json_fname=None,
                 paper_json_fname=None,
                 alignment_json_fname=None
                 ):
        """
        The last 3 arguments are obsolete (they were used to represent a labeled data sample, as we had
        experimented with few manually-labeled alignments between paper sentences and speech transcript).
        """

        self.transcript_jsn = None
        self.paper_jsn = None
        self.alignment_jsn = None
        self.transcript_sents = None
        self.paper_sents = None
        self.sections_sent_indices = {}
        self.section_per_sent = []

        if transcript_json_fname:
            with open(transcript_json_fname, encoding='utf-8') as in_file:
                self.transcript_jsn = json.load(in_file)

        if paper_json_fname:
            with open(paper_json_fname, encoding='utf-8') as in_file:
                self.paper_jsn = json.load(in_file)

        if alignment_json_fname:
            with open(alignment_json_fname, encoding='utf-8') as in_file:
                self.alignment_jsn = json.load(in_file)

        if transcript_fname:
            with open(transcript_fname) as in_file:
                self.transcript_sents = [sent.rstrip('\n') for sent in in_file]

            transcript_word_num = 0
            for sent in self.transcript_sents:
                # the sentences are not tokenized, but by counting spaces we can get approximately the number of
                # words (we don't need exact number here)
                transcript_word_num += sent.count(' ') + 1
            self.transcript_word_num = transcript_word_num

        if paper_text_fname:
            with open(paper_text_fname) as in_file:
                self.paper_sents = [sent.rstrip('\n') for sent in in_file]

        # this function handles the case that sections_info_fname is None
        self.__read_sections_info_file(sections_info_fname)

        # this function handles the case that section_per_sent_fname is None
        self.__read_section_per_sent_file(section_per_sent_fname)

    @staticmethod
    def __jsn_to_single_list(jsn):
        """
        method for reading transcript.json and alignment.json which share the same structure
        """
        keys = jsn.keys()
        # convert the keys from string to int (these keys are slide indices)
        # notice that the slide indices do not necessarily start from 0
        slide_indices_sorted = [int(key) for key in keys]
        # sort
        slide_indices_sorted.sort()

        # this list will hold the output list
        out_list = []

        for slide_i in slide_indices_sorted:
            slide_i_str = str(slide_i)
            # per slide, there is a list
            cur_list = jsn[slide_i_str]
            out_list.extend(cur_list)

        return out_list

    def get_transcript_sentences(self, punctuated: bool):
        if punctuated:
            if self.transcript_sents:
                transcript_sents = self.transcript_sents.copy()
            else:
                raise Exception("transcript_sents was not initialized")
        else:
            if self.transcript_jsn:
                transcript_sents = self.__jsn_to_single_list(self.transcript_jsn)
            else:
                raise Exception("transcript_jsn was not initialized")

        num_sents = len(transcript_sents)
        print("total number of sentences in the transcript: {}".format(num_sents))
        return transcript_sents

    def get_ground_truth_sent_ids(self):
        if not self.alignment_jsn:
            raise Exception("alignment_jsn was not initialized")

        gt_sent_ids = self.__jsn_to_single_list(self.alignment_jsn)
        num_sents = len(gt_sent_ids)
        print("total number of ground-truth sentences: {}".format(num_sents))
        return gt_sent_ids

    def __subsection_index_to_tuple(self, str_index):
        """
        converts subsection index (a string) to tuple of ints
        """
        split = str_index.split('.')
        # omit the last element as it is empty (since the string ends with '.')
        split = split[:-1]
        # string -> int
        split = [int(num) for num in split]
        tup = tuple(split)
        return tup

    def __tuple_to_subsection_index(self, tup):
        """
        converts a tuple of ints to subsection index (a string)
        """
        subsection_index = '.'.join('{}'.format(k) for k in tup)
        subsection_index += '.'
        return subsection_index

    def get_article_sentences_labeled(self, lower_case):
        """
        returns 2 lists:
          1. list of the article's sentences (lowercased in case lower_case is True)
          2. list of the full indices of the sentences (section index, subsection index, and so on up to
             the sentence index). these full indices are strings (numbers are separated by period).
        the order of sentences is similar to the order in the article text.
        """
        if not self.paper_jsn:
            raise Exception("paper_jsn was not initialized")

        jsn = self.paper_jsn
        keys = jsn.keys()
        # convert the keys from string to tuple of ints
        subsections_indices = [self.__subsection_index_to_tuple(key) for key in keys]
        # sort
        subsections_indices.sort()

        article_sentences = []
        sentences_full_indices = []

        for subsection_idx in subsections_indices:
            key = self.__tuple_to_subsection_index(subsection_idx)
            cur_sent_list = jsn[key]

            for sent_i, sent in enumerate(cur_sent_list):
                sent_full_idx = key + str(sent_i)
                sentences_full_indices.append(sent_full_idx)

                if lower_case:
                    sent = sent.lower()

                article_sentences.append(sent)

        return article_sentences, sentences_full_indices

    def get_article_sentences_unlabeled(self, lower_case):
        if self.paper_sents:
            paper_sents = self.paper_sents.copy()
            if lower_case:
                paper_sents = [sent.lower() for sent in paper_sents]

            return paper_sents
        else:
            raise Exception("paper_sents was not initialized")

    def __read_sections_info_file(self, sections_info_fname):
        if sections_info_fname and os.path.isfile(sections_info_fname):
            with open(sections_info_fname) as in_file:
                lines = [line.rstrip('\n') for line in in_file]

            # parse each line, relying on the format used by write_section in prepare_data_for_hmm.py
            for line in lines:
                splt = line.split('\t')
                section_name = splt[0]
                start_i = int(splt[1])
                num_sents = int(splt[2])

                sent_indices = list(range(start_i, start_i + num_sents))
                self.sections_sent_indices[section_name] = sent_indices

        # one ore more section names might be missing from the dictionary
        # due to missing file or lack of corresponding line in the file
        for section_name in CommonSectionNames:
            if section_name.value not in self.sections_sent_indices:
                self.sections_sent_indices[section_name.value] = []

    def __read_section_per_sent_file(self, section_per_sent_fname):
        if section_per_sent_fname and os.path.isfile(section_per_sent_fname):
            with open(section_per_sent_fname) as in_file:
                self.section_per_sent = [line.rstrip('\n') for line in in_file]

        # sanity check - verify that section_per_sent and sections_sent_indices agree on the sentences indices
        # of the 3 common sections
        for section_name in CommonSectionNames:
            sent_indices = [sent_i for sent_i, section_title in enumerate(self.section_per_sent) if
                             section_name.value == section_title]
            if sent_indices != self.sections_sent_indices[section_name.value]:
                print("--- mismatch: {}".format(section_name.value))
                print("paper: {}".format(section_per_sent_fname))
                print("sent_indices:")
                print(sent_indices)
                print("self.sections_sent_indices[{}]:".format(section_name.value))
                print(self.sections_sent_indices[section_name.value])

    def get_section_sent_indices(self, section_name: CommonSectionNames):
        if section_name.value in self.sections_sent_indices:
            return self.sections_sent_indices[section_name.value]
        else:
            raise Exception("unexpected section name: {}".format(section_name.value))

    def get_paper_sent_num(self):
        return len(self.paper_sents)

    def get_paper_sents(self):
        return self.paper_sents.copy()

    def get_transcript_word_num(self):
        return self.transcript_word_num

    def get_section_per_sent(self):
        return self.section_per_sent.copy()

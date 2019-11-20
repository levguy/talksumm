import numpy as np
from nltk.tokenize import word_tokenize

from ArticleDataSample import ArticleDataSample, CommonSectionNames


class SummaryCreator:
    """
    Given a data sample of a paper, and the durations (scores) of the paper's sentences, obtained using HmmArticle,
    this class enables you to generate summaries, according to different objectives:
    - Given a number k, the summary will include to top-k sentences, w.r.t their durations (scores). Relevant
      function: create_top_scored_sents_file
    - Given desired number of words, the summary will include roughly this number of words (roughly - because
      the summary will include whole-sentences only). Relevant function: create_summary_file_by_target
    - Given a desired compression ratio, the summary length will be determined accordingly. Here, length means number
      of sentences. Relevant function: create_summary_file_by_target
    A summary file created by create_top_scored_sents_file contains only the sentences chosen for summary.
    A summary file created by create_summary_file_by_target contains all the paper's sentences, where the ones
    chosen for summary are marked by the string "@highlight". This is the format of the "CNN / Daily Mail"
    summarization benchmark.
    """
    def __init__(self, article_data_sample: ArticleDataSample, durations=None, durations_fname=None):
        self.article_data_sample = article_data_sample
        # get the original article sentences (including related work, with original case)
        self.orig_sentences = self.article_data_sample.get_article_sentences_unlabeled(lower_case=False)

        if durations is not None:
            self.durations = np.copy(durations)
        elif durations_fname:
            self.load_durations_file(durations_fname)
        else:
            self.durations = None

        self.sent_len = self.calc_sent_len()

    def calc_sent_len(self):
        sent_len = np.zeros(len(self.orig_sentences), dtype=np.int)
        for sent_i, sent in enumerate(self.orig_sentences):
            sent_len[sent_i] = len(word_tokenize(sent))
        return sent_len

    def load_durations_file(self, durations_fname):
        with open(durations_fname) as in_file:
            self.durations = [int(line) for line in in_file]

    def get_summary_sent_indices(self, duration_thresh=1):
        if self.durations is None:
            raise Exception("self.durations not initialized")

        summary_sent_indices = []
        for sent_i, duration in enumerate(self.durations):
            if duration >= duration_thresh:
                summary_sent_indices.append(sent_i)

        return summary_sent_indices

    def create_summary_file_by_duration(self, out_fname, duration_thresh=1, exclude_related_work=False):
        """
        creates a file in the "CNN / Daily Mail" format: the files contain all paper sentences, where
        the summary sentences are highlighted.
        exclude_related_work: if True, sentences from Related Work section will not be written to the output file.
        """
        summary_sent_indices = self.get_summary_sent_indices(duration_thresh)
        if exclude_related_work:
            related_work_sent_indices = self.article_data_sample.get_section_sent_indices(CommonSectionNames.RELATED)
            ack_sent_indices = self.article_data_sample.get_section_sent_indices(CommonSectionNames.ACK)
            excluded_sent_indices = set(related_work_sent_indices + ack_sent_indices)

        else:
            excluded_sent_indices = {}

        highlight_str = "@highlight\n"

        with open(out_fname, 'w') as out_file:
            for sent_i, sent_str in enumerate(self.orig_sentences):

                if exclude_related_work and sent_i in excluded_sent_indices:
                    continue

                sent_str += '\n'
                out_file.write(sent_str)

                if sent_i in summary_sent_indices:
                    out_file.write(highlight_str)
                    out_file.write(sent_str)

    def create_summary_file_by_target(self, out_fname, target_name, target_value, exclude_related_work=False):
        """
        target_name: 'compress_ratio' or 'num_words'
        """
        n_thresholds = max(self.durations) + 1

        sent_mask_per_thresh = []
        for duration_thresh in range(n_thresholds):
            mask = np.asarray([duration >= duration_thresh for duration in self.durations], dtype=np.int)
            sent_mask_per_thresh.append(mask)

        if target_name == 'compress_ratio':
            num_summary_sents = np.zeros(n_thresholds, dtype=np.int)
            for duration_thresh in range(n_thresholds):
                num_summary_sents[duration_thresh] = sum(sent_mask_per_thresh[duration_thresh])
            orig_num_of_article_sentences = len(self.orig_sentences)
            compress_ratios = num_summary_sents / orig_num_of_article_sentences
            result_per_thresh = compress_ratios

        elif target_name == 'num_words':
            num_summary_words = np.zeros(n_thresholds, dtype=np.int)
            for duration_thresh in range(n_thresholds):
                num_summary_words[duration_thresh] = sum(sent_mask_per_thresh[duration_thresh] * self.sent_len)
            result_per_thresh = num_summary_words

        else:
            raise Exception("unexpected target_name")

        deltas = np.abs(result_per_thresh - target_value)

        chosen_duration_thresh = np.argmin(deltas)
        obtained_value = result_per_thresh[chosen_duration_thresh]

        self.create_summary_file_by_duration(out_fname, chosen_duration_thresh, exclude_related_work)

        return chosen_duration_thresh, obtained_value

    def create_scored_sents_in_sections_file(self, out_fname):
        section_per_sent = self.article_data_sample.get_section_per_sent()
        assert len(section_per_sent) == len(self.orig_sentences)

        sum_durations = sum(self.durations)

        cur_section = None

        with open(out_fname, 'w') as out_file:
            for sent_i, section_name in enumerate(section_per_sent):
                if section_name != cur_section:
                    cur_section = section_name
                    out_file.write("--- {}\n".format(cur_section))
                out_file.write("{}\t{}\t{:.2f}\t{}\n".format(sent_i,
                                                             self.durations[sent_i],
                                                             self.durations[sent_i] / sum_durations,
                                                             self.orig_sentences[sent_i]))

    def create_top_scored_sents_file(self, desired_num_sents, duration_thresh, out_fname):
        """
        sentences will be retrieved only if their duration is at least duration_thresh, which means that the number
        of retrieved sentences might be smaller than desired_num_sents
        """
        # scores = np.array(self.durations) / sum(self.durations)
        scores = np.array(self.durations)
        num_eligible_sents = np.sum(scores >= duration_thresh)
        num_retrieved_sents = min(desired_num_sents, num_eligible_sents)
        sorted_indices = np.flip(np.argsort(scores))
        top_score_indices = sorted_indices[:num_retrieved_sents]
        top_score_indices_orig_order = np.sort(top_score_indices)

        with open(out_fname, 'w') as out_file:
            for sent_i in top_score_indices_orig_order:
                out_file.write("{}\t{}\t{}\n".format(sent_i,
                                                     scores[sent_i],
                                                     self.orig_sentences[sent_i]))

        return num_retrieved_sents

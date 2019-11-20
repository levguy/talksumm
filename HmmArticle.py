import string
import copy
import itertools
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from enum import Enum
from itertools import compress
from tqdm import tqdm

from ArticleDataSample import ArticleDataSample, CommonSectionNames
from util import tprint, cosine_similarity
from w2v_utils import read_pretrained_w2v
from viterbi import viterbi


class HmmAlgo(Enum):
    VITERBI_0 = 0
    DUMMY = 1  # for debugging


class PredictedSeqInfoKey(Enum):
    """
    keys of dictionary containing information of the HMM's predicted sequence of sentences
    """
    SENT_I = "Sent i"
    BACKGROUND = "Backg"
    SENT_FULL_ID = "Sent ID"
    SENT_TEXT = "Sent Text"
    DURATION = "Duration"
    SPOKEN_WORDS = "Spoken words"
    IS_GROUND_TRUTH = "GT"

    @staticmethod
    def get_columns_order(labeled_data=False, include_background=False):
        """
        returns a list of keys describing the values to be printed, and their order
        """
        key_list = []
        key_list.append(PredictedSeqInfoKey.SENT_I.value)
        if include_background:
            key_list.append(PredictedSeqInfoKey.BACKGROUND.value)
        if labeled_data:
            key_list.append(PredictedSeqInfoKey.SENT_FULL_ID.value)
        key_list.append(PredictedSeqInfoKey.SENT_TEXT.value)
        key_list.append(PredictedSeqInfoKey.DURATION.value)
        key_list.append(PredictedSeqInfoKey.SPOKEN_WORDS.value)
        if labeled_data:
            key_list.append(PredictedSeqInfoKey.IS_GROUND_TRUTH.value)

        return key_list


class HmmArticleConfig:
    """
    configuration of HmmArticle
    """

    def __init__(self,
                 word_embed_fname: str,
                 labeled_data_mode: bool):
        """
        some configuration parameters have no default value - they are must be passed in the constructor
        the other configuration parameters may be set after instantiation.
        """

        # parameters with no default values
        self.word_embed_fname = word_embed_fname
        self.labeled_data_mode = labeled_data_mode

        # parameters with default values
        self.section_id_intro = 0
        self.section_id_related_work = None
        self.similarity_fname = None
        self.stay_prob = None  # if None, stay_prob will be determined heuristically as a function of paper & transcript lengths
        self.auto_stay_prob_first_approach = True  # selection between 2 approaches of auto-defining stay_prob
        self.trans_prob_decay = 0.75
        self.emis_prob_subtruct_min_factor = 0.8
        self.allow_backward_steps = True  # allow transitioning from one sentence to an earlier one
        # factor for making the backward steps less probable than forward steps.
        # this parameter is relevant only if allow_backward_steps is True
        self.backward_prob_factor = 2
        # we tried to use "background" to model those parts in the talk where the speaker utters words that are
        # unrelated to any sentence in the paper (like in Malmaud et al., www.cs.ubc.ca/~murphyk/Papers/naacl15.pdf).
        # however, better results were obtained without using background
        self.backg_stay_prob = None  # None means disable background
        self.backg_word_count_fname = None
        self.lower_case = True
        self.remove_stop_words = True
        self.hmm_algo = HmmAlgo.VITERBI_0
        # transcript_word_level_mode=True means that each time-step corresponds to a single spoken word of the
        # transcript - this is the mode which we describe in our paper. we have also tried "sentence-level" mode, in
        # which each time-step corresponds to a sentence in the transcript. for this, we used a pre-trained
        # punctuation-restoration model in order to split the transcript into sentences. however, better results were
        # obtained with the word-level mode.
        self.transcript_word_level_mode = True
        self.sent_sent_similarity_wordwise = True  # releavnt only when transcript_word_level_mode is False
        self.debug_mode = False
        self.wmd = False  # Word Mover's Distance

    def print_configuration(self):
        print("HmmArticleConfig:")
        for item in vars(self).items():
            print("%s: %s" % item)


class HmmArticle:
    """
    Given a data sample (article & transcript), this class prepares the HMM's probabilities and
    runs the Viterbi algorithm to obtain a predicted sequence of hidden states, i.e. paper sentences
    """

    def __init__(self, article_data_sample: ArticleDataSample, cfg: HmmArticleConfig):

        self.article_data_sample = article_data_sample
        self.cfg = copy.deepcopy(cfg)

        self.using_background = (self.cfg.backg_stay_prob is not None)
        self.w2v = {}
        self.w2v_mean = None
        self.w2v_dim = 0
        self.transcript_tokens = None
        self.transcript_ids = []
        self.transcript_sents = []
        self.id2word = []
        self.word2id = {}
        self.article_sentences = []
        self.sentences_full_indices = []
        self.section_idx_per_sentence = []
        # this dict will include also keys of Related Work section (which we omit), since the
        # reference summary might include sentences from this section
        self.full_index_to_sentence = {}
        self.intro_sent_indices = []
        # we exclude the sentences of Related Work and Acknowledgments sections
        self.excluded_sent_indices = {}
        self.article_all_sent_vecs = []
        self.transcript_all_sent_vecs = []
        self.start_prob = None
        self.transition_prob = None
        self.emission_prob = None
        self.model = None
        self.observed_seq = None
        self.predicted_seq_info = []
        # Here we will store the duration of each sentence, i.e. number of time-steps in which the sentence was
        # chosen by the Viterbi algorithm. This models the number of words uttered by the speaker to describe the
        # sentence, and can be used as importance score.
        self.durations = None
        self.print_predicted_sentences = False
        # word count from external corpus - for background word distribution
        self.backg_word_count = {}
        self.warnings = []

        if self.cfg.remove_stop_words:
            self.stop_words = self.get_stop_words()
            print('the following stop words and punctuations will be removed from article text and transcript:')
            print(self.stop_words)

        self.parse_transcript()

        self.process_article_sentences()

        self.n_article_sentences = len(self.article_sentences)
        self.n_states = 2 * self.n_article_sentences if self.using_background else self.n_article_sentences

        if self.cfg.transcript_word_level_mode:
            self.n_observations = len(self.id2word)
        else:
            self.n_observations = len(self.transcript_sents)

        print("n_observations: {}".format(self.n_observations))
        print("n_article_sentences: {}".format(self.n_article_sentences))
        print("n_states: {}".format(self.n_states))

        if self.cfg.allow_backward_steps:
            # setting it to n_article_sentences means that the probability will be distributed all the way backward
            # up to the first sentence.
            # we have also tried smaller values, i.e. limiting how far the backward-transition can be
            self.max_backward_steps = self.n_article_sentences
        else:
            self.max_backward_steps = 0

        if self.using_background:
            self.read_backg_word_count_file(self.cfg.backg_word_count_fname)

        self.hmm_probabilities_init()

        if self.cfg.labeled_data_mode:
            self.gt_unique_sent_ids = set(self.article_data_sample.get_ground_truth_sent_ids())

            # check if there is a ground-truth sentence from the Related Work section which was omitted
            if self.cfg.section_id_related_work is not None:
                for gt_sent_id in self.gt_unique_sent_ids:
                    if self.get_section_idx(gt_sent_id) == self.cfg.section_id_related_work:
                        warning = "WARNING: Related Work section ({}) was omitted but there is a ground-truth sentence ({}) from this section".format(
                            self.cfg.section_id_related_work, gt_sent_id)
                        self.warnings.append(warning)
                        print(warning)
        else:
            self.gt_unique_sent_ids = {}

    @staticmethod
    def get_stop_words():
        stop_words = set(nltk.corpus.stopwords.words("english"))
        punct = set(string.punctuation)
        stop_words.update(punct)
        return stop_words

    @staticmethod
    def get_section_idx(full_index):
        """
        extracts the section index out of full index (e.g.: 3.0.1 --> 3)
        """
        split = full_index.split('.', maxsplit=1)
        section_idx = int(split[0])
        return section_idx

    def process_article_sentences(self):
        if self.cfg.labeled_data_mode:
            self.article_sentences, self.sentences_full_indices = self.article_data_sample.get_article_sentences_labeled(
                self.cfg.lower_case)

            orig_num_of_sents = len(self.article_sentences)

            section_idx_per_sentence = []

            for sent_i, full_index in enumerate(self.sentences_full_indices):
                section_idx = self.get_section_idx(full_index)
                section_idx_per_sentence.append(section_idx)

                if section_idx == self.cfg.section_id_intro:
                    self.intro_sent_indices.append(sent_i)

                self.full_index_to_sentence[full_index] = self.article_sentences[sent_i]

            bool_filter = [section_idx != self.cfg.section_id_related_work for section_idx in section_idx_per_sentence]

        # unlabeled data
        else:
            self.article_sentences = self.article_data_sample.get_article_sentences_unlabeled(self.cfg.lower_case)

            self.intro_sent_indices = self.article_data_sample.get_section_sent_indices(CommonSectionNames.INTRO)

            related_work_sent_indices = self.article_data_sample.get_section_sent_indices(CommonSectionNames.RELATED)
            ack_sent_indices = self.article_data_sample.get_section_sent_indices(CommonSectionNames.ACK)
            self.excluded_sent_indices = set(related_work_sent_indices + ack_sent_indices)

            orig_num_of_sents = len(self.article_sentences)

            bool_filter = [sent_i not in self.excluded_sent_indices for sent_i in range(orig_num_of_sents)]

        print("original number of article sentences: {}".format(orig_num_of_sents))

        if self.cfg.debug_mode:
            desired_num_sentences = 5
            bool_filter = [False] * len(bool_filter)
            bool_filter[:desired_num_sentences] = [True] * desired_num_sentences

            self.intro_sent_indices = [0, 1]

            print("DEBUG mode: we take only the first {} sentences".format(desired_num_sentences))

        self.article_sentences = list(compress(self.article_sentences, bool_filter))
        if self.cfg.labeled_data_mode:
            self.sentences_full_indices = list(compress(self.sentences_full_indices, bool_filter))
            self.section_idx_per_sentence = list(compress(section_idx_per_sentence, bool_filter))

        num_of_sents = len(self.article_sentences)

        print("after removing sentences of Related Work section, number of article sentences is now: {}".format(
            num_of_sents))

        # avoid empty intro_sent_indices
        if len(self.intro_sent_indices) == 0:
            dummy_num_intro_sents = min(20, num_of_sents)
            self.intro_sent_indices = list(range(dummy_num_intro_sents))
            print("intro_sent_indices was empty. it was set to the first {} sentences".format(dummy_num_intro_sents))

    def parse_transcript(self):
        # we use punctuated=True also in transcript_word_level_mode (transcript.json is actually obsolete)
        transcript_sents = self.article_data_sample.get_transcript_sentences(punctuated=True)

        num_sents = len(transcript_sents)

        for sent_i, sent in enumerate(transcript_sents):
            sent = sent.replace("%HESITATION", "")

            if self.cfg.lower_case:
                sent = sent.lower()

            # replace the sentence string with a list of its tokens
            word_list = word_tokenize(sent)
            if self.cfg.remove_stop_words:
                word_list = [word for word in word_list if word not in self.stop_words]
            transcript_sents[sent_i] = word_list

        # list of lists -> one list of all tokens
        self.transcript_tokens = list(itertools.chain.from_iterable(transcript_sents))

        num_tokens = len(self.transcript_tokens)
        print("total number of tokens in the whole transcript: {}".format(num_tokens))

        # the unique tokens are the vocabulary of the transcript
        self.id2word = list(set(self.transcript_tokens))
        self.id2word.sort()
        print("vocabulary size: {}".format(len(self.id2word)))
        # print(self.id2word)

        # initialize word->id dictionary
        for word_i, word in enumerate(self.id2word):
            self.word2id[word] = word_i

        transcript_ids_per_sent = []

        for sent in transcript_sents:
            word_ids = []
            for word in sent:
                word_ids.append(self.word2id[word])

            transcript_ids_per_sent.append(word_ids)

        # list of lists -> one list of all token ids
        self.transcript_ids = list(itertools.chain.from_iterable(transcript_ids_per_sent))

        self.transcript_sents = transcript_sents

        if self.cfg.transcript_word_level_mode:
            if self.cfg.debug_mode:
                self.observed_seq = np.array([0, 2, 1, 1, 2, 0])
            else:
                self.observed_seq = np.asarray(self.transcript_ids)
        else:
            self.observed_seq = np.arange(num_sents)

    def read_backg_word_count_file(self, backg_word_count_fname):
        tprint("reading file: {}".format(backg_word_count_fname))

        with open(backg_word_count_fname) as file:
            for line in file:
                word, count = line.split()
                self.backg_word_count[word] = int(count)

        tprint("done")

    def prepare_sent_vecs(self, sent_list):
        """
        sent_list can be either a list of strings or a list of lists of tokens
        """
        # tokenize if needed
        if type(sent_list[0]) == str:
            sent_list_tokens = []
            for sent_i, sent in enumerate(sent_list):
                sent_list_tokens.append(word_tokenize(sent))

            sent_list = sent_list_tokens

        # now sent_list is necessarily a list of lists of tokens

        all_sent_vecs = []
        total_not_found = 0

        for sent_i, sent_tokens in enumerate(sent_list):
            sent_vecs = []

            for word in sent_tokens:
                if self.cfg.remove_stop_words and word in self.stop_words:
                    continue

                if word in self.w2v:
                    sent_vecs.append(self.w2v[word])
                else:
                    print("word not found: {}".format(word))
                    total_not_found += 1

            if not sent_vecs:
                sent_str = ' '.join(sent_tokens)
                warning = "WARNING: all words not found for sentence: {}".format(sent_str)
                # raise Exception(warning)
                self.warnings.append(warning)
                print(warning)

                sent_vecs.append(self.w2v_mean)

            all_sent_vecs.append(sent_vecs)

        print("total number of times word not found: {}".format(total_not_found))

        return all_sent_vecs

    @staticmethod
    def word_sent_similarity(word_vec, sent_vecs):
        """
        sent_vecs: a list of the vectors of the sentence's words
        """
        sent_len = len(sent_vecs)
        similarities = np.zeros(sent_len)

        for vec_i, vec in enumerate(sent_vecs):
            cosine_sim = cosine_similarity(vec, word_vec)

            # obtain positive similarity
            sim = np.exp(cosine_sim)

            similarities[vec_i] = sim

        max_sim = np.max(similarities)
        return max_sim

    def sent_sent_similarity(self, sent1_vecs, sent2_vecs):
        """
        sent1_vecs: a list of the word vectors of the 1st sentence
        sent2_vecs: same, for the 2nd sentence
        """
        if self.cfg.sent_sent_similarity_wordwise:
            similarities = []
            for word_vec in sent1_vecs:
                similarities.append(self.word_sent_similarity(word_vec, sent2_vecs))

            max_sim = max(similarities)
            return max_sim

        # cosine similarity between the mean vectors
        else:
            sent1_mean_w2v = np.mean(sent1_vecs, 0)
            sent2_mean_w2v = np.mean(sent2_vecs, 0)

            cosine_sim = cosine_similarity(sent1_mean_w2v, sent2_mean_w2v)

            sim = np.exp(cosine_sim)
        return sim

    def prepare_start_prob(self):
        """
        prepares the start probabilities
        """

        # we set start probability as uniform over the sentences in the Introduction section

        start_prob = np.zeros(self.n_article_sentences)

        num_sents_in_intro = len(self.intro_sent_indices)
        prob = 1 / num_sents_in_intro

        for sent_i in self.intro_sent_indices:
            start_prob[sent_i] = prob

        if not self.using_background:
            self.start_prob = start_prob
        else:
            self.start_prob = np.zeros(self.n_states)
            # we set probability of 1 to start with background==1
            self.start_prob[self.n_article_sentences:] = start_prob

    def prepare_transition_prob(self):
        """
        prepares the transition probabilities matrix
        """
        stay_prob = self.cfg.stay_prob
        if stay_prob is None:
            if not self.cfg.transcript_word_level_mode:
                raise Exception("None value for stay_prob is supported in transcript_word_level_mode only")
            # notice that in some very few cases, this ratio is larger than 1, we will handle this
            paper_trans_len_ratio = self.n_article_sentences / len(self.observed_seq)

            if self.cfg.auto_stay_prob_first_approach:
                # with this definition, the resulting stay_prob is around 0.3 in average
                factor = 3
                stay_prob = (1 - paper_trans_len_ratio) / factor
            else:
                # another approach which we tried, it achieved good results as well
                factor = 7
                stay_prob = 1 - (factor * paper_trans_len_ratio)

            stay_prob = max(stay_prob, 0.1)
            stay_prob = round(stay_prob, 2)

        transition_prob = np.zeros((self.n_article_sentences, self.n_article_sentences))

        leave_prob = 1 - stay_prob

        print("stay_prob: {:.3}".format(stay_prob))

        # helper vector for probability decay
        helper_vec = np.ones(self.n_article_sentences, dtype=np.float)
        for i in range(1, self.n_article_sentences):
            helper_vec[i] = self.cfg.trans_prob_decay * helper_vec[i - 1]

        for state_i in range(self.n_article_sentences):
            # notice that when state_i == self.n_article_sentences - 1, and if backward steps are not allowed,
            # then transition_prob[self.n_article_sentences - 1, :] will not sum up to 1.
            # even though there is nowhere to go on from the last state, we don't set the stay probability
            # to 1 here, as the viterbi algorithm exploits it and pushes to reach the last state ASAP.

            transition_prob[state_i, state_i] = stay_prob

            n_following_states = self.n_article_sentences - state_i - 1
            n_previous_states = min(state_i, self.max_backward_steps)

            right_vec = np.copy(helper_vec[: n_following_states])
            left_vec = np.flip(np.copy(helper_vec[: n_previous_states])) / self.cfg.backward_prob_factor

            # normalization factor such that sum(right_vec) + sum(left_vec) will sum up to leave_prob
            normalization_factor = (sum(right_vec) + sum(left_vec)) / leave_prob

            right_vec /= normalization_factor
            left_vec /= normalization_factor

            transition_prob[state_i, (state_i + 1):] = right_vec
            transition_prob[state_i, (state_i - n_previous_states): state_i] = left_vec

        if not self.using_background:
            self.transition_prob = transition_prob
        else:
            self.transition_prob = np.zeros((self.n_states, self.n_states))

            # the part of the matrix in which the background stays the same
            # in this case we multiply the sentence-transition probabilities by backg_stay_prob
            backg_stays_block = self.cfg.backg_stay_prob * transition_prob

            # the part of the matrix in which the background changes
            # in this case we multiply the sentence-transition probabilities by (1 - self.cfg.backg_stay_prob)
            backg_changes_block = (1 - self.cfg.backg_stay_prob) * transition_prob

            # top-left block: background stays at 0
            self.transition_prob[:self.n_article_sentences, :self.n_article_sentences] = backg_stays_block
            # bottom-right block: background stays at 1
            self.transition_prob[self.n_article_sentences:, self.n_article_sentences:] = backg_stays_block
            # bottom-left block: background changes from 1 to 0
            self.transition_prob[self.n_article_sentences:, :self.n_article_sentences] = backg_changes_block
            # top-right block: background changes from 0 to 1
            self.transition_prob[:self.n_article_sentences:, self.n_article_sentences:] = backg_changes_block

    def get_backg_distribution(self):
        if self.cfg.transcript_word_level_mode:
            # if a word didn't appear in the external text, we set it's count to 1
            word_dist = np.ones(self.n_observations)

            for word_i in range(self.n_observations):
                word = self.id2word[word_i]

                if word in self.backg_word_count:
                    word_dist[word_i] = self.backg_word_count[word]

            word_dist /= np.linalg.norm(word_dist)

            return word_dist

        else:
            raise Exception("currently background is only supported in transcript_word_level_mode")

    def prepare_emission_prob(self):
        """
        prepares the emission probabilities matrix
        """
        if self.cfg.similarity_fname and os.path.isfile(self.cfg.similarity_fname):
            tprint("loading similarity file: {}".format(self.cfg.similarity_fname))
            emission_prob = np.load(self.cfg.similarity_fname)
            tprint("done")

        else:
            is_glove = not self.cfg.word_embed_fname[-3:] == 'bin'

            self.w2v, self.w2v_mean = read_pretrained_w2v(self.cfg.word_embed_fname, is_glove=is_glove)
            self.w2v_dim = self.w2v_mean.shape[0]
            tprint("w2v dimension: {}".format(self.w2v_dim))

            self.article_all_sent_vecs = self.prepare_sent_vecs(self.article_sentences)

            if not self.cfg.transcript_word_level_mode:
                self.transcript_all_sent_vecs = self.prepare_sent_vecs(self.transcript_sents)

            emission_prob = np.zeros((self.n_article_sentences, self.n_observations))

            tprint("preparing similarities for emission probabilities...")
            # prepare word vectors in case of word level mode
            if self.cfg.transcript_word_level_mode:
                word_vecs = []

                for observation_i in range(self.n_observations):
                    word = self.id2word[observation_i]
                    if word in self.w2v:
                        word_vec = self.w2v[word]
                    else:
                        # word_vec = self.w2v["<unk>"]
                        word_vec = self.w2v_mean

                    word_vecs.append(word_vec)

            for state_i in tqdm(range(self.n_article_sentences)):
                for observation_i in range(self.n_observations):
                    if self.cfg.transcript_word_level_mode:
                        emission_prob[state_i, observation_i] = self.word_sent_similarity(
                            word_vecs[observation_i],
                            self.article_all_sent_vecs[state_i])
                    else:
                        if not self.cfg.wmd:
                            emission_prob[state_i, observation_i] = self.sent_sent_similarity(
                                self.transcript_all_sent_vecs[observation_i],
                                self.article_all_sent_vecs[state_i])
                        else:
                            emission_prob[state_i, observation_i] = -self.w2v.wmdistance(
                                self.transcript_sents[observation_i],
                                self.article_sentences[state_i])

            if self.cfg.similarity_fname:
                # save to file
                np.save(self.cfg.similarity_fname, emission_prob)
                tprint("created file: {}".format(self.cfg.similarity_fname))

        # manipulate the similarities and normalize
        for state_i in range(self.n_article_sentences):

            if self.cfg.wmd:
                emission_prob[state_i, :] -= np.max(emission_prob[state_i, :])

            # this works better than applying a second softmax
            if self.cfg.emis_prob_subtruct_min_factor != 0:
                min_val = np.min(emission_prob[state_i, :])
                emission_prob[state_i, :] -= self.cfg.emis_prob_subtruct_min_factor * min_val

            # normalize the similarities to obtain probabilities
            emission_prob[state_i, :] /= np.sum(emission_prob[state_i, :])

        if not self.using_background:
            self.emission_prob = emission_prob
        else:
            word_dist = self.get_backg_distribution()
            # for all sentences, the word distribution is set to word_dist
            backg_emission_prob = np.tile(word_dist, (self.n_article_sentences, 1))

            self.emission_prob = np.concatenate((emission_prob, backg_emission_prob))

    def hmm_probabilities_init(self):
        self.prepare_start_prob()
        self.prepare_transition_prob()
        self.prepare_emission_prob()

        if self.cfg.debug_mode:
            print("start_prob:")
            print(self.start_prob)
            print("transition_prob:")
            print(self.transition_prob)
            print("emission_prob:")
            print(self.emission_prob)

    def get_num_of_states(self):
        return self.n_states

    def get_num_of_article_sentences(self):
        return self.n_article_sentences

    def get_num_of_gt_sentences(self):
        """
        returns the number of ground-truth sentences (the ones which are labeled as positives)
        """
        if not self.cfg.labeled_data_mode:
            raise Exception("this method is unavailable for unlabeled sample")

        return len(self.gt_unique_sent_ids)

    def get_warnings(self):
        return self.warnings

    def state2sent(self, state_i):
        """
        given state index, this function returns the corresponding sentence index
        (these indices are different only in case background is enabled)
        """
        return state_i % self.n_article_sentences

    def state2backg(self, state_i):
        """
        returns 1 if the given state is in the background, 0 otherwise
        """
        return state_i // self.n_article_sentences

    def state2pair(self, state_i):
        """
        given a state index, returns pair of (sentence index, background value)
        """
        return self.state2sent(state_i), self.state2backg(state_i)

    def predict(self):
        """
        runs the Viterbi algorithm to obtain a predicted sequence of hidden states, i.e. paper sentences
        """
        tprint("predict...")

        if self.cfg.hmm_algo == HmmAlgo.VITERBI_0:
            predicted_path = viterbi(self.start_prob,
                                     self.transition_prob,
                                     self.emission_prob,
                                     self.observed_seq)

        elif self.cfg.hmm_algo == HmmAlgo.DUMMY:
            # for debugging - avoid waiting for prediction
            predicted_path = [20] * len(self.observed_seq)
            predicted_path[:3] = [10, 10, 12]
        else:
            raise Exception("unknown HMM algorithm")

        tprint("done")

        # if going backward is not allowed - validate it
        if not self.cfg.allow_backward_steps:
            for t in range(1, len(self.observed_seq)):
                assert (self.state2sent(predicted_path[t]) >= self.state2sent(predicted_path[t - 1]))

        log_prob = self.calc_log_prob(predicted_path, emission_prob_only=False)

        if self.using_background:
            foreg_pos = [self.state2backg(state_i) == 0 for state_i in predicted_path]
            backg_pos = [not bool_val for bool_val in foreg_pos]

            predicted_sents = list(compress(predicted_path, foreg_pos))

            print('foreground count: {}'.format(len(predicted_path)))
            print('background count: {}'.format(sum(backg_pos)))
        else:
            predicted_sents = predicted_path

        unique_sent_indices = list(set(predicted_sents))
        unique_sent_indices.sort()

        self.prepare_predicted_seq_info(predicted_path)

        print("predicted sequence summary:")
        for subseq_info in self.predicted_seq_info:
            sent_i = subseq_info[PredictedSeqInfoKey.SENT_I.value]
            backg = subseq_info[PredictedSeqInfoKey.BACKGROUND.value]
            if self.using_background:
                state_str = "({:4}, {})".format(sent_i, backg)
            else:
                state_str = "{:4}".format(sent_i)

            duration = subseq_info[PredictedSeqInfoKey.DURATION.value]

            print("{}: {:4}".format(state_str, duration))

        if self.print_predicted_sentences:
            print("predicted sentences:")
            for sent_i in unique_sent_indices:
                print("sentence {}:".format(sent_i))
                print(self.article_sentences[sent_i])

        print("\nnum of predicted unique sentences: {}".format(len(unique_sent_indices)))

        return self.predicted_seq_info, log_prob

    def calc_log_prob(self, path, emission_prob_only):
        """
        calculates the log-probability of a given path of hidden states
        """
        log_prob = 0
        for t in range(len(path)):
            log_prob += np.log(self.emission_prob[path[t], self.observed_seq[t]])

        if emission_prob_only:
            return log_prob

        log_prob += np.log(self.start_prob[path[0]])

        for t in range(1, len(path)):
            log_prob += np.log(self.transition_prob[path[t - 1], path[t]])

        return log_prob

    def prepare_predicted_seq_info(self, predicted_path):
        # will contain the indices where state was changed
        change_indices = []
        prev_state = None
        all_subseq_info = []
        observed_seq_len = len(self.observed_seq)
        self.durations = np.zeros(self.n_article_sentences, dtype=np.int)

        # collect the indices where state was changed
        for t in range(observed_seq_len):
            cur_state = predicted_path[t]
            if cur_state != prev_state:
                change_indices.append(t)
                prev_state = cur_state
        # this will aid in the next loop
        change_indices.append(observed_seq_len)

        # we start at the second index
        for i in range(1, len(change_indices)):
            t = change_indices[i]
            prev_t = change_indices[i - 1]

            cur_state_i = predicted_path[prev_t]
            cur_sent_i, cur_backg = self.state2pair(cur_state_i)
            if self.cfg.labeled_data_mode:
                cur_sent_id = self.sentences_full_indices[cur_sent_i]
                is_ground_truth = int(cur_sent_id in self.gt_unique_sent_ids)
            else:
                cur_sent_id = ''
                is_ground_truth = 0
            cur_sent_text = self.article_sentences[cur_sent_i]
            observed_subseq = self.observed_seq[prev_t:t]
            if self.cfg.transcript_word_level_mode:
                spoken_words_subseq = [self.id2word[word_i] for word_i in observed_subseq]
                spoken_words_str = ' '.join(spoken_words_subseq)
            else:
                spoken_sents = [' '.join(self.transcript_sents[tran_sent_i]) for tran_sent_i in observed_subseq]
                spoken_words_str = ' <EOS> '.join(spoken_sents)

            duration = len(observed_subseq)

            cur_subseq_info = {
                PredictedSeqInfoKey.SENT_I.value: cur_sent_i,
                PredictedSeqInfoKey.BACKGROUND.value: cur_backg,
                PredictedSeqInfoKey.SENT_FULL_ID.value: cur_sent_id,
                PredictedSeqInfoKey.SENT_TEXT.value: cur_sent_text,
                PredictedSeqInfoKey.DURATION.value: duration,
                PredictedSeqInfoKey.SPOKEN_WORDS.value: spoken_words_str,
                PredictedSeqInfoKey.IS_GROUND_TRUTH.value: is_ground_truth
            }

            all_subseq_info.append(cur_subseq_info)

            # update durations if foreground
            if cur_backg == 0:
                # the same sentence might appear several times in the path with backg == 0
                self.durations[cur_sent_i] += duration

        self.predicted_seq_info = all_subseq_info

    def sent_ids_to_str(self, sent_ids):
        """
        given a list of full indices of sentences, this method creates a string of
        the corresponding sentences, separated by newlines
        """
        sentences = [self.full_index_to_sentence[sent_id] for sent_id in sent_ids]
        out_str = '\n'.join(sentences) + '\n'
        return out_str

    def get_summary_sent_indices(self, duration_thresh=1):
        if self.durations is None:
            raise Exception("you must call predict() before calling assess()")

        summary_sent_indices = []
        for sent_i, duration in enumerate(self.durations):
            if duration >= duration_thresh:
                summary_sent_indices.append(sent_i)

        return summary_sent_indices

    def get_summary_sent_ids(self, duration_thresh=1):
        """
        returns the full indices of the chosen sentences
        """
        summary_sent_indices = self.get_summary_sent_indices(duration_thresh)
        summary_sent_ids = [self.sentences_full_indices[i] for i in summary_sent_indices]
        return summary_sent_ids

    def assess(self, duration_thresh=1):
        """
        This function is relevant only for labeled data
        duration_thresh: sentences which were included in the predicted path, but with duration less
        than duration_thresh, will be excluded from the summary
        """
        if not self.cfg.labeled_data_mode:
            raise Exception("you can call this function in labeled-data-mode only")

        print("duration_thresh = {}".format(duration_thresh))
        summary_sent_ids = self.get_summary_sent_ids(duration_thresh)

        # print("summary_sent_ids:")
        # print(summary_sent_ids)

        num_gt_sentences = len(self.gt_unique_sent_ids)
        print("number of unique ground-truth sentences: {}".format(num_gt_sentences))

        prediction_labels = [sent_id in self.gt_unique_sent_ids for sent_id in summary_sent_ids]
        # print("prediction_labels:")
        # print(prediction_labels)

        true_positives = sum(prediction_labels)

        precision = true_positives / len(prediction_labels)
        recall = true_positives / num_gt_sentences
        # avoid division by zero
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        summary_len = len(summary_sent_ids)

        return precision, recall, f1, summary_len

    def get_summary_num_of_sents(self, duration_thresh=1):
        summary_sent_indices = self.get_summary_sent_indices(duration_thresh)
        summary_num_of_sents = len(summary_sent_indices)
        return summary_num_of_sents

    def get_durations_including_excluded_sents(self):
        """
        combines zero values into the durations_vector at the locations of the excluded sentences
        """
        durations = np.zeros(len(self.durations) + len(self.excluded_sent_indices), dtype=np.int)
        idx_reduced = 0
        idx_all = 0

        while idx_all < len(durations):
            if idx_all not in self.excluded_sent_indices:
                durations[idx_all] = self.durations[idx_reduced]
                idx_reduced += 1
            idx_all += 1

        assert idx_reduced == len(self.durations)

        return durations

    def create_durations_file(self, out_fname):
        durations = self.get_durations_including_excluded_sents()

        out_str = '\n'.join([str(val) for val in durations]) + '\n'

        with open(out_fname, 'w') as out_file:
            out_file.write(out_str)

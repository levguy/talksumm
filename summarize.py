import sys
import os
from multiprocessing import Pool
import math
import argparse
import copy
import itertools

from Logger import Logger
from util import print_table, files_in_dir
from SummaryCreator import SummaryCreator
from ArticleDataSample import ArticleDataSample
from HmmArticle import HmmArticle, HmmArticleConfig, PredictedSeqInfoKey


def summarize(args):
    col_order = PredictedSeqInfoKey.get_columns_order()
    failed_articles = []

    articles_folder = os.path.join(args.data_folder, "text")
    transcript_folder = os.path.join(args.data_folder, "transcript")
    sections_info_folder = os.path.join(args.data_folder, "sections_info")
    section_per_sent_folder = os.path.join(args.data_folder, "section_per_sent")

    article_names = args.article_names
    print("number of articles: {}".format(len(article_names)))

    predict_enable = not args.no_predict
    # log only if we are in predict mode
    logging_enable = predict_enable

    for article_i, article_name in enumerate(article_names):
        if logging_enable:
            # set up log file for current article
            log_filename = os.path.join(args.log_folder, article_name)
            if os.path.isfile(log_filename):
                raise Exception("log file already exists: {}".format(log_filename))

            logger = Logger(log_filename)
            sys.stdout = sys.stderr = logger
            print("Logging to file: {}\n".format(log_filename))

        print("--- paper {}: {}\n".format(article_i, article_name))

        article_fname = os.path.join(articles_folder, article_name)
        transcript_fname = os.path.join(transcript_folder, article_name)
        sections_info_fname = os.path.join(sections_info_folder, article_name)
        section_per_sent_fname = os.path.join(section_per_sent_folder, article_name)

        # remove the ".txt" extension and add numpy extension
        similarity_fname = article_name[:-4] + '.npy'
        similarity_fname = os.path.join(args.similarity_folder, similarity_fname)

        try:
            article_data_sample = ArticleDataSample(transcript_fname,
                                                    article_fname,
                                                    sections_info_fname,
                                                    section_per_sent_fname)

            # prepare configuration
            cfg = HmmArticleConfig(args.word_embed_path, labeled_data_mode=False)
            cfg.similarity_fname = similarity_fname

            cfg.print_configuration()
            print("")

            durations_folder = os.path.join(args.base_summaries_folder, "durations")
            os.makedirs(durations_folder, mode=0o775, exist_ok=True)
            durations_fname = os.path.join(durations_folder, article_name)

            alignment_folder = os.path.join(args.base_summaries_folder, "alignment")
            os.makedirs(alignment_folder, mode=0o775, exist_ok=True)
            alignment_fname = os.path.join(alignment_folder, article_name)

            top_scored_sents_folder = os.path.join(args.base_summaries_folder,
                                                   "top_scored_sents.num_sents_{}_thresh_{}".format(args.num_sents,
                                                                                                    args.thresh))
            os.makedirs(top_scored_sents_folder, mode=0o775, exist_ok=True)
            top_scored_sents_fname = os.path.join(top_scored_sents_folder, article_name)

            if predict_enable:
                hmm_article = HmmArticle(article_data_sample, cfg)

                predicted_seq_info, log_prob = hmm_article.predict()

                print("log_prob = {}".format(log_prob))

                print("predicted sequence info:\n")
                alignment_str = print_table(predicted_seq_info, col_order)
                with open(alignment_fname, 'w') as out_file:
                    out_file.write(alignment_str + "\n")

                print("\n")

                hmm_article.create_durations_file(durations_fname)

            summary_creator = SummaryCreator(article_data_sample,
                                             durations_fname=durations_fname)

            if os.path.isfile(top_scored_sents_fname):
                print("file exists: {}".format(top_scored_sents_fname))
            else:
                summary_creator.create_top_scored_sents_file(args.num_sents,
                                                             args.thresh,
                                                             top_scored_sents_fname)

            if predict_enable:
                warnings = hmm_article.get_warnings()
                if len(warnings) > 0:
                    for warning in warnings:
                        print("- {}".format(warning))

        except Exception as ex:
            print("EXCEPTION WAS CAUGHT FOR PAPER: {}".format(article_name))
            print(ex)
            failed_articles.append(article_name)

    return failed_articles


def main(args):
    predict_enable = not args.no_predict

    os.makedirs(args.out_folder, mode=0o775, exist_ok=True)

    # take the basename and remove the extension
    word_embed_description = os.path.basename(args.word_embed_path)[:-4]

    experiment_folder = f'embed_{word_embed_description}'

    args.base_summaries_folder = os.path.join(args.out_folder, experiment_folder, "output")
    os.makedirs(args.base_summaries_folder, mode=0o775, exist_ok=(not predict_enable))

    args.similarity_folder = os.path.join(args.out_folder, experiment_folder, "similarity")
    os.makedirs(args.similarity_folder, mode=0o775, exist_ok=True)
    args.log_folder = os.path.join(args.base_summaries_folder, "log")
    os.makedirs(args.log_folder, mode=0o775, exist_ok=(not predict_enable))

    article_names = files_in_dir(os.path.join(args.data_folder, "transcript"))

    num_processors = args.num_processors
    print("num_processors: {}".format(num_processors))

    if args.num_processors > 1:  # multiprocessing
        num_articles = len(article_names)
        papers_per_process = math.ceil(num_articles / num_processors)
        args_list = [copy.copy(args) for _ in range(num_processors)]
        for i in range(num_processors):
            args_list[i].article_names = article_names[i*papers_per_process: (i+1)*papers_per_process]

        p = Pool(num_processors)
        failed_lists = p.map(summarize, args_list)

        # list of lists -> one list
        failed_list = list(itertools.chain.from_iterable(failed_lists))

    else:  # run on single processor
        args.article_names = article_names
        failed_list = summarize(args)

    num_failed = len(failed_list)
    if num_failed > 0:
        print("FAILED ARTICLES ({}):".format(num_failed))
        for article_name in failed_list:
            print(article_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This script applies the HMM to generate scores for the papers sentences, and to create summaries'
    )
    parser.add_argument('--data_folder',
                        help='data folder')
    parser.add_argument('--out_folder',
                        help='output folder')
    parser.add_argument('--word_embed_path',
                        help='path to word embedding file (both GloVe & word2vec bin-file formats are supported')
    parser.add_argument('--num_processors', type=int, default=1,
                        help='number of processors (use 1 to avoid multiprocessing)')
    parser.add_argument('--no_predict', action='store_true',
                        help='disable HMM prediction (relevant if you have already applied the HMM and obtained'
                             'sentence scores)')
    parser.add_argument('--num_sents', type=int, default=30,
                        help='desired number of top-scored sentences in the generated summary. '
                             'sentences will be retrieved only if their duration is at least \'thresh\', which means '
                             'that the number of retrieved sentences might be smaller than \'num_sents\'')
    parser.add_argument('--thresh', type=int, default=1,
                        help='duration threshold for retrieving sentences, as described in the help of \'num_sents\'')

    args = parser.parse_args()
    main(args)

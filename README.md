# TalkSumm

This repository contains a dataset and related code for the ACL 2019 paper *[TalkSumm: A Dataset and Scalable Annotation Method for Scientific Paper Summarization Based on Conference Talks](https://www.aclweb.org/anthology/P19-1204)*.

The dataset contains 1705 automatically-generated summaries of scientific papers from ACL, NAACL, EMNLP, SIGDIAL (2015-2018), and ICML (2017-2018).
We are not allowed to publish the original papers, however they are publicly available.
For downloading them, refer to *data/talksumm_papers_urls.txt* which contains the papers' titles and URLs,
and to the script *data/get_pdfs.py*, contributed by Tomas Goldsack, which can be used for downloading the pdf files of the papers.
The summaries can be found at *data/talksumm_summaries.zip*. The name of each summary file is the title of the corresponding paper.
 
Using our code, you can generate summaries given papers and transcripts of their conference talks.
Below are instructions for running our code.

If you use this repository, please cite our paper:
```
@inproceedings{lev-etal-2019-talksumm,
    title = "{T}alk{S}umm: A Dataset and Scalable Annotation Method for Scientific Paper Summarization Based on Conference Talks",
    author = "Lev, Guy  and
      Shmueli-Scheuer, Michal  and
      Herzig, Jonathan  and
      Jerbi, Achiya  and
      Konopnicki, David",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1204",
    pages = "2125--2131",
    abstract = "Currently, no large-scale training data is available for the task of scientific paper summarization. In this paper, we propose a novel method that automatically generates summaries for scientific papers, by utilizing videos of talks at scientific conferences. We hypothesize that such talks constitute a coherent and concise description of the papers{'} content, and can form the basis for good summaries. We collected 1716 papers and their corresponding videos, and created a dataset of paper summaries. A model trained on this dataset achieves similar performance as models trained on a dataset of summaries created manually. In addition, we validated the quality of our summaries by human experts.",
}
```

# Running the TalkSumm Model

## Python Environment
We used Python 3.6.8. In requirements.txt you can find the requirements.

## Word Embedding
Our model relies on word embedding. We used the pre-trained GloVe
trained on Wikipedia 2014 + Gigaword, available at https://nlp.stanford.edu/projects/glove.
You can download it by calling:

```
wget http://nlp.stanford.edu/data/glove.6B.zip
```

This zip file contains several versions, we used the 300-dimensional one
(glove.6B.300d.txt). It is uncased.

## Preparing Input for the TalkSumm Model
For summarizing a paper, the input to the TalkSumm Model is as
follows: (1) A text file containing the paper's sentences, each sentence
in a separate line; (2) Optional text files containing information about
the paper's sections; (3) A text file containing the transcript of the
paper's conference talk.

### Preparing Paper Text Files
Given papers in pdf format, we used [science-parse](https://github.com/allenai/science-parse) to convert them to structured json files.
After you prepare the papers' json files in a folder, use *prepare_data_for_hmm.py* to process the json files and create the needed text files representing the papers.
In the folder *example/json* we provide a single json file of a paper.
Processing it can be done by calling:

```
python prepare_data_for_hmm.py --json_folder=example/json --out_folder=example --glove_path=/path/to/glove/embedding
```

*prepare_data_for_hmm.py* creates three folders:

*text*: Contains the text files of the papers (a sentence per line).

*sections_info*: Each file in this folder contains the sentence index range (first index and number of sentences) of the Introduction, Related Work, and Acknowledgments sections (in case they are identified) of the corresponding paper.

*section_per_sent*: Each file in this folder stores, at line *i*, the title of the section to which sentence *i* belongs.

### Preparing Transcripts
Additional input to the TalkSumm model is the transcript of the paper's conference talk.
Please refer to our paper where we describe how we prepared the transcript files.
In the folder *example/transcript* we provide a transcript file for the example paper (this transcript file contains multiple lines, but a file containing all text in a single line is fine as well).

## Generating Summaries
After preparing the input for the TalkSumm model, you can run it to obtain importance scores for the paper's sentences. Then, creating a summary can be done by taking a subset of top-ranked sentences, up to a desired summary length.
The relevant script is *summarize.py*, which goes over the papers in *data_folder*, and for each paper, it creates a corresponding HMM model, obtains the sentence-scoring, and creates a summary of *num_sents* sentences. Sentences with score less than *thresh* will not be added to the summary, so the resulting summary might contain less than *num_sents* sentences.
Running this script on our example paper can be done as follows:

```
python summarize.py --data_folder=example --out_folder=example/output --word_embed_path=/path/to/glove/embedding --num_sents=30 --thresh=1
```

In case you run this script on a large number of papers, you can reduce execution time by multiprocessing - use the *num_processors* argument to set the desired number of processors (by default it is set to 1).

Assuming that you use the glove.6B.300d word embedding, the "experiment name" will be *embed_glove.6B.300d*, and a folder of that name will be created for the experiment, under *example/output*.
Under this folder, the following folders will be created, containing the following files:

*similarity*: A file storing the similarities between each transcript word and each paper sentence. This enables faster re-running of the HMM after changing its parameters in a way that doesn't affect the word-sentence similarities.

*output/durations*: A file containing the "duration" of each paper sentence, i.e. the number of time-steps in which the sentence was chosen by the Viterbi algorithm. This models the number of words uttered by the speaker to describe the sentence, and can be used as importance score.

*output/top_scored_sents.num_sents_30_thresh_1*: A file containing the summary of the paper. It consists of the *num_sents* top-ranked sentences, with duration at least *thresh*. The format of this file is as follows:
- Each line contains: sentence index (in original paper), sentence score (i.e. duration), then the sentence itself. The fields are tab-separated.
- The order of the sentences is according to their order in the paper.

*output/alignment*: A file containing a table showing the alignment between the transcript words and the paper sentences, as obtained by the HMM. The format of this table is the same as Table 4 in the appendix of our paper.

*output/log*: A log file of the run.

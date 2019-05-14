# Introduction

This repository is a Python implementation to generate term pairs of the English Bigram Relatedness Dataset (BiRD). Details on how we created the BiRD can be found in our paper: [Big BiRD: A Large, Fine-Grained, Bigram Relatedness Dataset for Examining Semantic Composition](http://saifmohammad.com/WebDocs/BiRD-NAACL2019.pdf).

BiRD can be used for two purposes: (1) to evaluate methods of semantic composition and (2) to analyse to obtain insights into bigram semantic relatedness. Both purposes are studied in detail in [our paper](http://saifmohammad.com/WebDocs/BiRD-NAACL2019.pdf). An interactive visualizations of the data to explore and the annotation questionnaire which is used for data annotation are also available through the [project's webpage](http://saifmohammad.com/WebPages/BiRD.html). 

# Usage

## Dependencies

Requirements:
- Python3
- nltk>=3

Install required Python dependencies using the command below.
pip install -r python/requirements.txt

## Files

1. `generate_related_terms.py` Main script to obtain the related terms for any bigram AB.
2. `bigrams_file.txt` List of bigrams of which the related terms will be extracted. Each line is a bigram.
3. `phrase_table_unigram_bigram.txt` This file is the output of the processed phrase table of [NRC Portage Machine Translation Toolkit](http://www.aclweb.org/anthology/W10-1717). It only contains one-word and two-word English translations of french phrases (disregarding frequency).
4. `wikipedia_unigram_bigram.pickle` The list of unigram and bigrams and their frequencies in Wikipedia. We used English Wikipedia dump 2018 in our work. We only consider adjective-noun or noun-noun bigrams.
5. `semantic_composition_evaluation.py` Main script to examine semantic composition methods on BiRD. This script calls the word embeddings from fastText, GloVe and term_context matrix from the `word_embeddings` folder in the project and reproduces the results of the Table 4 in the paper. Note that these results were obtained from a subset of BiRD (3,159 pairs) since some words in BiRD do not occur in some of the corpora used to create the word vectors embeddings.

## Folders

 `word_embeddings.zip`  contains three embedding files. Unzip the file for further process. The files `fasttext.txt`, `glove.txt` and `term-context.txt` in the folder contain only the words and their vectors which occur on a subset of BiRD (3,159 pairs). 


## Running the code

In order to obtain the related terms for any bigram, run the script `generate_related_terms.py`. The command to run the script is as follows:

`python generate_related_terms.py -p2pt PATH_TO_PHRASE_TABLE -p2wikipedia PATH_TO_WIKIPEDIA -freq 30 -syn_number 5 -p2in PATH_TO_INPUT -p2out PATH_TO_OUTPUT`

where

- *p2pt* is the path to the generated  `phrase_table_unigram_bigram.txt` file from phrase table,
- *p2wikipedia* is the path to the generated `wikipedia_unigram_bigram.pickle` file from Wikipedia,
- *freq* is the frequency threshold of English phrases in the phrase table,
- *syn_number* is the maximum number of required related term for each bigram AB,
- *p2in* is the path to the input file `bigrams_file.txt`,
- *p2out* is the path to the output file where all related terms of the bigrams is saved.


If *p2bigram* argument is not set in the argument list, you are asked to enter a bigram as the input after running the script. In this case, if you type *exit* it quits the program.
If *p2out* is not set in the argument list, the obtained related terms are not saved in the output file.
If a bigram does not exist in WordNet or phrase table, the program outputs "not found" into STDOUT.

In order to examine methods of semantic composition on BiRD run the script `semantic_composition_evaluation.py` as follows:

`python semantic_composition_evaluation.py -p2bird PATH_TO_BiRD -p2embedding PATH_TO_WORD_EMBEDDING_TEXT_FILE -p2out PATH_TO_OUTPUT_FILE`

For example:

`python semantic_composition_evaluation.py -p2bird BiRD.txt -p2embedding word_embeddings/term-context.txt -p2out output.txt`

where:
-p2bird is the path to the BiRD file,
-p2embedding is the path to the embedding file. In this project, three embedding files from fastText, glove and term-context matrix are available in the `word_embeddings` folder. You can use other word embeddings. The file format should be as follows:
<word><space><vector>
please see the files in the `word_embedding` folder.
-p2out is the path to write the results of the semantic composition evaluation to the output file.

# Reference

Please cite our paper [1] to reference to our dataset or code.

[1] Big BiRD: A Large, Fine-Grained, Bigram Relatedness Dataset for Examining Semantic Composition. Shima Asaadi, Saif M. Mohammad, and Svetlana Kiritchenko. In Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL-2019), June 2019, Minnesota, USA.

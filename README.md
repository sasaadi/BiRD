# Introduction

A python implementation to generate term pairs of the English Bigram Relatedness Dataset (BiRD). Details on how we created the BiRD can be found in our paper: [Big BiRD: A Large, Fine-Grained, Bigram Relatedness Dataset: A Resource For Examining Semantic Composition](URL).

BiRD can be used for two purposes: (1) to evaluate methods of semantic composition and (2) to analyse to obtain insights into bigram semantic relatedness. Both purposes are studied in detail in [our paper](URL). The visualizations of the data and the annotation questionnaire are also available through the [project's webpage](http://saifmohammad.com/WebPages/BiRD.html). 

# Usage

## Dependencies

Requirements:
- Python3
- nltk>=3

Install required Python dependencies using the command below.
pip install -r python/requirements.txt

## Files

1. `data.zip` is a compressed file which contains `phrase_table_unigram_bigram.txt` and `wikipedia_unigram_bigram.pickle` files. Unzip the files to proceed.
2. `generate_related_terms.py` Main script to obtain the related terms for any bigram AB.
3. `bigrams_file.txt` List of bigrams of which the related terms will be extracted. Each line is a bigram.
4. `phrase_table_unigram_bigram.txt` This file is the output of the processed phrase table of [NRC Portage Machine Translation Toolkit](http://www.aclweb.org/anthology/W10-1717). It only contains one-word and two-word English translations of french phrases (disregarding frequency).
5. `wikipedia_unigram_bigram.pickle` The list of unigram and bigrams and their frequencies in Wikipedia. We used English Wikipedia dump 2018 in our work. We only consider adjective-noun or noun-noun bigrams. 

## Running the code

In order to obtain the related terms for any bigram, run the script `generate_related_terms.py`. The command to run the script is as follows:

`python generate_related_terms.py -p2pt PATH_TO_PHRASE_TABLE -p2wikipedia PATH_TO_WIKIPEDIA -freq 30 syn_number 5 -p2in PATH_TO_INPUT -p2out PATH_TO_OUTPUT`

where:
-p2pt is the path to the generated  `phrase_table_unigram_bigram.txt` file from phrase table,
-p2wikipedia is tha path to the generated `wikipedia_unigram_bigram.pickle` file from Wikipedia,
-freq is the frequency threshold of English phrases in the phrase table,
-syn_number is the maximum number of required related term for each bigram,
-syn_number is the maximum number of required related term for each AB,
-p2in is the path to the input file `bigrams_file.txt`,
-p2out is the path to the output file where all related terms of the bigrams is saved.

If p2bigram argument is not set in the argument list, you are asked to enter a bigram as the input after running the script. In this case, if you type 'exit', it quits the program.
If p2out is not set in the argument list, the obtained related terms are not saved in the output file.
If a bigram does not exist in WordNet or phrase table, the program outputs "not found" into STDOUT.

# Reference

Please cite our paper [1] to reference to our dataset or code.

[1] Big Bird: A Large, Fine-Grained, Bigram Relatedness Dataset. Shima Asaadi, Saif M. Mohammad, and Svetlana Kiritchenko. In Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL-2019), June 2019, Minnesota, USA.


from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from operator import itemgetter
import re
import argparse
import logging
import pickle
import os, sys

wnl = WordNetLemmatizer()

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)



# Check eligibility of a synonym unigram term
def eligible_syn(name, wiki_phrases, bigram, syn_freq):
    black_list = ["canada", "canadian"]
    name = wnl.lemmatize(name)
    if bool(re.compile(r'[^A-Z.]').search(name)):
        name = name.lower()
        if set(name.split()).isdisjoint(bigram.split()) and \
        set(name.split()).isdisjoint(["".join(bigram.split())]) and \
        bool(wn.synsets(name)) and str(wn.synsets(name)[0]).split(".")[1] == 'n' and \
        (name in wiki_phrases) and wiki_phrases[name] >= syn_freq and not(name in black_list):
            return True
        else:
            False
    else:
        return False



# Check eligibility of a synonym bigram term
def eligible_bi_syn(bi_word, wiki_phrases, bigram, syn_freq):
    words = bi_word.lower().split()
    words[0] = wnl.lemmatize(words[0])
    words[1] = wnl.lemmatize(words[1])
    if set(words).isdisjoint(bigram.split()) and \
    bool(wn.synsets(words[1])) and str(wn.synsets(words[1])[0]).split(".")[1] == 'n' and \
    (bi_word in wiki_phrases) and wiki_phrases[bi_word] >= syn_freq:
        return True
    else:
        return False



# Check eligibility of a related term
def eligible_relation_candidate(lemma, name, bigram):
    if bool(re.compile(r'[^A-Z.]').search(name)):
        name = name.lower()
        if str(lemma).split(".")[1] == 'n' and (not name in bigram) and len(name.split("_")) <= 2:
            return True
    return False



# Check eligibility of a related term
def eligible_candidate(lemma, name, bigram):
    if bool(re.compile(r'[^A-Z.]').search(name)):
        name = name.lower()
        if str(lemma).split(".")[1] == 'n' and set(name.split("_")).isdisjoint(bigram.split("_")) and \
        set(name.split("_")).isdisjoint(["".join(bigram.split("_"))]):
            return True
    return False



# Extract hypernym of a bigram if exist and is eligible
def get_hypernyms(syn, bigram):
    hypernyms = []
    if bool(syn.hypernyms()):
        for s in syn.hypernyms():
            for l in s.lemmas():
                name = l.name()
                if eligible_relation_candidate(l, name, bigram.lower()):
                    hypernyms.append(" ".join(name.split("_")).lower())
                    return hypernyms
    return hypernyms



# Extract hyponym of a bigram if exist and is eligible
def get_hyponyms(syn, bigram):
    hyponyms = []
    if bool(syn.hyponyms()):
        for s in syn.hyponyms():
            for l in s.lemmas():
                name = l.name()
                if eligible_relation_candidate(l, name, bigram.lower()):
                    hyponyms.append(" ".join(name.split("_")).lower())
                    return hyponyms
    return hyponyms



# Extract holonym of a bigram if exist and is eligible
def get_holonyms(syn, bigram):
    holonyms = []
    if bool(syn.member_holonyms()):
        for s in syn.member_holonyms():
            for l in s.lemmas():
                name = l.name()
                if eligible_relation_candidate(l, name, bigram.lower()):
                    holonyms.append(" ".join(name.split("_")).lower())
                    return holonyms
    return holonyms



# Extract meronym of a bigram if exist and is eligible
def get_meronyms(syn, bigram):
    meronyms = []
    if bool(syn.part_meronyms()):
        for s in syn.part_meronyms():
            for l in s.lemmas():
                name = l.name()
                if eligible_relation_candidate(l, name, bigram.lower()):
                    meronyms.append(" ".join(name.split("_")).lower())
                    return meronyms
    return meronyms



# Check if the second word of a bigram is a Noun in WordNet
def check_pos(synonym):
    words = synonym.split("_")
    if bool(wn.synsets(words[1])):
        for s in wn.synsets(words[1]):
            for l in s.lemmas():
                if str(l).split(".")[1] == 'n':
                    return True



# Extract unigram and bigram synonyms from WordNet
def get_synonyms(syn, bigram, count):
    bi_syn = []
    uni_syn = []
    syn_flag = True
    bi_syn_flag = True
    for l in syn.lemmas():
        name = l.name()
        if len(name.split("_")) == 1 and eligible_candidate(l, name, bigram.lower()) and syn_flag:
            uni_syn.append(name.lower())
            if len(uni_syn) == count:
                syn_flag = False
        if len(name.split("_")) == 2 and eligible_candidate(l, name, bigram.lower()) and bi_syn_flag:
            if bool(check_pos(name)):
                bi_syn.append(" ".join(name.split("_")).lower())
                if len(bi_syn) == count:
                    bi_syn_flag = False
    return uni_syn, bi_syn



# Extract all types of related terms from WordNet
def get_all_relations(bigram, out_to_file, count):
    re_bigram = " ".join(reversed(bigram.split()))
    phrase = "_".join(bigram.split())
    if bool(wn.synsets(phrase)):
        syn = wn.synsets(phrase)[0]
        [uni_syn, bi_syn] = get_synonyms(syn, bigram, count)  # Get unigram and bigram synonyms from WN
        if len(uni_syn) > 0:
            print("Unigram synonyms:", ', '.join(item for item in uni_syn))
            if out_to_file:
                write_to_file(out_to_file, bigram, uni_syn, "WordNet_synonym")
        else:
            print("Unigram synonyms: not found")

        if len(bi_syn) > 0:
            print("Bigram synonyms:", ', '.join(item for item in bi_syn))
            if out_to_file:
                write_to_file(out_to_file, bigram, bi_syn, "WordNet_synonym")
        else:
            print("Bigram synonyms: not found")

        hypernyms = get_hypernyms(syn, bigram)  # Get hypernyms from WN
        if len(hypernyms) == 0:
            print("Hypernyms: not found")
        else:
            print("Hypernyms:", ', '.join(item for item in hypernyms))
            if out_to_file:
                write_to_file(out_to_file, bigram, hypernyms, "WordNet_is-a")

        hyponyms = get_hyponyms(syn, bigram)  # Get hyponyms from WN
        if len(hyponyms) == 0:
            print("Hyponyms: not found")
        else:
            print("Hyponyms:", ', '.join(item for item in hyponyms))
            if out_to_file:
                write_to_file(out_to_file, bigram, hyponyms, "WordNet_is-a")

        holonyms = get_holonyms(syn, bigram)  # Get holonyms from WN
        if len(holonyms) == 0:
            print("Holonyms: not found")
        else:
            print("Holonyms:", ', '.join(item for item in holonyms))
            if out_to_file:
                write_to_file(out_to_file, bigram, holonyms, "WordNet_part-whole")

        meronyms = get_meronyms(syn, bigram)  # Get meronyms from WN
        if len(meronyms) == 0:
            print("Meronyms: not found")
        else:
            print("Meronyms:", ', '.join(item for item in meronyms))
            if out_to_file:
                write_to_file(out_to_file, bigram, meronyms, "WordNet_part-whole")



# Check if bigram exist in WordNet and Call the function to obtain the related terms of the bigram from WordNet
def extract_wordnet_information(bigram, args):
    phrase = "_".join(bigram.split())
    if bool(wn.synsets(phrase)):
        get_all_relations(bigram, args.p2out, args.syn_num)
    else:
        print("Bigram does not exist in WordNet")



# Write the extracted related terms in the output file if set
def write_to_file(path_to_save, bigram, related_terms, relation):
    with open(path_to_save, "a+") as f:
        for term in related_terms:
            f.write(bigram + "\t" + term + "\t" + relation + "\n")



# Obtain at most args.syn_number of most related unigrams and args.syn_number of bigrams, called co-aligned terms
def get_final_terms(all_unigrams, all_bigrams, num, sum_freq, out_to_file):
    final_unigram = dict()
    final_bigram = dict()
    for w, f in all_unigrams.items():
        final_unigram[w] = abs(int(f) - sum_freq)
    for w, f in all_bigrams.items():
        final_bigram[w] = abs(int(f) - sum_freq)
    sorted_bigrams = sorted(final_bigram.items(), key=itemgetter(1))
    sorted_unigrams = sorted(final_unigram.items(), key=itemgetter(1))
    if len(sorted_bigrams) >= num:
        most_related_bi = [sorted_bigrams[i][0].lower() for i in range(num)]
    else:
        most_related_bi = [i[0].lower() for i in sorted_bigrams]
    if len(sorted_unigrams) >= num:
        most_related_uni = [sorted_unigrams[i][0].lower() for i in range(num)]
    else:
        most_related_uni = [i[0].lower() for i in sorted_unigrams]
    if most_related_uni == []:
        print("Most related unigrams: not found")
    else:
        print("Most related unigrams:", ', '.join(item for item in most_related_uni))
        if out_to_file:
            write_to_file(out_to_file, bigram, most_related_uni, "PhraseTable_co-aligned")
    if most_related_bi == []:
        print("Most related bigrams: not found")
    else:
        print("Most related bigrams:", ', '.join(item for item in most_related_bi))
        if out_to_file:
            write_to_file(out_to_file, bigram, most_related_bi, "PhraseTable_co-aligned")



# Extract information from the generated phrase table which only contains one-word and two-word English translations of
# french phrases (disregarding frequency)
def extract_phrasetable_information(args, phrase, wiki_phrases):
    phrase_table = []
    syn_bigrams = []
    sum_freq = 0.0
    freq = int(args.freq)
    num = int(args.syn_num)
    all_bigrams = dict()
    all_unigrams = dict()
    with open(args.p2pt, "r") as fl:
        for line in fl:
            if not line.strip():
                phrase_table.append(syn_bigrams)
                syn_bigrams = []
            else:
                line = line.strip("\n")
                line = line.lstrip("\t")
                if len(line.split("\t")) == 2:
                    line = (line.split("\t")[0], float(line.split("\t")[1]))
                    syn_bigrams.append(line)
    for items in phrase_table:
        for pair in items:
            if pair[0] == phrase and pair[1] >= freq:
                sum_freq += pair[1]
                for i in items:
                    if len(i[0].split()) == 2:
                        if eligible_bi_syn(i[0], wiki_phrases, phrase, freq):
                            words = i[0].lower().split()
                            words[0] = wnl.lemmatize(words[0])
                            words[1] = wnl.lemmatize(words[1])
                            bi_syn = " ".join(words)
                            if bi_syn in all_bigrams:
                                all_bigrams[bi_syn] += i[1]
                            else:
                                all_bigrams[bi_syn] = i[1]
                    elif len(i[0].split()) == 1:
                        if eligible_syn(i[0], wiki_phrases, phrase, freq):
                            word = wnl.lemmatize(i[0].lower())
                            if word in all_unigrams:
                                all_unigrams[word] += i[1]
                            else:
                                all_unigrams[word] = i[1]
    get_final_terms(all_unigrams, all_bigrams, num, sum_freq, args.p2out)



# Input arguments
def get_args():
    parser = argparse.ArgumentParser(
        description="Extract all the related terms for a given bigram.")
    parser.add_argument('-p2pt', '--path-to-generated-phrase-table-unigrams-bigrams', help="Path to the generated file from phrase table (PT)", dest='p2pt')
    parser.add_argument('-p2wikipedia', '--path-to-generated-wikipedia-unigrams-bigrams', help="Path to the generated file from Wikipedia", dest='p2wiki')
    parser.add_argument('-freq', '--frequency-threshold-of-english-phrases-in-phrase-table', help="freq threshold of English phrases in PT", dest='freq')
    parser.add_argument('-syn_number', '--number-of-required-related-terms', help="number of required related terms", dest='syn_num')
    parser.add_argument('-p2in', '--path-to-input-bigrams-file', help="path to input bigrams file", default=None, dest='p2in')
    parser.add_argument('-p2out', '--path-to-output-file', help="path to output file", default=None, dest='p2out')
    args = parser.parse_args()
    return args



# Main function to get the input arguments and read Wikipedia
if __name__ == "__main__":
    args = get_args()
    if args.p2out:
        os.remove(args.p2out)
    with open(args.p2wiki, "rb") as handle:
        wiki_phrases = pickle.load(handle)
    if args.p2in == None:
        while(True):
            bigram = input("Enter a bigram (e.g. radio station) or 'exit': ")
            if bigram == 'exit':
                exit()
            print("***********************************************")
            print("Extracting information from WordNet:")
            print("***********************************************")
            extract_wordnet_information(bigram, args)
            print("\n***********************************************")
            print("Extracting information from PhraseTable:")
            print("***********************************************")
            extract_phrasetable_information(args, bigram, wiki_phrases)
    else:
        logging.debug("Reading from bigram file")
        with open(args.p2in,"r") as f:
            for line in f:
                bigram = line.strip("\n")
                sys.stdout.write("\n-Input bigram '" + str(bigram) + "':" + "\n")
                print("***********************************************")
                print("Extracting information from WordNet:")
                print("***********************************************")
                extract_wordnet_information(bigram, args)
                print("\n***********************************************")
                print("Extracting information from PhraseTable:")
                print("***********************************************")
                extract_phrasetable_information(args, bigram, wiki_phrases)



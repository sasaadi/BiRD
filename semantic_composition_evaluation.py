
import numpy as np
import scipy.stats
import scipy.spatial
import argparse


def read_dataset(path_to_bird):
    dataset = {}
    with open(path_to_bird,"r") as fr:
        titels = fr.readline()  # Skip the titles in the file
        for line in fr:
            s_line = line.split("\t")
            dataset[(s_line[1],s_line[2])] = float(s_line[6])
    return dataset



def get_word_vector_embeddings(path_to_embedding):
    dimension = 300
    model = dict()
    with open(path_to_embedding,"r") as file:
        for line in file:
            word = line.split()[0]
            vector = [float(x) for x in line.split()[1:]]
            vector = np.asarray(vector)
            model[str(word)] = vector
    return model, dimension



# Write the results of pearson and spearman of semantic composition methods to the output file
# path_to_save = "code_for_release"
def write_results_to_output(method, pr, sp, path_to_output):
    with open(path_to_output,"a+") as write_file:
        write_file.write("Method: " + method + ":\n")
        write_file.write("  Pearson: " + str(pr) + "\n")
        write_file.write("  Spearman: " + str(sp) + "\n")
        write_file.write("\n**************************************************\n")



# Additive method
def compute_addition_rankings(model, dataset, a, b, path_to_output):
    predict_scores = []
    target_scores = []
    for items, score in dataset.items():
        if items[0].split()[0] in model and items[0].split()[1] in model:
            out_vec1 = a * model[items[0].split()[0]] + b * model[items[0].split()[1]]
        else:
            continue
        if len(items[1].split()) == 2:
            if items[1].split()[0] in model and items[1].split()[1] in model:
                out_vec2 = a * model[items[1].split()[0]] + b * model[items[1].split()[1]]
            else:
                continue
        else:
            if items[1] in model:
                out_vec2 = model[items[1]]
            else:
                continue
        predict_scores.append(1.0 - float(scipy.spatial.distance.cosine(out_vec1,out_vec2)))
        target_scores.append(score)
    mini = min(predict_scores)
    maxi = max(predict_scores)
    normalized_predict_scores = [(i - mini)/(maxi-mini) for i in predict_scores]
    pearson = scipy.stats.pearsonr(target_scores, normalized_predict_scores)[0]
    spearman = scipy.stats.spearmanr(target_scores, normalized_predict_scores)[0]
    print("Method: vector addition")
    print("Pearson:", pearson)
    print("Spearman:", spearman, "\n")
    write_results_to_output("Vector addition with alpha: " + str(a) + " beta: " + str(b) , pearson, spearman, path_to_output)



# Multiplicative method
def compute_multiplication_rankings(model, dataset, path_to_output):
    predict_scores = []
    target_scores = []
    for items, score in dataset.items():
        if items[0].split()[0] in model and items[0].split()[1] in model:
            out_vec1 = np.multiply(model[items[0].split()[0]], model[items[0].split()[1]])
        else:
            continue
        if len(items[1].split()) == 2:
            if items[1].split()[0] in model and items[1].split()[1] in model:
                out_vec2 = np.multiply(model[items[1].split()[0]], model[items[1].split()[1]])
            else:
                continue
        else:
            if items[1] in model:
                out_vec2 = model[items[1]]
            else:
                continue
        predict_scores.append(1.0 - float(scipy.spatial.distance.cosine(out_vec1,out_vec2)))
        target_scores.append(score)
    mini = min(predict_scores)
    maxi = max(predict_scores)
    normalized_predict_scores = [(i - mini) / (maxi - mini) for i in predict_scores]
    pearson = scipy.stats.pearsonr(target_scores, normalized_predict_scores)[0]
    spearman = scipy.stats.spearmanr(target_scores, normalized_predict_scores)[0]
    print("Method: vector multiplication")
    print("Pearson:", pearson)
    print("Spearman:", spearman, "\n")
    write_results_to_output("Vector multiplication", pearson, spearman, path_to_output)



# Convolution function from Mitchell and Lapata (2010)
def convolve(u, v, dimen):
    convolve = []
    for i in range(dimen):
        x = 0
        for j in range(dimen):
            x += u[j] * v[i - j]
        convolve.append(x)
    return np.array(convolve)



# Convolution product method
def compute_convolution_rankings(model, dataset, dimen, path_to_output):
    predict_scores = []
    target_scores = []
    for items, score in dataset.items():
        if items[0].split()[0] in model and items[0].split()[1] in model:
            out_vec1 = convolve(model[items[0].split()[0]], model[items[0].split()[1]], dimen)
        else:
            continue
        if len(items[1].split()) == 2:
            if items[1].split()[0] in model and items[1].split()[1] in model:
                out_vec2 = convolve(model[items[1].split()[0]], model[items[1].split()[1]], dimen)
            else:
                continue
        else:
            if items[1] in model:
                out_vec2 = model[items[1]]
            else:
                continue
        predict_scores.append(1.0 - float(scipy.spatial.distance.cosine(out_vec1,out_vec2)))
        target_scores.append(score)
    mini = min(predict_scores)
    maxi = max(predict_scores)
    normalized_predict_scores = [(i - mini) / (maxi - mini) for i in predict_scores]
    pearson = scipy.stats.pearsonr(target_scores, normalized_predict_scores)[0]
    spearman = scipy.stats.spearmanr(target_scores, normalized_predict_scores)[0]
    print("Method: convolution product")
    print("Pearson:", pearson)
    print("Spearman:", spearman, "\n")
    write_results_to_output("Convolution product", pearson, spearman, path_to_output)



# Head only method
def compute_head_only_ranking(model, dataset, path_to_output):
    predict_scores = []
    target_scores = []
    for items, score in dataset.items():
        if items[0].split()[0] in model and items[0].split()[1] in model:
            out_vec1 = model[items[0].split()[1]]
        else:
            continue
        if len(items[1].split()) == 2:
            if items[1].split()[0] in model and items[1].split()[1] in model:
                out_vec2 = model[items[1].split()[1]]
            else:
                continue
        else:
            if items[1] in model:
                out_vec2 = model[items[1]]
            else:
                continue
        predict_scores.append(1.0 - float(scipy.spatial.distance.cosine(out_vec1, out_vec2)))
        target_scores.append(score)
    mini = min(predict_scores)
    maxi = max(predict_scores)
    normalized_predict_scores = [(i - mini) / (maxi - mini) for i in predict_scores]
    pearson = scipy.stats.pearsonr(target_scores, normalized_predict_scores)[0]
    spearman = scipy.stats.spearmanr(target_scores, normalized_predict_scores)[0]
    print("Method: head only")
    print("Pearson:", pearson)
    print("Spearman:", spearman, "\n")
    write_results_to_output("Head only", pearson, spearman, path_to_output)



# Modifier only method
def compute_modifier_only_ranking(model, dataset, path_to_output):
    predict_scores = []
    target_scores = []
    for items, score in dataset.items():
        if items[0].split()[0] in model and items[0].split()[1] in model:
            out_vec1 = model[items[0].split()[0]]
        else:
            continue
        if len(items[1].split()) == 2:
            if items[1].split()[0] in model and items[1].split()[1] in model:
                out_vec2 = model[items[1].split()[0]]
            else:
                continue
        else:
            if items[1] in model:
                out_vec2 = model[items[1]]
            else:
                continue
        predict_scores.append(1.0 - float(scipy.spatial.distance.cosine(out_vec1, out_vec2)))
        target_scores.append(score)
    mini = min(predict_scores)
    maxi = max(predict_scores)
    normalized_predict_scores = [(i - mini) / (maxi - mini) for i in predict_scores]
    pearson = scipy.stats.pearsonr(target_scores, normalized_predict_scores)[0]
    spearman = scipy.stats.spearmanr(target_scores, normalized_predict_scores)[0]
    print("Method: modifier only")
    print("Pearson:", pearson)
    print("Spearman:", spearman, "\n")
    write_results_to_output("Modifier only", pearson, spearman, path_to_output)


# Dilation function from Mitchell and Lapata (2010)
def dilation(v, u, lda):
    out = np.ndarray(len(v),dtype=float)
    for i in range(len(v)):
        out = v * sum(u*u) + (lda - 1)*u*sum(u*v)
    return out


# Dilation method
def compute_dilation_ranking(model, dataset, lda, path_to_output):
    predict_scores = []
    target_scores = []
    for items, score in dataset.items():
        if items[0].split()[0] in model and items[0].split()[1] in model:
            head = model[items[0].split()[1]]
            modi = model[items[0].split()[0]]
            out_vec1 = dilation(head,modi,lda)
        else:
            continue
        if len(items[1].split()) == 2:
            if items[1].split()[0] in model and items[1].split()[1] in model:
                head = model[items[1].split()[1]]
                modi = model[items[1].split()[0]]
                out_vec2 = dilation(head, modi, lda)
            else:
                continue
        else:
            if items[1] in model:
                out_vec2 = model[items[1]]
            else:
                continue
        predict_scores.append(1.0 - float(scipy.spatial.distance.cosine(out_vec1, out_vec2)))
        target_scores.append(score)
    mini = min(predict_scores)
    maxi = max(predict_scores)
    normalized_predict_scores = [(i - mini) / (maxi - mini) for i in predict_scores]
    pearson = scipy.stats.pearsonr(target_scores, predict_scores)[0]
    spearman = scipy.stats.spearmanr(target_scores, normalized_predict_scores)[0]
    print("Method: dilation")
    print("Pearson: ", pearson)
    print("Spearmanr: ", spearman, "\n")
    write_results_to_output("Dilation", pearson, spearman, path_to_output)



# Input arguments
def get_args():
    parser = argparse.ArgumentParser(
        description="Examine semantic composition methods in BiRD.")
    parser.add_argument('-p2embedding', '--type-of-word-embedding-tobe-used-to-evaluate-semantic-composition-methods', help="type of word embedding", default=None, dest='p2embedding')
    parser.add_argument('-p2bird', '--path-to-dataset-file', help="path to dataset file", default=None, dest='p2bird')
    parser.add_argument('-p2out', '--path-to-output-file', help="path to output file", default=None, dest='p2out')
    args = parser.parse_args()
    return args




if __name__ == "__main__":

    lda = 2
    dimen = 0
    model = dict()
    args = get_args()
    # Create a model from the requested word embedding
    [model, dimen] = get_word_vector_embeddings(args.p2embedding)
    # Read BiRD dataset from file
    dataset = read_dataset(args.p2bird)
    # Examine all composition methods on BiRD using three types of word embeddings
    for alpha in np.arange(0,1.1,0.1):
        beta = 1 - alpha
        compute_addition_rankings(model, dataset, round(alpha,2), round(beta,2), args.p2out)
    compute_multiplication_rankings(model, dataset, args.p2out)
    compute_convolution_rankings(model, dataset, dimen, args.p2out)
    compute_dilation_ranking(model, dataset, lda, args.p2out)
    compute_head_only_ranking(model, dataset, args.p2out)
    compute_modifier_only_ranking(model, dataset, args.p2out)




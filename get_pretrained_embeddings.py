import json
import numpy as np
import torch


def get_pretrained_embeddings(
        data_folder,
        emb_dim=300):
    """
    embed_dim can be 50/100/200/300
    output: weights_matrix for embedding layer in LSTM

    """
    # from file"WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json" extract vocabulary in training set

    with open(data_folder + "/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json", 'r') as load_f:
        load_dict = json.load(load_f)
    words = load_dict.keys()
    vocabulary = list(words)

    print("number of words in vocabulary:", len(vocabulary))
    # import pretrained glove embeddings
    path = data_folder + "/glove.6B/glove.6B.{}d.txt".format(emb_dim)
    f = open(path)
    embeddings_index = {}
    print('Loading GloVe from:', path, '...', end='')
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    print("Done.\n Proceeding with Embedding Matrix...", end="")

    def glove(word):
        # return embedding_vector
        return embeddings_index[word]

    matrix_len = len(vocabulary)
    weights_matrix = torch.from_numpy(np.zeros((matrix_len, emb_dim)))
    words_found = 0

    # Check whether a word  is on GloVe’s vocabulary
    # If not, initialize a random vector.
    for i, word in enumerate(vocabulary):
        try:
            weights_matrix[i] = torch.from_numpy(glove(word))
            words_found += 1
        except KeyError:
            #print("''{}'' is not in the vocabualory of pretrained Glove".format(word))
            weights_matrix[i] = torch.from_numpy(np.random.uniform(-0.1, 0.1, emb_dim))

    return weights_matrix.float()

#
# data_folder = '../dataset/modified/processed_data'
# weights_matrix = get_pretrained_embeddings(data_folder = data_folder )


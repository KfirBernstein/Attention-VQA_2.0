import sys
import pickle
import numpy as np

# Save a dictionary into a pickle file.
def save_dict_as_pickle(my_dict,fileName):
    print ('pickeling')
    p = pickle.Pickler(open(fileName, "wb"))
    p.fast = True
    p.dump(my_dict)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def load_embeddings(embed_dir):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embed_dir))
    return embedding_index

if __name__== "__main__":
    input = str(sys.argv[1])
    output = str(sys.argv[2])
    glove = load_embeddings(input)
    save_dict_as_pickle(glove, output)
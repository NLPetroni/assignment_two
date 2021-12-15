import os
import requests
import zipfile
import numpy as np
from urllib.request import urlopen
from io import BytesIO

import torch


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_data(data_path):
    toy_data_path = os.path.join(data_path, 'fever_data.zip')
    toy_data_url_id = "1wArZhF9_SHW17WKNGeLmX-QTYw9Zscl1"
    toy_url = "https://docs.google.com/uc?export=download"

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(toy_data_path):
        print("Downloading FEVER data splits...")
        with requests.Session() as current_session:
            response = current_session.get(toy_url,
                                           params={'id': toy_data_url_id},
                                           stream=True)
        save_response_content(response, toy_data_path)
        print("Download completed!")

        print("Extracting dataset...")
        with zipfile.ZipFile(toy_data_path) as loaded_zip:
            loaded_zip.extractall(data_path)
        print("Extraction completed!")


def add_oov(start_voc, oovs, embedding_matrix, sentences):
    """
    Computes new embedding matrix, adding embeddings for oovs
    Parameters:
      start_voc: dict, starting vocabulary that is extended with oovs
      oovs: set of string, oovs to add to the starting vocabulary and embedding matrix
      embedding_matrix: starting embedding matrix (numpy)
      sentences: list of list of strings, set used to compute oov embeddings
    Returns tuple (voc, emb) where voc is dict from words to idx (in emb) and emb is (numpy) embedding matrix with oovs
    """
    oovs = oovs - set(start_voc.keys())
    start_voc_size, emb_size = embedding_matrix.shape
    oov_embeddings = np.zeros((start_voc_size + len(oovs), emb_size))
    oov_embeddings[:start_voc_size] = embedding_matrix
    new_voc = dict(start_voc)

    for i, oov in enumerate(oovs):
        context_words = [new_voc[word]
                         for sentence in filter(lambda s: oov in s, sentences)
                         for word in sentence if word in new_voc and word not in (oov, '<PAD>')]
        oov_embeddings[start_voc_size + i] = np.mean(oov_embeddings[context_words], axis=0)
        new_voc[oov] = start_voc_size + i
    return new_voc, oov_embeddings


def get_glove(emb_size=100, number_token=False):
    """
    Download and load glove embeddings. 
    Parameters:
      emb_size: embedding size (50/100/200/300-dimensional vectors).    
    Returns tuple (voc, emb) where voc is dict from words to idx (in emb) and emb is (numpy) embedding matrix
    """
    n_tokens = 400000 + 1  # glove vocabulary size + PAD
    if emb_size not in (50, 100, 200, 300):
        raise ValueError(f'wrong size parameter: {emb_size}')

    if number_token:
        n_tokens += 1
    download_and_unzip('http://nlp.stanford.edu/data/glove.6B.zip', save_dir='glove')
    vocabulary = dict()
    embedding_matrix = np.ones((n_tokens, emb_size))

    with open(f'glove/glove.6B.{emb_size}d.txt', encoding="utf8") as f:
        for i, line in enumerate(f):
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embedding_matrix[i] = coefs
            vocabulary[word] = i

    # add embedding for and padding and number token
    if number_token:
        embedding_matrix[n_tokens - 2] = 0
        vocabulary['<PAD>'] = n_tokens - 2
        digits = list(filter(lambda s: re.fullmatch('\d+(\.\d*)?', s) is not None, vocabulary.keys()))
        embedding_matrix[n_tokens - 1] = np.mean(embedding_matrix[[vocabulary[d] for d in digits]], axis=0)
        vocabulary['<NUM>'] = n_tokens - 1
    else:
        embedding_matrix[n_tokens - 1] = 0
        vocabulary['<PAD>'] = n_tokens - 1
    return vocabulary, embedding_matrix


def download_and_unzip(url, save_dir='.'):
    # downloads and unzips url, if not already downloaded
    # used for downloading dataset and glove embeddings
    fname = url.split('/')[-1][:-4] if save_dir == '.' else save_dir
    if fname not in os.listdir():
        print(f'downloading and unzipping {fname}...', end=' ')
        r = urlopen(url)
        zipf = zipfile.ZipFile(BytesIO(r.read()))
        zipf.extractall(path=save_dir)
        print(f'completed')
    else:
        print(f'{fname} already downloaded')


def get_wandbkey():
    if os.path.exists('res/wandb_key.txt'):
        with open("res/wandb_key.txt", "r") as txt_file:
            return txt_file.read()
    else:
        import getpass
        key = getpass.getpass('wandb_key.txt is missing. Enter here your key:')
        with open("res/wandb_key.txt", "w") as txt_file:
            txt_file.write(key)
        return key


def __pad_line(line, max_len):
    pad_token = 400000
    res = line.copy()
    diff = max_len - len(line)
    padding = [pad_token for _ in range(diff)]
    res = res + padding
    return res


def pad_data(data):
    """
    Pads and masks every element in data
    Args:
        data: list of tokenized sequences
    Returns:
        padded_data: a torch.Tensor containing padded data
        data_lengths: a list containing original lenghts of the vectors
    """
    data_lengths = [len(d) for d in data]
    max_len = max(data_lens)
    padded_data = []
    for line in data:
        l = __pad_line(line, max_len)
        padded_data.append(l)
    padded_data = torch.as_tensor(padded_data)
    return padded_data, data_lengths

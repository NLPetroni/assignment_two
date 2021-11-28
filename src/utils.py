import os
import requests
import zipfile
import numpy as np

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
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
      embbedding_matrix: starting embedding matrix (numpy)
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
    oov_embeddings[i] = np.mean(oov_embeddings[context_words])
    new_voc[oov] = start_voc_size + i
  return new_voc, oov_embeddings

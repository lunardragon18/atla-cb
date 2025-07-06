from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

class embeds:
    def __init__(self,saved = False):
        self.saved = saved


    def generate_model(self):
        with open('avatar_chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)

        if self.saved:
            indexing = faiss.read_index('avatar_index.faiss')

            return indexing, chunks

        model = SentenceTransformer("all-MiniLM-L6-v2")
        data = [item['text'] for item in chunks]

        embeddings = model.encode(data, convert_to_numpy=True)

        dim = embeddings.shape[1]

        indexing = faiss.IndexFlatL2(dim)
        indexing.add(np.array(embeddings))

        faiss.write_index(indexing , "avatar_index.faiss")
        return indexing, chunks



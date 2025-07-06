import pickle
import numpy as np
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import faiss
from sentence_transformers import SentenceTransformer
import torch

class ChatBox:
    def __init__(self):
        self.base_llm ="meta-llama/Llama-3.1-8B-Instruct"
        self.embed_model = "all-MiniLM-L6-v2"
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.chunks = None
        self.index = None
        self.generate()


    def generate(self):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.float16,
            device_map="auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_llm,
            quantization_config=bnb_config,
            trust_remote_code=True
        ).to("cuda")
        self.model.config.use_cache =False

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.embedder = SentenceTransformer(self.embed_model)
        with open("avatar_chunks.pkl",'rb') as f:
            self.chunks = pickle.load(f)
        self.index = faiss.read_index("avatar_index.faiss")



    def question_answer(self,query):

        query_embedding  = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')



        D , I = self.index.search(np.array(query_embedding) , 5)
        retrieval = [self.chunks[idx]['text'] for idx in I[0]]

        context = "\n\n".join(retrieval)

        sys_prompt = f"""You are an expert bot on the show Avatar : The Last Airbender. A suitable context 
                         will be provide based on the question asked. Use this context to give the best 
                         possible answer. If you cant formulate the answer based on context, say 'I am not
                         sure of the answer.'
                     "Context" : {context}
                     "Question" : {query}
                     "Answer" :  """

        inputs = self.tokenizer([sys_prompt],return_tensors='pt',padding=True).to("cuda")
        output = self.model.generate(
            **inputs,
            max_new_tokens = 256,
            do_sample =True,
            temperature = 0.7,
            top_p = 0.9,
            eos_token_id = self.tokenizer.eos_token_id
        )

        output_d = self.tokenizer.decode(output[0],skip_special_tokens = True)
        answer = output_d.split('Answer')[-1].strip()
        return answer







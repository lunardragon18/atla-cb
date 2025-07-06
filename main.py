import pickle
import gradio as gr
from chatbox.chatbox import ChatBox
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Scrapping.scrapping import Scrapper
from utils.utils import get_data
from embedding.embedding import embeds

def response(query):
    chatbox = ChatBox()
    return chatbox.question_answer(query)

def main():
    scrapper = Scrapper("avatar_characters")
    scrapper.make_data()
    data = get_data("avatar_characters")
    with open('avatar_chunks.pkl' ,'wb') as f:
        pickle.dump(data,f)

    embed_model = embeds()
    embed_model.generate_model()

    iface = gr.Interface(
        fn = response,
        inputs = gr.Textbox(lines = 2, placeholder = "Fire away the question!!!"),
        outputs = "texts",
        title="ATLA Q BOT",
        description="Ask anything about ATLA show"

    )

    iface.launch()







if __name__ == "__main__":
    main()
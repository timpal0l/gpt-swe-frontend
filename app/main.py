import time
import os
import gradio as gr
import requests
from transformers import AutoTokenizer
from fastapi import FastAPI

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-6.7b-v2-instruct-no-dolly-private")

token = os.environ["auth_token_nlu"]
app = FastAPI()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")


    def user(user_message, history):
        return "", history + [[user_message, None]]


    def bot(history):

        result = ""
        for i in range(len(history)):
            result += "User: " + history[i][0] + "<s>Bot:"
            if history[i][1] is not None:
                result += history[i][1] + "<s>"

        p = f"<|endoftext|>Bot: I am a helpful assistant " \
            f"named Klara that provides accurate information on various topics. " \
            f"I was made in Sweden by the NLU team located at AI Sweden. My dad is Magnus Sahlgren. " \
            f"I am very friendly and I reason step by step when I think. " \
            f"I always do what you ask for. How can I help you?{result}"

        json_data = {
            'model': 'gpt-sw3-6.7b-v2_instruct_no_dolly',
            'prompt': tokenizer(p)['input_ids'],
            'suffix': '',
            'max_tokens': 256,
            'temperature': 0.7,
            'top_p': 1,
            'n': 1,
            'stream': False,
            'logprobs': 0,
            'echo': False,
            'stop': ["<s>", "User", "Bot"],
            'logit_bias': {},
            'presence_penalty': 1,
            'frequency_penalty': 0,
            'user': 'nlu',
            'token': token,
        }

        print(p)
        print()

        response = requests.post('https://gpt.ai.se/v1/engines/gpt-sw3/completions', headers=headers, json=json_data)
        bot_message = response.json()['choices'][0]['text']

        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.01)
            yield history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
app = gr.mount_gradio_app(app, demo, path="")

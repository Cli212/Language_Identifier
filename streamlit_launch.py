import sys
from io import StringIO

sys.path.append('../')

import json
import streamlit as st
import os
import json
import pandas as pd
import torch
from model import Identifier
from utils import inference
from transformers import (AdamW, AutoTokenizer, get_linear_schedule_with_warmup)

with open('data/data_100.json') as f:
    data = json.load(f)
id2code = dict(zip(range(len(data)), data.keys()))
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('./data/languages.json') as f:
    code2lan = json.load(f)


@st.cache(allow_output_mutation=True)
def instantiate_model():
    # Model
    model = Identifier(tokenizer.vocab_size, None, 768, len(data), None, None, None)
    model.load_state_dict(torch.load('models/model.pt', map_location='cpu'))
    model.to(device)
    return model


model = instantiate_model()
st.title('Language Identifier')
st.write("This identifier is based on pre-trained BERT base multilingual model")
text = st.text_input('Input your sentence here')
if text is not '':
    submit_button = st.button("Identify")
    if submit_button:
        result = inference([text], model, tokenizer, 128, device)[0]
        st.write(f'Your input sentence seems to be in {code2lan[id2code[result]]}')

uploaded_file = st.file_uploader("Choose a file that has a column named `text`")

if uploaded_file is not None:
     # To read file as bytes:
     # bytes_data = uploaded_file.getvalue()
     # st.write(bytes_data)
     #
     # # To convert to a string based IO:
     # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
     # st.write(stringio)
     #
     # # To read file as string:
     # string_data = stringio.read()
     # st.write(string_data)

     # Can be used wherever a "file-like" object is accepted:
     try:
        df = pd.read_csv(uploaded_file)
     except:
        df = pd.read_excel(uploaded_file)
     texts = df['text'].values.tolist()
     result = inference(texts, model, tokenizer, 128, device)
     result = [code2lan[id2code[i]] for i in result]
     df['prediction'] = result
     st.write(df)




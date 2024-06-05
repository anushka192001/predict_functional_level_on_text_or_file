import streamlit as st
import numpy as np
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
import new  
import predict_on_bulk_data
import pandas as pd



def functional_level_prediction(text):
   result = new.find_functional_label(text)
   return result

user_input = st.text_area('Enter Profile Description To Get Its Functional Level')
print(type(user_input))
button = st.button('Predict')

if user_input and button:
    print(type(user_input))
    result = functional_level_prediction(user_input) 
    st.write('functional_level: ',result)

st.write('OR')


uploaded_file = st.file_uploader('Upload a .csv or .json file(format["text","labels"] or format["text"])', type=["csv", "json"])
upload_button = st.button('Predict On File')


if uploaded_file and upload_button:
     if uploaded_file.type == "application/json":
        df = pd.read_json(uploaded_file)
     elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file) 

     st.write('Given data has no. of rows: ',len(df))
     if 'text' not in df.columns:
        st.write('given file do not contain "text" named column')

     elif 'label' not in df.columns:
      text_not_str = df['text'].apply(lambda x: not isinstance(x, str))
      if text_not_str.any():
         st.write('some row has "text" named column type not equal to str instance')
      else:
       resultant_data = predict_on_bulk_data.get_predicted_dataframe(df)
       CSV = resultant_data.to_csv(index=False).encode('utf-8') 
       st.download_button(
        label="Download_with_predicted_labels",
        data=CSV,
        file_name='predicted_labels.csv',
        mime='text/csv',
        )
       
     else:
        text_not_str = df['text'].apply(lambda x: not isinstance(x, str))
        label_not_list = df['label'].apply(lambda x: not (isinstance(x, list) or isinstance(x, str)))

        if text_not_str.any():
         st.write('some row has "text" named column type not equal to str instance')
        elif label_not_list.any():
          st.write('some row has "label" named column type not equal to str or list instance') 
        else:
         resultant_data,hamming_score,flat_score = predict_on_bulk_data.get_predicted_dataframe_and_hammingscore_and_flat_score(df)
         st.write('hamming_score: ',hamming_score)
         st.write('flat_score: ',flat_score)
         CSV = resultant_data.to_csv(index=False).encode('utf-8') 
         st.download_button(
         label="Download_with_predicted_labels",
         data=CSV,
         file_name='predicted_labels.csv',
         mime='text/csv',
         )
      
    

    

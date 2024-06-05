
import numpy as np                
import transformers   
import torch  
import warnings 
warnings.simplefilter('ignore')   
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
import logging
logging.basicConfig(level=logging.ERROR) 


#run
# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

#run
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


#run 
# these are 56 unique functions currently in this dataset
unique_functions = ['pre sales','banking','legal','iot','branding','controller','testing','engineering','training','admin','security','data engineering','cyber security','automation','operations','marketing','infrastructure','digital','learning','tax','production','digital marketing','manufacturing','hr','purchase','devops','product management','applications','product security','solutions','inside sales','research','hiring','accounts','risk','artificial intelligence','constomer service','support','compliance','media','accounts recievable','data','blockchain','payroll','sales','network security','accounts payable','analytics','cloud','fraud','corporate finance','distribution','social media','it','account management','finance']


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)


#run
class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.max_len = max_len
         
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        text = text

        return {
            'text': text,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
    
#run
import torch
import torch.nn.functional as F
from transformers import DistilBertModel
from torch.nn import Linear, Dropout, Tanh, Sigmoid
from huggingface_hub import PyTorchModelHubMixin
  
# Creating the customized distillbert model

class DistilBERTClass(torch.nn.Module,PyTorchModelHubMixin):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
         
        # unFreeze all layers
        for name, param in self.l1.named_parameters():
                     param.requires_grad = True
             
         
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 56)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0] 
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)  
        pooled_output = torch.nn.Tanh()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output) 
        output = torch.sigmoid(output)
        return output

model = DistilBERTClass()
#model = torch.nn.DataParallel(model)
model.to(device)

model1 = DistilBERTClass.from_pretrained('anushkaSingh/distillbert_classifier')


#run
def validation(testing_loader):
    model1.eval()
    fin_outputs=[]   
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            text = data['text']
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)

            outputs = model1(ids, mask, token_type_ids)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist()) 
    return  fin_outputs


#run
#actually this function is for test data not validation (dont get confused with the function name)
def validation(testing_loader):
    model1.eval()
    fin_outputs=[]   
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model1(ids, mask, token_type_ids)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist()) 
    return fin_outputs


#run
def flat_accuracy(preds, labels):
    res = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        res[i] = np.all(preds[i] == labels[i]) 
    return np.sum(res) / labels.shape[0]


def find_functional_label(text):
    data = [[text]]
    test_data = pd.DataFrame(data, columns=['text'])
    testing_set = MultiLabelDataset(test_data, tokenizer, 512)
    test_params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 0
                }
    testing_loader = DataLoader(testing_set, **test_params) 
    outputs= validation(testing_loader) 
    final_outputs = np.array(outputs) >=0.5
    text_outputs = dict()
    for i,output in enumerate(final_outputs[0]):#run
      if output == True:
         text_outputs[unique_functions[i]] = outputs[0][i]
    if len(text_outputs) == 0:
       max_value = max(outputs)
       max_index = outputs.index(max_value)
       text_outputs[unique_functions[max_index]] = outputs[0][max_index]   
    return text_outputs
    


    




    


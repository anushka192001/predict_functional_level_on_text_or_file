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
from sklearn import metrics
logging.basicConfig(level=logging.ERROR) 

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def loss_fn(outputs, targets):
    return torch.nn.BCELoss()(outputs, targets)  

unique_functions = ['pre sales','banking','legal','iot','branding','controller','testing','engineering','training','admin','security','data engineering','cyber security','automation','operations','marketing','infrastructure','digital','learning','tax','production','digital marketing','manufacturing','hr','purchase','devops','product management','applications','product security','solutions','inside sales','research','hiring','accounts','risk','artificial intelligence','constomer service','support','compliance','media','accounts recievable','data','blockchain','payroll','sales','network security','accounts payable','analytics','cloud','fraud','corporate finance','distribution','social media','it','account management','finance']

#run
class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.encoded_labels
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
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32 
VALID_BATCH_SIZE = 32
EPOCHS = 10  
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

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

#model = torch.nn.DataParallel(model)

def validation(testing_loader):
    model1.eval()
    fin_text = []
    fin_targets=[]
    fin_outputs=[]   
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            text = data['text']
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model1(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist()) 
            fin_text.extend(text)
    return fin_text, fin_outputs,fin_targets

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

def flat_accuracy(preds, labels):
    res = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        res[i] = np.all(preds[i] == labels[i]) 
    return np.sum(res) / labels.shape[0]  


model1 = DistilBERTClass.from_pretrained('anushkaSingh/distillbert_classifier')


def get_predicted_dataframe_and_hammingscore_and_flat_score(data):
    data = data[['text','label']]
    label_str = data['label'].apply(lambda x: isinstance(x, str))
    def string_to_list(string): 
      return [s.strip() for s in string.split(',')]

    if label_str.all():
          data['label'] = data['label'].apply(string_to_list)

    text_not_str = data['text'].apply(lambda x: not isinstance(x, str))
    label_not_list = data['label'].apply(lambda x: not isinstance(x, list))
    if text_not_str.any() or label_not_list.any():
      return 'Some rows do not match the specified types.' , 'Some rows do not match the specified types.'
    
    final_labels = []
    for row in  data['label']:
     curr_labels = []
     for func in unique_functions:
          if func in row:
               curr_labels.append(1)
          else:
              curr_labels.append(0)
     final_labels.append(curr_labels)
     
    data['Encoded_Final_Functional_level'] = final_labels   
    data = data.rename(columns={'text': 'text', 'Encoded_Final_Functional_level': 'encoded_labels'}) 
    new_df = data  

    test_data=new_df.reset_index(drop=True) 
    testing_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN)
    test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    testing_loader = DataLoader(testing_set, **test_params) 
    #model1 = DistilBERTClass.from_pretrained('anushkaSingh/distillbert_classifier')
    text, outputs,targets = validation(testing_loader)  
    final_outputs = np.array(outputs) >=0.5
    val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))
    flat_score = flat_accuracy(np.array(targets), np.array(final_outputs))

    new_targets = [] 
    for target in targets:
      this_target = []
      for i,label in enumerate(list(target)):
        if int(label)==1:
          this_target.append(unique_functions[i])
      new_targets.append(this_target)

    new_predicted_targets = []
    for j,prediction in enumerate(final_outputs):
        this_prediction = []
        for i,label in enumerate(list(prediction)):
         if int(label)==1:
          this_prediction.append(unique_functions[i])
        if len(this_prediction) == 0:
          max_index = outputs[j].index(max(outputs[j]))
          this_prediction.append(unique_functions[max_index])
        new_predicted_targets.append(this_prediction)   
  
    data = {
    'concatenated_sentence': text,
    'predicted_label': new_predicted_targets,
    'actual_labels' : new_targets
     } 

    for label in unique_functions:
       data[label] = [] 

    for output in outputs:   
      for i,label in enumerate(list(output)):
       data[list(unique_functions)[i]].append(label*100)  
    df = pd.DataFrame(data) 
    return df, val_hamming_score,flat_score



#run
class MultiLabelDataset_no_targets(Dataset):

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
    

def validation_no_targets(testing_loader):
    model1.eval()
    fin_text = []
    fin_outputs=[]   
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            text = data['text']
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model1(ids, mask, token_type_ids)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist()) 
            fin_text.extend(text)
    return fin_text, fin_outputs



def get_predicted_dataframe(data):
    data = data[['text']]
    text_not_str = data['text'].apply(lambda x: not isinstance(x, str))
    if text_not_str.any():
      return 'Some rows do not match the specified types.' , 'Some rows do not match the specified types.'
    new_df = data  
    test_data=new_df.reset_index(drop=True) 
    testing_set = MultiLabelDataset_no_targets(test_data, tokenizer, MAX_LEN)
    test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    testing_loader = DataLoader(testing_set, **test_params) 
    #model1 = DistilBERTClass.from_pretrained('anushkaSingh/distillbert_classifier')
    text, outputs = validation_no_targets(testing_loader)  
    final_outputs = np.array(outputs) >=0.5


    new_predicted_targets = []
    for j,prediction in enumerate(final_outputs):
        this_prediction = []
        for i,label in enumerate(list(prediction)):
         if int(label)==1:
          this_prediction.append(unique_functions[i])
        if len(this_prediction) == 0:
          max_index = outputs[j].index(max(outputs[j]))
          this_prediction.append(unique_functions[max_index])
        new_predicted_targets.append(this_prediction)   
  
    data = {
    'concatenated_sentence': text,
    'predicted_label': new_predicted_targets,
     } 

    for label in unique_functions:
       data[label] = [] 

    for output in outputs:   
      for i,label in enumerate(list(output)):
       data[list(unique_functions)[i]].append(label*100)  
    df = pd.DataFrame(data) 
    return df


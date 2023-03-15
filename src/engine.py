import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import numpy as np
import json
from tqdm import tqdm
device = 'cpu'

DATA_CSV_PATH = "/home/pd/datasets/yelp_reviews/yelp_reviews_2048.csv"
DATA_PATH = "/home/pd/datasets/yelp_reviews/yelp_reviews.json"
SUMMARY_PATH = '/home/pd/summaries/yelp_summary_13Mar23.txt'
OUTPUT_MODEL_PATH = '/home/pd/models/yelp_sentiment.bin'
OUTPUT_VOCAB_PATH = '/home/pd/models/yelp_sentiment_vocab.bin'

FROM_CSV_CONDENSED = True #otherwise getting from full json file

#read-in data
lowstar_review_limit = 1024
review_limit = np.inf
sample_per_cat = 1024
max_num_words = 50

rev = pd.DataFrame()
if FROM_CSV_CONDENSED:
    path = DATA_PATH
    review_fields_wanted = ['text','lowstar']
    rev = pd.DataFrame(columns=review_fields_wanted)
    with open(path,encoding='utf-8') as d:
        counter = 0
        lowstar_counter = 0
        for line in d:
            L = json.loads(line)
            lowstar = L['stars'] == 1 or (L['stars'] == 2)
            fivestar = L['stars'] == 5
            not1or5 = not(lowstar or fivestar)
            if len(L['text'].split()) > max_num_words or not1or5:
                continue
            if lowstar:
                lowstar_counter += 1
                L['lowstar'] = 1
            else:
                L['lowstar'] = 0
            less_fields = {key: L[key] for key in review_fields_wanted }
            rev.loc[counter] = less_fields
            counter += 1
            if counter == review_limit or lowstar_counter == lowstar_review_limit:
                break

                
    rev = rev.rename(columns = {'text':'_text','lowstar':'_lowstar'})


    rev = rev.groupby('_lowstar').apply(lambda x: x.sample(sample_per_cat)).reset_index(drop=True)
    rev['TARGETS'] = rev['_lowstar']
else:
    rev = pd.read_csv(DATA_CSV_PATH)

print(f'Number of 1 star reviews:{rev._lowstar[rev._lowstar == 1].count()}')
print(f'Number of 5 star reviews:{rev._lowstar[rev._lowstar == 0].count()}')
print(rev._text[rev._lowstar == 1].sample(5))
print(rev._text[rev._lowstar == 0].sample(5))


#def dataset
class DFToTokenized(Dataset):
    def __init__(self,df,tokenizer,max_len):
        self.len = len(df)
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self,index):

        review = ' '.join(self.data['_text'][index].split())
        inp = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        tokens = inp['input_ids']
        mask = inp['attention_mask']

        return {
            'ids': torch.tensor(tokens, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.TARGETS[index], dtype=torch.uint8)
        } 

    def __len__(self):
        return self.len

#init train/test params
MAX_LEN = 64
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 1 #set to 1 for printing of individual wrong predictions
EPOCHS = 10
LEARNING_RATE = 1e-05
AUTO_SCALE_GRAD = False
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

train_frac = 0.8
train_dataset=rev.sample(frac=train_frac,random_state=200)
test_dataset=rev.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

training_set = DFToTokenized(train_dataset, tokenizer, MAX_LEN)
testing_set = DFToTokenized(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

single_params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 0
}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

single_loader = DataLoader(training_set,**single_params)

#def model
class DBertMultiCat(torch.nn.Module):
    def __init__(self):
        super(DBertMultiCat, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


#INIT model, loss, optimizer
model = DBertMultiCat()
for p in model.l1.parameters():
    p.requires_grad = False
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
params_with_grad = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params =  params_with_grad, lr=LEARNING_RATE)
if AUTO_SCALE_GRAD:
    scaler = torch.cuda.amp.GradScaler()

#train loop def
def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0),total=len(training_loader),
        position=0, leave=True):

        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.uint8)

        if AUTO_SCALE_GRAD:
            with torch.cpu.amp.autocast():
                outputs = model(ids, mask)
                loss = loss_function(outputs, targets)
        else:
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")


        optimizer.zero_grad(set_to_none=True)
        if(AUTO_SCALE_GRAD):
            scaler.scale(loss).backward()
            # # When using GPU
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 


#train engine
for epoch in range(1):
    train(epoch)

#detokenize/validation def
def DBDetokenize(a):
    a_orig = [tokenizer.decode(x) for x in a['ids'].squeeze().tolist() if x != 0]
    a_orig = ([x.replace(' ' , '') for x in a_orig])
    return " ".join(a_orig)

def valid(model, testing_loader):
    tr_loss = 0 #added
    nb_tr_steps = 0 #added
    nb_tr_examples = 0 #added
    max_wrong_outputs = 10
    wrong_outputs = 0
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask)#.squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            #print individual wrong responses to file
            if VALID_BATCH_SIZE == 1 and wrong_outputs < max_wrong_outputs: 
                wrong_outputs += 1
                with open(SUMMARY_PATH,'a') as f:
                    f.write(DBDetokenize(data))
                    f.write(f'Should be: {1 if targets.item() else 5}')
                    f.write('\n')
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu


#validation run
acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

model_to_save = model
torch.save(model_to_save, OUTPUT_MODEL_PATH)
tokenizer.save_vocabulary(OUTPUT_VOCAB_PATH)
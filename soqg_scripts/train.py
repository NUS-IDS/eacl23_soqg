import pandas as pd
import os
import numpy as np
import torch
from torch import cuda
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from datetime import datetime
import sys
from tabulate import tabulate
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging


def train(epoch, tokenizer, model, device, loader, optimizer, training_logger):
    logging.info("The number of batches to load / training epoch is " + str(len(loader)))
    model.train()
    total_loss = 0
    
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        total_loss += loss.item()

        if _ % 1000 == 0:
            training_logger.append([str(epoch), str(_), str(loss)])
            logging.info(tabulate(training_logger))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info(f"Epoch Total Train Loss: {total_loss}")
    return total_loss

def validate(epoch, tokenizer, model, device, loader, scheduler, valid_logger):

    logging.info("The number of batches to load / validation epoch is " + str(len(loader)))
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
    
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = outputs[0]
            total_loss += loss.item()

            if _ % 1000 == 0:
                valid_logger.append([str(epoch), str(_), str(loss)])
                logging.info(tabulate(valid_logger))

    logging.info(f"Epoch Total Valid Loss: {total_loss}")
    scheduler.step(total_loss)       
    return total_loss

def predict(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    logging.info("Validation Predictions: \n")
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150,
              do_sample=True,
              top_k = 5,
              top_p = 0.6,
              repetition_penalty=1.2,
              no_repeat_ngram_size = 3
              )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%1000==0:
                logging.info(f'Completed validation step {_}')

            predictions.extend(preds)
            actuals.extend(target)
            
    total_num_pred = len(predictions)        
    for i, prediction in enumerate(predictions):
        if i%int(0.1*total_num_pred) == 0:
            logging.info(f"Pred@{i}:{prediction}") 
    return predictions, actuals

class T5_Dataset(Dataset):

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
         ):
        
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        source_text = " ".join(source_text.split())
        target_text = "[Question] " + " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def start_training(
    train_dataset, val_dataset, test_dataset, source_text, target_text, model_params, output_dir="./T5_train_Outputs/"
):

    best_epoch = -999999
    torch.manual_seed(model_params["SEED"]) 
    np.random.seed(model_params["SEED"]) 
    torch.backends.cudnn.deterministic = True
    logging.info(f"""[Model]: Loading {model_params["MODEL_PATH"]}...\n""")
    tokenizer = T5Tokenizer.from_pretrained(model_params["TOKENIZER_PATH"])
    tokenizer.add_tokens("[Question]")
    logging.info(f"[Question] token added {tokenizer.added_tokens_encoder}")
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL_PATH"])
    model = model.to(device)
    logging.info(f"[Data]: Reading data...\n")
    train_dataset = train_dataset[[source_text, target_text]]
    val_dataset = val_dataset[[source_text, target_text]]
    test_dataset = test_dataset[[source_text, target_text]]
    logging.info(f"TRAIN Dataset: {train_dataset.shape}")
    logging.info(f"VALID Dataset: {val_dataset.shape}\n")
    logging.info(f"TEST Dataset: {test_dataset.shape}\n")
    training_set = T5_Dataset(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = T5_Dataset(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    test_set = T5_Dataset(
        test_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }
    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }
    test_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, threshold=0.01)
    logging.info(f"[Initiating Fine Tuning]...\n")
    lowest_valid_loss = 999999
    
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        try:

            training_logger = [["Epoch: ", "Step: ", "Loss: "]]
            valid_logger = [["Epoch: ", "Step: ", "Loss: "]]

            epoch_train_loss = train(epoch, tokenizer, model, device, training_loader, optimizer, training_logger)
            epoch_valid_loss = validate(epoch, tokenizer, model, device, val_loader, scheduler, valid_logger)

            if epoch_valid_loss < lowest_valid_loss:
                logging.info(f"Lowest Valid Loss reached @ epoch {epoch}")
                lowest_valid_loss = epoch_valid_loss
                predictions, actuals = predict(epoch, tokenizer, model, device, test_loader)
                logging.info(f"[Saving Model @ Epoch {epoch}]...\n")
                best_epoch = epoch
                path = os.path.join(output_dir, "model_files_T5_large_conditional")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)             
                test_df["Generated_Question"] = predictions
                test_df["Actual_Question"] = actuals
                test_df.to_csv(os.path.join(output_dir, "t5_prediction_test_df.csv"))
             
        except Exception as e:
            logging.info(e)

    logging.info(f"Training completed")
    logging.info(f"Best Model @{best_epoch}")

if __name__ == "__main__": 

    device = 'cuda:0' if cuda.is_available() else 'cpu'
    logging.basicConfig(filename = "train_progress.log", level = logging.INFO)
    train_df_chunk_I = pd.read_csv("../data/soqg_dataset/train_chunk_I.csv", index_col=0)
    train_df_chunk_II = pd.read_csv("../data/soqg_dataset/train_chunk_II.csv", index_col=0)
    train_df_chunk_III = pd.read_csv("../data/soqg_dataset/train_chunk_III.csv", index_col=0)
    train_df = pd.concat([train_df_chunk_I, train_df_chunk_II, train_df_chunk_III], axis = 0)
    valid_df = pd.read_csv("../data/soqg_dataset/valid.csv", index_col=0)
    test_df = pd.read_csv("../data/soqg_dataset/test.csv", index_col=0)
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    logging.info(len(train_df))
    logging.info(len(valid_df))
    logging.info(len(test_df))

    model_params = {
        "TOKENIZER_PATH": "t5-large", 
        "MODEL_PATH": "t5-large", 
        "TRAIN_BATCH_SIZE": 4,  
        "VALID_BATCH_SIZE": 4, 
        "TRAIN_EPOCHS": 20,  
        "LEARNING_RATE": 5e-5, 
        "MAX_SOURCE_TEXT_LENGTH": 400, 
        "MAX_TARGET_TEXT_LENGTH": 80,  
        "SEED": 0,  
    }

    try:
        start_training(
            train_dataset=train_df,
            val_dataset=valid_df,
            test_dataset=test_df,
            source_text="input",
            target_text="target",
            model_params=model_params,
            output_dir="./Train_Outputs/",
        )
    except:
        logging.exception("")

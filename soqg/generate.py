from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch import cuda

def generate(labels, context, num_seq = 3, top_p = .6, top_k = 5):
    for label in labels:
        tokenized_input = tokenizer(f"{label}: {context}", return_tensors="pt")
        print(f"LABEL: {label}")
        with torch.no_grad():
            ids = tokenized_input["input_ids"].long().to(device)
            mask = tokenized_input["attention_mask"].long().to(device)
            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              do_sample = True,
              top_k = top_k,
              top_p = top_p,
              repetition_penalty=1.2, 
              no_repeat_ngram_size=3,
              num_return_sequences=num_seq
              )
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)  for g in generated_ids]
        for pred in preds:
            print(pred)
            print("-")
        print("="*100)
        
if __name__ == "__main__":
    
    labels = ['reasons_evidence',
     'alternate_viewpoints_perspectives',
     'assumptions',
     'clarity',
     'implication_consequences']

    device = 'cuda:0' if cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained("./Train_Outputs/model_files_T5_large_conditional/")
    model = T5ForConditionalGeneration.from_pretrained("./Train_Outputs/model_files_T5_large_conditional/")
    model.eval()
    model.to(device)
    
    while True:
        user_input = input()
        if user_input:
            generate(labels, user_input, num_seq = 3, top_p = .6, top_k = 5)
        else:
            break

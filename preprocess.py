import pandas as pd
import torch

def read_data(path_imgs = "/kaggle/input/flickr8k/Images", captions_path = "/kaggle/input/flickr8k/captions.txt"):
    
    captions = []
    img_paths = []
    with open(captions_path,"r") as captions_file:
        file_lines = captions_file.read().split("\n")
        file_lines = file_lines[1:-1]

        i = 0
        captions_i = [] 

        for line in file_lines:

            splitted_line = line.split(",")
            img_path = path_imgs + "/" +splitted_line[0]

            if i == 0:
                captions_i = []

            i += 1

            caption = splitted_line[1]
            captions_i.append(caption)
            if i == 5:
                img_paths.append(img_path)
                captions.append(captions_i)
                i = 0
            

    df = pd.DataFrame({"caption":captions ,"img_path": img_paths})
    
    return df

def convert_df_caption_per_row(df):
    paths = df["img_path"]
    
    df_dict = {"img_path":[],
              "caption":[]}
    
    captions = df["caption"]
    for i, path in enumerate(paths):
        for caption in captions.iloc[i]:
            df_dict["caption"].append(caption)
            df_dict["img_path"].append(path)

    return pd.DataFrame(df_dict)

def find_max_token_len(raw_inputs,  tokenizer):
    max_len = 0
    for img_captions in raw_inputs:
        for caption in img_captions:
            caption_tokenized = tokenizer(caption)["input_ids"]
            len_i = len(caption_tokenized)

            if len_i > max_len:
                max_len = len_i
    return max_len


def create_text_model_input(tokenizer, token_max_len, raw_inputs):
    
    tokenizer_output = {"input_ids":[],
                     #"token_type_ids":[],
                     "attention_mask":[]}
    
    
    for img_caption in raw_inputs:
        img_captions_tokenized = tokenizer(img_caption, padding = 'max_length', return_tensors = 'pt',max_length = token_max_len)
        tokenizer_output["input_ids"].append(img_captions_tokenized["input_ids"])
        #tokenizer_output["token_type_ids"].append(img_captions_tokenized["token_type_ids"])
        tokenizer_output["attention_mask"].append(img_captions_tokenized["attention_mask"])
        
    tokenizer_output["input_ids"] = torch.stack(tokenizer_output["input_ids"])
    #tokenizer_output["token_type_ids"] = torch.stack(tokenizer_output["token_type_ids"])
    tokenizer_output["attention_mask"] = torch.stack(tokenizer_output["attention_mask"])

    return tokenizer_output



'''
df = read_data()

convert_df_caption_per_row(df)

'''




        
    
import pandas as pd
import open_clip
import torch
import json
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
from tqdm import tqdm
from torchvision import datasets, transforms
from eva.vision.data import datasets
from eva.vision.data.transforms.common import ResizeAndCrop

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = '/export/livia/home/vision/Bkarimian/CONCH/checkpoints/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
model = model.to(device)
_ = model.eval()
text_encoder = model.text
vision_encoder = model.visual
entity_name = pd.read_csv('text/entity.csv')
level = 3
filter_semantic_name = []
if level == 2:
    filter_semantic_name = ['Neoplastic Process;']
elif level == 2:
    filter_semantic_name = ['Disease or Syndrome;']
elif level == 3:
    filter_semantic_name = ['Pathologic Function;']
if len(filter_semantic_name) != 0:
  mask = entity_name['semantic_name'].isin(filter_semantic_name)
  indices = entity_name.index[mask]
  entity_name = entity_name.iloc[indices].reset_index(drop=True)
texts = entity_name['entity_name'].tolist()
texts = ["a histopathology image of " + t for t in texts]
tokenizer = get_tokenizer()
def get_text_embedding(text):
    with torch.no_grad():
      text = tokenize(texts=[text], tokenizer=tokenizer).to(device)
      text_features = model.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True) 
      return text_features

def get_image_embedding(image):
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_preprocessed)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    return image_embedding, image_preprocessed

def get_pairing(data, embed_tensor):
    value = []
    index = []
    embed_tensor = torch.tensor(embed_tensor).to(device)
    for idx in tqdm(range(len(data))):
        img, label, _ = data.__getitem__(idx)
        image_embedding, image_preprocessed = get_image_embedding(img)
        dot_prod = (image_embedding*embed_tensor).sum(dim = 1)
        v = dot_prod.cpu().max().item()
        idx = dot_prod.cpu().argmax().item()
        value.append(v)
        index.append(idx)
    return value, index

caption = []
embed_tensor = []
print("calculating the embeddings:")
for i in tqdm(range(len(texts))):
    text = texts[i]
    caption.append(text)
    text_embedding = get_text_embedding(text)
    embed_tensor.append(text_embedding.squeeze(0).cpu().numpy().tolist())

mean=(0.48145466, 0.4578275, 0.40821073)
std=(0.26862954, 0.26130258, 0.27577711)    
preprocess = ResizeAndCrop(size = 448, mean = mean, std = std)

from Data.PCAMTDataLoaderConch import PachDataset
import numpy as np
print("training:")
split = "train"
data =  datasets.CRC(
                        root="/export/datasets/public/crc",
                        split=split,
                        download = False,
                        transforms = preprocess,
                    )

data.prepare_data()
data.configure()
value, index = get_pairing(data, embed_tensor)
np.save("train_crc_values_tqx_"+str(level)+".npy", value)
np.save("train_crc_indexes_tqx_"+str(level)+".npy", index)
print("training done")
split = "val"
data =  datasets.CRC(
                        root="/export/datasets/public/crc",
                        split=split,
                        download = False,
                        transforms = preprocess,
                    )

data.prepare_data()
data.configure()
value, index = get_pairing(data, embed_tensor)
np.save("val_crc_values_tqx_"+str(level)+".npy", value)
np.save("val_crc_indexes_tqx_"+str(level)+".npy", index)
print("val done")

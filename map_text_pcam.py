import pandas as pd
import open_clip
import torch
import json
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
from tqdm import tqdm
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = './CONCH/checkpoints/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
model = model.to(device)
_ = model.eval()
text_encoder = model.text
vision_encoder = model.visual
with open("text/text_gpt_pcam.txt") as file:
    texts = [line.rstrip() for line in file]
tokenizer = get_tokenizer()
def get_text_embedding(text):
    with torch.no_grad():
      text = tokenize(texts=[text], tokenizer=tokenizer).to(device)
      text_features = model.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True) 
      return text_features

from torch.utils.data import DataLoader

def get_image_embeddings_batch(image_preprocessed):
    image_preprocessed = image_preprocessed.to(device)
    with torch.no_grad():
        image_embeddings = model.encode_image(image_preprocessed)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    return image_embeddings, image_preprocessed

def get_pairing(data, embed_tensor, batch_size=512):
    embed_tensor = torch.tensor(embed_tensor).to(device)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=5)
    
    value = []
    index = []

    for images, labels in tqdm(dataloader):
        image_embeddings, _ = get_image_embeddings_batch(images)
        dot_prod = torch.matmul(image_embeddings, embed_tensor.T)  # (B, D) x (D, N) → (B, N)

        v = dot_prod.cpu().max(dim=1).values.tolist()
        idx = dot_prod.cpu().argmax(dim=1).tolist()

        value.extend(v)
        index.extend(idx)

    return value, index

embed_tensor = []
print("calculating the embeddings:")
for i in tqdm(range(len(texts))):
    text = texts[i]
    text_embedding = get_text_embedding(text)
    embed_tensor.append(text_embedding.squeeze(0).cpu().numpy().tolist())
    
from Data.PCAMTDataLoaderConch import PachDataset
import numpy as np
print("training:")
data =  datasets.PCAM(
                root="/export/datasets/public/",
                download=True,
                transform=preprocess,
                split = "train"
            )
value, index = get_pairing(data, embed_tensor)
np.save("./Paired_indexes/text_gpt_pcam_train_v.npy", value)
np.save("./Paired_indexes/text_gpt_pcam_train_indexes.npy", index)
print("training done")
data =  datasets.PCAM(
                root="/export/datasets/public/",
                download=True,
                transform=preprocess,
                split = "test"
            )
value, index = get_pairing(data, embed_tensor)
np.save("./Paired_indexes/text_gpt_pcam_test_v.npy", value)
np.save("./Paired_indexes/text_gpt_pcam_test_indexes.npy", index)
print("testing done")

data =  datasets.PCAM(
                root="/export/datasets/public/",
                download=True,
                transform=preprocess,
                split = "val"
            )
value, index = get_pairing(data, embed_tensor)
np.save("./Paired_indexes/text_gpt_pcam_val_v.npy", value)
np.save("./Paired_indexes/text_gpt_pcam_val_indexes.npy", index)
print("val done")

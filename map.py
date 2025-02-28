import pandas as pd
import open_clip
import torch
import json
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
from tqdm import tqdm
from torchvision import datasets
from eva.vision.data import datasets
from eva.vision.data.transforms.common import ResizeAndCrop
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = './CONCH/checkpoints/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
model = model.to(device)
_ = model.eval()
text_encoder = model.text
vision_encoder = model.visual
with open("text/text_colo.txt") as file:
    texts = [line.rstrip() for line in file]

tokenizer = get_tokenizer()
def get_text_embedding(text):
    with torch.no_grad():
      text = tokenize(texts=[text], tokenizer=tokenizer).to(device)
      text_features = model.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True) 
      return text_features

def get_image_embedding(image):
    image = transforms.ToPILImage()(image).convert("RGB")
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
print(len(embed_tensor))
import numpy as np
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
print("training:")
'''
data =  datasets.CRC(
                    root="/export/datasets/public/crc",
                    split="train",
                    download = False,
                    transforms = None,
                )
data.prepare_data()
data.configure()
value, index = get_pairing(data, embed_tensor)
np.save("crc_train_values.npy", value)
np.save("crc_train_indexes.npy", index)
print("training done")'''
data =  datasets.CRC(
                    root="/export/datasets/public/crc",
                    split="val",
                    download = False,
                    transforms = None,
                )
data.prepare_data()
data.configure()
value, index = get_pairing(data, embed_tensor)
np.save("test_pcam_values_prompt.npy", value)
np.save("test_pcam_indexes_prompt.npy", index)
print("testing done")
'''
data =  datasets.PCAM(
                root=".",
                download=True,
                transform=preprocess,
                split = "val"
            )
value, index = get_pairing(data, embed_tensor)
np.save("val_pcam_values_prompt.npy", value)
np.save("val_pcam_indexes_prompt.npy", index)
print("val done")
'''
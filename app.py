import os
import cv2
import yaml
import warnings
import torch
import numpy as np
import streamlit as st
import albumentations as A
import albumentations.pytorch as AP
from PIL import Image

# Local modules
from tokenizers import Tokenizer
from src.modeling import MoT



# Disable warnings and error messages for parallelism.
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


st.title("Bristol-Myers Squibb – Molecular Translation")
st.markdown("---")
torch.no_grad()

with open("models/mot-large-finetune.yml", "r") as fp:
    cfg = yaml.load(fp, yaml.FullLoader)


tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])


model = MoT(
    image_size=cfg["model"]["image_size"],
    num_channels=1,
    patch_size=cfg["model"]["patch_size"],
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=cfg["model"]["max_seq_len"],
    num_encoder_layers=cfg["model"]["num_encoder_layers"],
    num_decoder_layers=cfg["model"]["num_decoder_layers"],
    hidden_dim=cfg["model"]["hidden_dim"],
    num_attn_heads=cfg["model"]["num_attn_heads"],
    expansion_ratio=cfg["model"]["expansion_ratio"],
    encoder_dropout_rate=0.0,
    decoder_dropout_rate=0.0,
    use_torchscript=True,
)
model.load_state_dict(torch.load(cfg["predict"]["weight_path"]))
model.eval()

image_size = 384
transform = A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=0.5, std=0.5),
                AP.ToTensorV2()])

image = st.file_uploader("Upload an image..")

if image:
    st.subheader("You uploaded:")
    st.image(image)

    button = st.button("Get InChI")
    if button:
        image = Image.open(image)
        image = np.array(image)
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        transformed_image = transformed_image.unsqueeze(0)
        inchis = model.generate(
            transformed_image,
            max_seq_len=cfg["model"]["max_seq_len"],
            tokenizer=tokenizer,
        )

        st.subheader("The InChi for the given chemical compound is:")
        st.write(inchis)


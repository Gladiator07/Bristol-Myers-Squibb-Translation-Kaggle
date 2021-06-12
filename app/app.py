import streamlit as st

st.title("BMS")


# def predict():
import os
import warnings
from typing import Any, Dict
import sys
sys.path.insert(0, '/media/atharva/DATA1/Bristol-Myers-Kaggle/src')
from src.modeling import MoT 
# import pandas as pd
import torch
# import tqdm
import yaml
from tokenizers import Tokenizer
# from torch.utils.data import DataLoader
import cv2
# from src.data import BMSDataset, TestTransform
from src.modeling import MoT
from torchvision import transforms
# Disable warnings and error messages for parallelism.
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image
torch.no_grad()

with open("mot-large-finetune.yml", "r") as fp:
    cfg = yaml.load(fp, yaml.FullLoader)
# def main(cfg: Dict[str, Any]):
tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])
# sample_submission = pd.read_csv(cfg["data"]["label_csv_path"])
# sample_submission["image_dir"] = cfg["data"]["image_dir"]

# Create dataset and dataloader for test images through a sample submission file.
# dataset = BMSDataset(
#     sample_submission,
#     transform=TestTransform(cfg["model"]["image_size"]),
# )
# dataloader = DataLoader(
#     dataset,
#     cfg["predict"]["batch_size"],
#     num_workers=os.cpu_count(),
#     pin_memory=True,
# )

# Create a MoT model with given configurations. Note that the parameters will be
# moved to CUDA memory and converted to half precision if `use_fp16` is specified.
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
transform = transforms.Compose([transforms.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
                transforms.Normalize(mean=0.5, std=0.5),
                transforms.ToTensor()])


# if cfg["environ"]["precision"] == 16:
#     model.half()

# Predict InChI strings from the test images.
# with open(cfg["environ"]["name"] + ".csv", "w") as fp:
#     fp.write("image_id,InChI\n")

#     for image_ids, images, _ in tqdm.tqdm(dataloader):
image = st.file_uploader("Upload an image..")
# if cfg["environ"]["precision"] == 16:
#     images = images.half()
if image:
# Generate the InChI strings and update to the submission file.
    st.image(image)
    image = Image.open(image)
    image = transform(image)
    image = image.unsqueeze(0)
    inchis = model.generate(
        image,
        max_seq_len=cfg["model"]["max_seq_len"],
        tokenizer=tokenizer,
    )
    st.write(inchis)
# for image_id, inchi in zip(image_ids, inchis):
#     fp.write(f'{image_id},"{inchi}"\n')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("config")
#     args = parser.parse_args()

#     with open(args.config, "r") as fp:
#         cfg = yaml.load(fp, yaml.FullLoader)
#     main(cfg)

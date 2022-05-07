from fastapi import FastAPI
import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from fastapi.middleware.cors import CORSMiddleware
import requests
app = FastAPI()
import json

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    )
# prepare models:
device = 'cuda'
dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)
projectId = "28FNT2SlVaxu2dZrgAYolOkhXNX"
projectSecret = "5847d2bf588a435ef262228c0bd2321a"
endpoint = "https://ipfs.infura.io:5001"
# pipeline utils:
realesrgan = get_realesrgan('x2', device=device)
clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=8)
scores = []
@app.options("/nft/{query}")
async def read_i(query : str):
   return {"query": "success"}
@app.get("/nft/{query}")
async def read_item(query : str):
 global scores
 text= query
 seed_everything(42)
 pil_images = []
 for top_k, top_p, images_num in [
    (2048, 0.995, 1),
]:
    _pil_images, _scores = generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, bs=8, top_p=top_p)
    pil_images += _pil_images
    scores += _scores
    img=pil_images[0]
    files = {
    'file': img
}
    resp1=await requests.post(endpoint + '/api/v0/add', files=files, auth=(projectId, projectSecret))
    output=await json.loads(resp1)
    hasho=resp1["Hash"] 
 return {"query": hasho}


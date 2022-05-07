from fastapi import FastAPI
import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from fastapi.middleware.cors import CORSMiddleware
import requests
import io
from PIL import Image
app = FastAPI()
import json
import requests
import json
url = "https://api.nft.storage/upload"
from requests.structures import CaseInsensitiveDict
headers = CaseInsensitiveDict()
headers["accept"] = "application/json"
headers["Authorization"] = "Bearer WyIweDRhNjRmZTI0MGVkNGIzMDM5OWI0ZTUwOTdlMGNjMzkyNzAxY2MyM2JkNDU2MDZkMzkwZDRjMjE4ZDBlYWIzMTQ0MWQ0MjM0MTA3NjJlYWRkOGNlNzM5MTliNjI4MjdhNGY2OTlhODg1Njg4ZTdkYzc3MTRiMTYyMzJlZDhmYWI2MWIiLCJ7XCJpYXRcIjoxNjUxOTQ4ODc3LFwiZXh0XCI6MTY1MTk1NjA3NyxcImlzc1wiOlwiZGlkOmV0aHI6MHhEQzIwQkJmZjgyNWM4MzM2N2Y5QTg3OTJlNzUwODgxNzkxNTY0OUY3XCIsXCJzdWJcIjpcImlFRkVhbkxJd2lJQlVwU2JSMXFFejBlLWE5OVNqeHZuT0JHMkVZYm4ySXM9XCIsXCJhdWRcIjpcIlpvYmw1QzJHRWVvT1dudXdpb0RURDRBSnd1NlhFTW5WSEttWjZWOFZZLUU9XCIsXCJuYmZcIjoxNjUxOTQ4ODc3LFwidGlkXCI6XCJmYzZmMzFkOS0wNmE0LTRjZDAtYWVhOS1iMzhmM2M3YTk4NmZcIixcImFkZFwiOlwiMHg4OTI0N2JiOGFhNzYwNDU5OWE5YmZiNzBkMDI3M2QwMDk0MWI2M2RhM2RhMDk0M2Y5MDM5MmQ2NjYzNjM3MzY0NTFjOGQ3ZTE0Y2E5MTc5NmY4OWVlZjYwNDVmNDhmM2MxNjI2ZDAwYTNkNDhjNzRmYmVhZTcxOTNjNWYzMmJjMjFiXCJ9Il0="
headers["Content-Type"] = "image/*"

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
 img=pil_images[0].save(f"{query}.png")
 data=f"@{query}.png"
 resp = requests.post(url, headers=headers, data=data)
 a=resp.json()
 i= a["value"]["cid"]
 print(i)
 print(resp.status_code)
 return {"query": i}


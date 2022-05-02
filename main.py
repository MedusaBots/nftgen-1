from fastapi import FastAPI
import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    )
# prepare models:
device = 'cpu'
dalle = get_rudalle_model('Malevich', pretrained=True, fp16=False, device=device)
tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)

# pipeline utils:
realesrgan = get_realesrgan('x2', device=device)
clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=8)
scores = []
@app.options("/nft/{query}")
async def read_i(query : str):
   return {"query": success}
@app.get("/nft/{query}")
async def read_item(query : str):
 text= query
 seed_everything(42)
 pil_images = []
 for top_k, top_p, images_num in [
    (2048, 0.995, 24),
]:
    _pil_images, _scores =await generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, bs=8, top_p=top_p)
    pil_images += _pil_images
    scores += _scores
 return {"query": pil_images[69]}


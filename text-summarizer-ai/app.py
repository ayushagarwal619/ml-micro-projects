from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

try:
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
    print("Model loaded successfully")
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    import traceback
    traceback.print_exc()
    raise

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

class DialogueInput(BaseModel):
    dialogue: str
    length: str = "medium"

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue : str, length: str = "medium") -> str:
    dialogue = clean_data(dialogue) # clean

    # Determine max length based on length preference
    length_map = {"short": 50, "medium": 150, "long": 300}
    target_max_length = length_map.get(length, 150)

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=target_max_length,
        num_beams=4,
        early_stopping=True
    )
    
    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue, dialogue_input.length)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
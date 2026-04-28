import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import os
import base64
import io
import json
import hashlib
from pathlib import Path

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)


def embed_image(image_data):
    # Normalize input to PIL RGB and ensure reasonable size
    if isinstance(image_data, str):
        image = Image.open(image_data)
    else:
        image = image_data

    if not isinstance(image, Image.Image):
        raise ValueError("embed_image expects a PIL.Image.Image or file path")

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    width, height = image.size
    # Skip pathological tiny images that confuse the preprocessor
    if min(width, height) < 16:
        raise ValueError("Image too small for reliable embedding")

    # Upsample very small images to a minimal size to avoid ambiguous shapes
    min_side = min(width, height)
    if min_side < 32:
        scale = 32.0 / float(min_side)
        new_w = max(32, int(round(width * scale)))
        new_h = max(32, int(round(height * scale)))
        image = image.resize((new_w, new_h), Image.BILINEAR)

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text_data):
    inputs = clip_processor(
        text=text_data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    

def process_pdf(pdf_path):
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    doc = fitz.open(pdf_path)
    for i,page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            temp_doc = Document(page_content=text, metadata={"page":i, "type":"text"})
            chunks = splitter.split_documents([temp_doc])

            for chunk in chunks:
                embedding = embed_text(chunk.page_content)
                all_embeddings.append(embedding)
                all_docs.append(chunk)
        
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Normalize mode
                if pil_image.mode not in ("RGB", "L"):
                    pil_image = pil_image.convert("RGB")
                elif pil_image.mode == "L":
                    pil_image = pil_image.convert("RGB")

                width, height = pil_image.size
                # Skip tiny/odd images outright
                if min(width, height) < 16:
                    continue

                # Upsample very small images to avoid ambiguous shapes
                min_side = min(width, height)
                if min_side < 32:
                    scale = 32.0 / float(min_side)
                    new_w = max(32, int(round(width * scale)))
                    new_h = max(32, int(round(height * scale)))
                    pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

                # Store base64 for UI display
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_id = f"page_{i}_img_{img_index}"
                image_data_store[image_id] = img_base64

                # Compute embedding; skip if embedding fails
                try:
                    embedding = embed_image(pil_image)
                except Exception as e:
                    print(f"Error embedding image {img_index} on page {i}: {e}")
                    continue

                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )

                all_docs.append(image_doc)
                all_embeddings.append(embedding)

            except Exception as e:
                print(f"Error processing image {img_index} on page {i}: {e}")
                continue

    return all_docs, all_embeddings, image_data_store


def compute_file_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_image_store(image_data_store, save_dir, pdf_hash):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "pdf_hash": pdf_hash,
        "images": image_data_store,
    }
    out_path = Path(save_dir) / "image_store.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_image_store(save_dir, expected_hash=None):
    in_path = Path(save_dir) / "image_store.json"
    if not in_path.exists():
        return None
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if expected_hash is not None and payload.get("pdf_hash") != expected_hash:
            return None
        return payload.get("images", {})
    except Exception:
        return None


def extract_image_store(pdf_path):
    image_data_store = {}
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_id = f"page_{i}_img_{img_index}"
                image_data_store[image_id] = img_base64
            except Exception:
                continue
    return image_data_store





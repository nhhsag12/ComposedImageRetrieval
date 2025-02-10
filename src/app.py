import streamlit as st
import torch
import clip
import os
import numpy as np
from PIL import Image
from pymilvus import MilvusClient
from combiner import Combiner
from data_utils import targetpad_transform
from dotenv import load_dotenv

load_dotenv()


# Load the CLIP model
@st.cache_resource()
def load_clip_model(model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("RN50x4", device=device, jit=False)
    input_dim = model.visual.input_resolution
    preprocess = targetpad_transform(target_ratio=1.25, dim=input_dim)

    if model_path:
        saved_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(saved_state_dict["CLIP"])

    model.eval()
    return model, preprocess, device


# Load the Combiner model
def load_combiner(model, combiner_path)->Combiner:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = model.visual.output_dim
    combiner = Combiner(clip_feature_dim=feature_dim, projection_dim=2560, hidden_dim=5120).to(device)

    saved_state_dict = torch.load(combiner_path, map_location=device)
    combiner.load_state_dict(saved_state_dict["Combiner"])

    return combiner


# Connect to Milvus
@st.cache_resource()
def connect_milvus():
    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN")
    return MilvusClient(uri=uri, token=token)


# Encode image into features
def encode_image(image, model, preprocess, device):
    with torch.inference_mode():
        image = preprocess(image).unsqueeze(0).to(device)
        features = model.encode_image(image)
    return features.cpu().numpy().tolist()  # Ensure list format


# Encode text into features
def encode_text(text, model, device):
    with torch.inference_mode():
        text_token = clip.tokenize([text]).to(device)
        text_embedding = model.encode_text(text_token)
    return text_embedding.cpu().numpy().tolist()


# Encode image-text query
def encode_query(image, text, model, combiner: Combiner, preprocess, device):
    image_embedding = torch.tensor(encode_image(image, model, preprocess, device)).to(device)
    text_embedding = torch.tensor(encode_text(text, model, device)).to(device)

    with torch.inference_mode():
        combined_embedding = combiner.combine_features(image_embedding, text_embedding)

    return combined_embedding.cpu().numpy().tolist()


# Search in Milvus
def search_milvus(milvus_client, query_vector, collection_name, top_k=9):
    data_dir = os.getenv("DATA_DIR")

    # Perform search
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        output_fields=["image_path"],
        limit=top_k,
        search_params={"metric_type": "COSINE", "params": {}}
    )[0]  # Extract the first search result list

    # Extract image paths
    retrieved_images = [os.path.join(data_dir, "images", hit.get("entity").get("image_path")) for hit in search_results]
    return retrieved_images


# Streamlit UI
st.title("Composed Image Retrieval Demo")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
query_text = st.text_input("Query Text (e.g., 'solid black with no sleeves'):")


clip_model_path = os.getenv("CLIP_MODEL_PATH")
combiner_path = os.getenv("COMBINER_PATH")

# Load models
model, preprocess, device = load_clip_model(clip_model_path)
combiner = load_combiner(model, combiner_path)
milvus_client = connect_milvus()

if st.button("Search") and uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Query Image", use_container_width=True)
    st.sidebar.image(image, caption="Query Image", use_container_width=True)

    query_vector = encode_query(
        image=image,
        text=query_text,
        combiner=combiner,
        preprocess=preprocess,
        model=model,
        device=device
    )[0]

    results = search_milvus(milvus_client, query_vector, "multimodal_rag_demo")  # Replace with actual collection name

    if not results:
        st.warning("No similar images found in the database.")
    else:
        st.write("### Retrieved Results:")
        cols = st.columns(5)

        for i, image_path in enumerate(results):
            with cols[i % 5]:
                st.image(image_path, caption=f"Result {i+1}")

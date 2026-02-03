import streamlit as st
import torch
import os
import sys

# Add current directory to path so we can import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fastpitch import FastPitch2Wave
from utils import get_basic_config
from vocoder import load_hifigan

st.set_page_config(page_title="Arabic TTS", page_icon="🎙️")

st.title("🎙️ Arabic Text-to-Speech")
st.markdown("Generate Arabic speech from text using FastPitch (Multispeaker).")

@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    model_path = 'pretrained/fastpitch_ar_ms.pth'
    vocoder_config_path = 'pretrained/hifigan-asc-v1/config.json'
    vocoder_path = 'pretrained/hifigan-asc-v1/hifigan-asc.pth'
    
    # Auto-download if missing
    files_to_check = [
        {
            "path": model_path,
            "url": "https://drive.google.com/file/d/18IYUSRXvLErVjaDORj_TKzUxs90l61Ja/view?usp=sharing"
        },
        {
            "path": vocoder_path,
            "url": "https://drive.google.com/file/d/1zSYYnJFS-gQox-IeI71hVY-fdPysxuFK/view?usp=sharing"
        },
        {
             "path": "pretrained/diacritizers/shakkelha_rnn_3_big_20.pth",
             "url": "https://drive.google.com/file/d/1CbDjbuBr-798x88vjLGtMPSB2Y1KwD68/view?usp=sharing"
        },
        {
             "path": "pretrained/diacritizers/shakkala_second_model6.pth",
             "url": "https://drive.google.com/file/d/1hgMGqXLTc58Gq_bN7WpuBWscBxX-rXXd/view?usp=sharing"
        }
    ]
    
    import gdown
    import pathlib
    
    root_dir = pathlib.Path(__file__).parent

    # Check if download is needed
    needs_download = False
    for item in files_to_check:
        p = root_dir.joinpath(item['path'])
        if not p.exists():
            needs_download = True
            break
            
    if needs_download:
        with st.spinner("Preparing system... (This may take a minute on first run)"):
            for item in files_to_check:
                p = root_dir.joinpath(item['path'])
                if not p.exists():
                    if not p.parent.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                    # quiet=True suppresses terminal output
                    gdown.download(item['url'], output=p.as_posix(), quiet=True, fuzzy=True)

    # Ensure config.json exists
    import json
    if not os.path.exists(vocoder_config_path):
        config_data = {
            "resblock": "1",
            "num_gpus": 0,
            "batch_size": 16,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999,
            "seed": 1234,
            "upsample_rates": [8,8,2,2],
            "upsample_kernel_sizes": [16,16,4,4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3,7,11],
            "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
            "segment_size": 8192,
            "num_mels": 80,
            "num_freq": 1025,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 8000,
            "fmax_for_loss": None,
            "num_workers": 4,
            "dist_config": {
                "dist_backend": "nccl",
                "dist_url": "tcp://localhost:54321",
                "world_size": 1
            }
        }
        with open(vocoder_config_path, 'w') as f:
            json.dump(config_data, f, indent=4)

    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Download failed.")
        return None

    # Load Model
    # Explicitly using FastPitch2Wave for endto-end inference
    model = FastPitch2Wave(
        model_sd_path=model_path,
        vocoder_sd=vocoder_path,
        vocoder_config=vocoder_config_path,
        arabic_in=True
    )
    
    model = model.to(device)
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Input
text_input = st.text_area("Enter Arabic Text:", value="اَلسَّلامُ عَلَيكُم يَا صَدِيقِي", height=150)
text_input = text_input.strip()

# Options
col1, col2 = st.columns(2)

with col1:
    # Speaker Selection
    speaker_map = {
        "Male 1": 0,
        "Male 2": 1,
        "Female 1": 2,
        "Female 2": 3
    }
    selected_speaker_label = st.selectbox("Select Voice:", list(speaker_map.keys()))
    speaker_id = speaker_map[selected_speaker_label]

with col2:
    speed = st.slider("Speed:", 0.5, 1.5, 1.0, 0.1)

denoise = st.checkbox("Denoise (Cleaner audio, slower)", value=True)
denoise_val = 0.005 if denoise else 0.0

if st.button("Generate Audio"):
    if not model:
        st.error("Model not loaded.")
    elif not text_input:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating..."):
            try:
                # Run Inference
                with torch.no_grad():
                     wave = model.tts(
                        text_input,
                        speed=speed,
                        speaker_id=speaker_id,
                        denoise=denoise_val
                    )
                
                # Play Audio
                # wave is a tensor, we need to convert or play it. 
                # st.audio accepts numpy array or bytes.
                # wave shape is usually [T]
                
                audio_numpy = wave.cpu().numpy()
                st.audio(audio_numpy, sample_rate=22050)
                
                st.success("Audio generated!")
                
            except Exception as e:
                st.error(f"Error during generation: {e}")

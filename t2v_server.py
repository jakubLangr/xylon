# t2v_server_debug.py
import time
import torch
import gc
import os
import psutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from dfloat11 import DFloat11Model
import uvicorn

# Configure data directory and environment
DATA_DIR = Path("/data")
DATA_DIR.mkdir(exist_ok=True)

# Aggressive memory management
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

# Set cache directories to /data
os.environ["HF_HOME"] = str(DATA_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(DATA_DIR / "torch")
os.environ["TRANSFORMERS_CACHE"] = str(DATA_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(DATA_DIR / "datasets")

# Create subdirectories
print("Creating directories in /data...")
(DATA_DIR / "models").mkdir(exist_ok=True)
(DATA_DIR / "outputs").mkdir(exist_ok=True)
(DATA_DIR / "temp").mkdir(exist_ok=True)
print(f"Created directories: {list(DATA_DIR.iterdir())}")

# Change working directory
print(f"Changing to /data directory: {DATA_DIR}")
os.chdir(DATA_DIR)
print(f"Current working directory: {os.getcwd()}")

app = FastAPI(title="Wan2.2 T2V API Debug", version="1.0.0")

pipe = None
model_loaded = False

def print_memory_usage(stage: str):
    """Print current memory usage"""
    ram = psutil.virtual_memory()
    print(f"\n=== MEMORY at {stage} ===")
    print(f"RAM: {ram.used/1024**3:.2f}GB used / {ram.total/1024**3:.2f}GB total ({ram.percent:.1f}%)")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_mem:.2f}GB used / {gpu_total:.2f}GB total")
    print("=" * 40)

def clear_memory():
    """Aggressive memory cleanup"""
    print("Clearing memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Memory cleared")

async def load_models():
    """Load models with extensive debugging"""
    global pipe, model_loaded
    
    if model_loaded:
        print("Models already loaded, skipping...")
        return
    
    try:
        print("\n" + "="*60)
        print("STARTING MODEL LOADING PROCESS")
        print("="*60)
        print_memory_usage("start")
        
        print("\nStep 1: Initial memory cleanup...")
        clear_memory()
        print_memory_usage("after initial cleanup")
        
        print("\nStep 2: About to load AutoencoderKLWan VAE...")
        print("Loading from: Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        print("Subfolder: vae")
        print("Cache dir:", str(DATA_DIR / "huggingface"))
        print_memory_usage("before VAE load")
        
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            subfolder="vae",
            torch_dtype=torch.float16,  # Use float16 instead of float32
            low_cpu_mem_usage=True,
            cache_dir=str(DATA_DIR / "huggingface")
        )
        print("✓ VAE loaded successfully")
        print_memory_usage("after VAE load")
        clear_memory()
        
        print("\nStep 3: About to load WanPipeline...")
        print("Loading from: Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        print("Using loaded VAE")
        print_memory_usage("before pipeline load")
        
        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            vae=vae,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=str(DATA_DIR / "huggingface")
        )
        print("✓ Pipeline loaded successfully")
        print_memory_usage("after pipeline load")
        clear_memory()
        
        print("\nStep 4: Enabling CPU offloading...")
        print("Enabling model_cpu_offload...")
        pipe.enable_model_cpu_offload()
        print("✓ Model CPU offload enabled")
        
        print("Enabling sequential_cpu_offload...")
        pipe.enable_sequential_cpu_offload()
        print("✓ Sequential CPU offload enabled")
        print_memory_usage("after CPU offload setup")
        
        print("\nStep 5: About to load first DFloat11 model...")
        print("Loading from: DFloat11/Wan2.2-T2V-A14B-DF11")
        print_memory_usage("before DFloat11-1")
        
        DFloat11Model.from_pretrained(
            "DFloat11/Wan2.2-T2V-A14B-DF11",
            device="cpu",
            cpu_offload=True,
            bfloat16_model=pipe.transformer,
            cache_dir=str(DATA_DIR / "huggingface")
        )
        print("✓ First DFloat11 model loaded")
        print_memory_usage("after DFloat11-1")
        clear_memory()
        
        print("\nStep 6: About to load second DFloat11 model...")
        print("Loading from: DFloat11/Wan2.2-T2V-A14B-2-DF11")
        print_memory_usage("before DFloat11-2")
        
        DFloat11Model.from_pretrained(
            "DFloat11/Wan2.2-T2V-A14B-2-DF11",
            device="cpu",
            cpu_offload=True,
            bfloat16_model=pipe.transformer_2,
            cache_dir=str(DATA_DIR / "huggingface")
        )
        print("✓ Second DFloat11 model loaded")
        print_memory_usage("after DFloat11-2")
        clear_memory()
        
        model_loaded = True
        print("\n" + "="*60)
        print("✓ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60)
        print_memory_usage("final state")
        
    except Exception as e:
        print(f"\n❌ ERROR during model loading: {e}")
        print(f"Error type: {type(e)}")
        print_memory_usage("error state")
        clear_memory()
        raise

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "low quality, blurry"
    width: int = 512
    height: int = 320
    num_frames: int = 25
    guidance_scale: float = 4.0
    guidance_scale_2: float = 3.0
    num_inference_steps: int = 15
    fps: int = 8

@app.on_event("startup")
async def startup_event():
    print("FastAPI startup event triggered...")
    await load_models()

@app.get("/")
async def root():
    print_memory_usage("API root call")
    return {
        "message": "Debug Wan2.2 T2V API", 
        "model_loaded": model_loaded,
        "working_directory": str(os.getcwd()),
        "data_directory": str(DATA_DIR)
    }

@app.post("/generate")
async def generate_video(request: GenerationRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"\n🎬 Starting video generation: {request.prompt}")
        print_memory_usage("before generation")
        clear_memory()
        
        print("Calling pipeline...")
        output = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            guidance_scale=request.guidance_scale,
            guidance_scale_2=request.guidance_scale_2,
            num_inference_steps=request.num_inference_steps,
        ).frames[0]
        
        print("✓ Video frames generated")
        print_memory_usage("after generation")
        
        # Save video
        temp_dir = DATA_DIR / "temp" / f"gen_{int(time.time())}"
        temp_dir.mkdir(exist_ok=True)
        output_path = temp_dir / "output.mp4"
        
        print(f"Saving video to: {output_path}")
        export_to_video(output, str(output_path), fps=request.fps)
        print("✓ Video saved")
        
        clear_memory()
        print_memory_usage("after cleanup")
        
        return FileResponse(str(output_path), media_type="video/mp4")
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        print_memory_usage("generation error")
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n🚀 Starting Wan2.2 T2V Server with Debug Logging")
    print(f"Data directory: {DATA_DIR}")
    print(f"Cache directories configured to use /data")
    print_memory_usage("server startup")
    
    uvicorn.run(
        "t2v_server_debug:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
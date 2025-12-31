import argparse
import base64
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr
from pydantic import BaseModel

from .cli import generate_image, edit_image


# ---------- Helpers ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Server-side defaults and diversity controls -----
# These can be overridden with environment variables.
DEFAULT_NEGATIVE = os.environ.get(
    "QWEN_DEFAULT_NEGATIVE",
    "text, words, letters, fonts, typography, handwriting, captions, subtitles, labels, signage, poster text, UI text, numbers, digits, logo, watermark, signature, brand name, overlay text, speech bubble, callouts, comic, cartoon, illustration, vector art, 3d render, anime, painting",
)

PHOTO_MODIFIERS = [
    "Photorealistic, DSLR 50mm, natural lighting, realistic textures, shallow depth of field",
    "Photorealistic, 85mm portrait look, bokeh background, soft studio lighting",
    "Photorealistic, 35mm documentary style, natural light, subtle film grain",
    "Photorealistic, 24mm wide-angle, interior office lighting, realistic shadows",
]

ENFORCE_PHOTOREALISTIC = os.environ.get("QWEN_ENFORCE_PHOTOREALISTIC", "1") == "1"
STYLE_JITTER = os.environ.get("QWEN_STYLE_JITTER", "1") == "1"

try:
    CFG_BASE = float(os.environ.get("QWEN_CFG_BASE", "5.0"))
except Exception:
    CFG_BASE = 5.0

try:
    CFG_JITTER = float(os.environ.get("QWEN_CFG_JITTER", "0.75"))
except Exception:
    CFG_JITTER = 0.75


def apply_server_side_defaults(
    prompt: str,
    negative_prompt: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[str, Optional[str], float, int]:
    """Apply server-side defaults and diversity controls."""
    # Apply photorealistic enforcement
    if ENFORCE_PHOTOREALISTIC:
        photo_modifier = random.choice(PHOTO_MODIFIERS)
        prompt = f"{prompt}, {photo_modifier}"

    # Apply style jitter
    if STYLE_JITTER and cfg_scale is None:
        cfg_scale = max(1.0, CFG_BASE + random.uniform(-CFG_JITTER, CFG_JITTER))

    # Set default negative prompt if none provided
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE

    # Set random seed if none provided
    if seed is None:
        seed = random.randint(0, 2**63 - 1)

    return prompt, negative_prompt, cfg_scale or 4.0, seed


# ---------- Pydantic Models ----------
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    fast: Optional[bool] = False
    ultra_fast: Optional[bool] = False
    seed: Optional[int] = None
    num_images: Optional[int] = 1
    aspect: Optional[str] = "16:9"
    cfg_scale: Optional[float] = None
    lora: Optional[str] = None
    batman: Optional[bool] = False
    outdir: Optional[str] = None
    return_base64: Optional[bool] = False


class EditRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    fast: Optional[bool] = False
    ultra_fast: Optional[bool] = False
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None
    lora: Optional[str] = None
    batman: Optional[bool] = False
    outdir: Optional[str] = None
    return_base64: Optional[bool] = False


# ---------- FastAPI App ----------
app = FastAPI(
    title="Qwen Image MPS API",
    description="REST API for Qwen-Image generation and editing on Apple Silicon (MPS)",
    version="0.5.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/files", StaticFiles(directory=DEFAULT_OUTPUT_DIR), name="files")


# ---------- API Endpoints ----------
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


@app.get("/api/version")
async def get_version():
    """Returns package version."""
    try:
        from . import __version__
        return {"version": __version__}
    except ImportError:
        return {"version": "0.5.0"}


@app.post("/api/generate")
async def generate_image_api(request: GenerateRequest):
    """Generate images from text prompt."""
    try:
        # Apply server-side defaults
        prompt, negative_prompt, cfg_scale, seed = apply_server_side_defaults(
            request.prompt,
            request.negative_prompt,
            request.cfg_scale,
            request.seed,
        )

        # Create args object for generate_image function
        class Args:
            def __init__(self):
                self.prompt = prompt
                self.negative_prompt = negative_prompt
                self.steps = request.steps or 50
                self.fast = request.fast
                self.ultra_fast = request.ultra_fast
                self.seed = seed
                self.num_images = request.num_images
                self.aspect = request.aspect
                self.cfg_scale = cfg_scale
                self.lora = request.lora
                self.batman = request.batman
                self.output_dir = request.outdir or str(DEFAULT_OUTPUT_DIR)

        args = Args()

        # Generate images
        paths = []
        base64_images = []

        for step_info in generate_image(args):
            # The generator yields step information, we need to collect the final paths
            if hasattr(step_info, 'saved_path'):
                paths.append(step_info.saved_path)
            elif isinstance(step_info, str) and os.path.exists(step_info):
                paths.append(step_info)

        # Convert to base64 if requested
        if request.return_base64:
            for path in paths:
                with open(path, "rb") as f:
                    base64_images.append(base64.b64encode(f.read()).decode())

        # Create URLs for files
        urls = []
        for path in paths:
            rel_path = Path(path).relative_to(PROJECT_ROOT)
            urls.append(f"/files/{rel_path}")

        return {
            "paths": paths,
            "urls": urls,
            "base64": base64_images if request.return_base64 else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/edit")
async def edit_image_api(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    steps: Optional[int] = Form(None),
    fast: Optional[bool] = Form(False),
    ultra_fast: Optional[bool] = Form(False),
    seed: Optional[int] = Form(None),
    cfg_scale: Optional[float] = Form(None),
    lora: Optional[str] = Form(None),
    batman: Optional[bool] = Form(False),
    outdir: Optional[str] = Form(None),
    return_base64: Optional[bool] = Form(False),
):
    """Edit an uploaded image."""
    try:
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            shutil.copyfileobj(image.file, tmp_file)
            tmp_path = tmp_file.name

        # Apply server-side defaults
        prompt, negative_prompt, cfg_scale, seed = apply_server_side_defaults(
            prompt,
            negative_prompt,
            cfg_scale,
            seed,
        )

        # Create args object for edit_image function
        class Args:
            def __init__(self):
                self.input = tmp_path
                self.prompt = prompt
                self.negative_prompt = negative_prompt
                self.steps = steps or 50
                self.fast = fast
                self.ultra_fast = ultra_fast
                self.seed = seed
                self.cfg_scale = cfg_scale
                self.lora = lora
                self.batman = batman
                self.output_dir = outdir or str(DEFAULT_OUTPUT_DIR)
                self.output = None

        args = Args()

        # Edit image
        saved_path = edit_image(args)

        # Clean up temporary file
        os.unlink(tmp_path)

        # Convert to base64 if requested
        base64_image = None
        if return_base64 and saved_path:
            with open(saved_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode()

        # Create URL for file
        url = None
        if saved_path:
            rel_path = Path(saved_path).relative_to(PROJECT_ROOT)
            url = f"/files/{rel_path}"

        return {
            "path": saved_path,
            "url": url,
            "base64": base64_image,
        }

    except Exception as e:
        # Clean up temporary file on error
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download")
async def download_file(path: str):
    """Download a file (paths limited to the repo directory)."""
    try:
        # Security check: ensure path is within the project directory
        file_path = Path(path).resolve()
        if not str(file_path).startswith(str(PROJECT_ROOT.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Gradio Interface ----------
def create_gradio_interface():
    """Create the Gradio web interface."""

    def generate_wrapper(
        prompt: str,
        negative_prompt: str,
        steps: int,
        fast: bool,
        ultra_fast: bool,
        seed: int,
        num_images: int,
        aspect: str,
        cfg_scale: float,
        batman: bool,
        progress=gr.Progress(),
    ):
        """Wrapper for generate_image function to use with Gradio."""
        try:
            # Create args object
            class Args:
                def __init__(self):
                    self.prompt = prompt
                    self.negative_prompt = negative_prompt if negative_prompt else None
                    self.steps = steps
                    self.fast = fast
                    self.ultra_fast = ultra_fast
                    self.seed = seed if seed > 0 else None
                    self.num_images = num_images
                    self.aspect = aspect
                    self.cfg_scale = cfg_scale
                    self.lora = None
                    self.batman = batman
                    self.output_dir = str(DEFAULT_OUTPUT_DIR)

            args = Args()

            # Collect generated images
            images = []
            for step_info in generate_image(args):
                # Update progress based on step
                if hasattr(step_info, 'step'):
                    progress(step_info.step / steps)
                
                # Check if this is a final image path
                if hasattr(step_info, 'saved_path') and os.path.exists(step_info.saved_path):
                    images.append(step_info.saved_path)
                elif isinstance(step_info, str) and os.path.exists(step_info):
                    images.append(step_info)

            return images

        except Exception as e:
            return [f"Error: {str(e)}"]

    def edit_wrapper(
        image: gr.Image,
        prompt: str,
        negative_prompt: str,
        steps: int,
        fast: bool,
        ultra_fast: bool,
        seed: int,
        cfg_scale: float,
        batman: bool,
        progress=gr.Progress(),
    ):
        """Wrapper for edit_image function to use with Gradio."""
        try:
            if image is None:
                return "Please upload an image to edit."

            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name

            # Create args object
            class Args:
                def __init__(self):
                    self.input = tmp_path
                    self.prompt = prompt
                    self.negative_prompt = negative_prompt if negative_prompt else None
                    self.steps = steps
                    self.fast = fast
                    self.ultra_fast = ultra_fast
                    self.seed = seed if seed > 0 else None
                    self.cfg_scale = cfg_scale
                    self.lora = None
                    self.batman = batman
                    self.output_dir = str(DEFAULT_OUTPUT_DIR)
                    self.output = None

            args = Args()

            # Edit image
            progress(0.5)
            saved_path = edit_image(args)
            progress(1.0)

            # Clean up temporary file
            os.unlink(tmp_path)

            return saved_path

        except Exception as e:
            return f"Error: {str(e)}"

    # Create Gradio interface
    with gr.Blocks(title="Qwen Image MPS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Qwen Image MPS")
        gr.Markdown("Generate and edit images with Qwen models on Apple Silicon (MPS)")

        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3,
                        )
                        negative_prompt_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid in the image...",
                            lines=2,
                        )
                        with gr.Row():
                            steps_input = gr.Slider(
                                minimum=1, maximum=100, value=50, step=1, label="Steps"
                            )
                            cfg_scale_input = gr.Slider(
                                minimum=1.0, maximum=20.0, value=4.0, step=0.1, label="CFG Scale"
                            )
                        with gr.Row():
                            fast_checkbox = gr.Checkbox(label="Fast Mode (8 steps)")
                            ultra_fast_checkbox = gr.Checkbox(label="Ultra Fast Mode (4 steps)")
                            batman_checkbox = gr.Checkbox(label="Batman Mode 🦇")
                        with gr.Row():
                            seed_input = gr.Number(
                                label="Seed (0 for random)", value=0, precision=0
                            )
                            num_images_input = gr.Slider(
                                minimum=1, maximum=4, value=1, step=1, label="Number of Images"
                            )
                        aspect_input = gr.Dropdown(
                            choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                            value="16:9",
                            label="Aspect Ratio",
                        )
                        generate_btn = gr.Button("Generate", variant="primary")

                    with gr.Column():
                        output_gallery = gr.Gallery(
                            label="Generated Images", show_label=True, elem_id="gallery"
                        )

                generate_btn.click(
                    fn=generate_wrapper,
                    inputs=[
                        prompt_input,
                        negative_prompt_input,
                        steps_input,
                        fast_checkbox,
                        ultra_fast_checkbox,
                        seed_input,
                        num_images_input,
                        aspect_input,
                        cfg_scale_input,
                        batman_checkbox,
                    ],
                    outputs=output_gallery,
                )

            with gr.TabItem("Edit"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Image")
                        prompt_input_edit = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="Describe how to edit the image...",
                            lines=3,
                        )
                        negative_prompt_input_edit = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid in the edit...",
                            lines=2,
                        )
                        with gr.Row():
                            steps_input_edit = gr.Slider(
                                minimum=1, maximum=100, value=50, step=1, label="Steps"
                            )
                            cfg_scale_input_edit = gr.Slider(
                                minimum=1.0, maximum=20.0, value=4.0, step=0.1, label="CFG Scale"
                            )
                        with gr.Row():
                            fast_checkbox_edit = gr.Checkbox(label="Fast Mode (8 steps)")
                            ultra_fast_checkbox_edit = gr.Checkbox(label="Ultra Fast Mode (4 steps)")
                            batman_checkbox_edit = gr.Checkbox(label="Batman Mode 🦇")
                        seed_input_edit = gr.Number(
                            label="Seed (0 for random)", value=0, precision=0
                        )
                        edit_btn = gr.Button("Edit Image", variant="primary")

                    with gr.Column():
                        output_image = gr.Image(label="Edited Image")

                edit_btn.click(
                    fn=edit_wrapper,
                    inputs=[
                        image_input,
                        prompt_input_edit,
                        negative_prompt_input_edit,
                        steps_input_edit,
                        fast_checkbox_edit,
                        ultra_fast_checkbox_edit,
                        seed_input_edit,
                        cfg_scale_input_edit,
                        batman_checkbox_edit,
                    ],
                    outputs=output_image,
                )

    return demo


# ---------- Main Function ----------
def main():
    """Main function to run the web server."""
    parser = argparse.ArgumentParser(description="Qwen Image MPS Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--api-only", action="store_true", help="Run API only (no Gradio UI)")
    args = parser.parse_args()

    # Global app is already defined at module level
    global app
    
    # Create and mount Gradio interface if not API-only
    if not args.api_only:
        demo = create_gradio_interface()
        app = gr.mount_gradio_app(app, demo, path="/")

    import uvicorn

    print(f"Starting Qwen Image MPS web server on http://{args.host}:{args.port}")
    if not args.api_only:
        print(f"Gradio UI available at http://{args.host}:{args.port}/")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

## 📖 About the Project

**WoodAI Studio** is an automated system for the woodworking and interior design sector. The main challenge of this project was to solve the **"ControlNet Bottleneck"**: image diffusion models fail when interpreting 2D technical floor plans because they read dimension lines (e.g., "156cm") and text as if they were 3D geometry, generating hallucinations (like floating drawers or text carved into the wood).

To solve this, we developed an **Autonomous Agent** that orchestrates a hybrid pipeline: **Classic Computer Vision (OpenCV)** for image cleaning, **Multimodal LLMs (Google Gemini)** for semantic extraction, and **Diffusion Models (Leonardo Kino XL + ControlNet)** for photorealistic rendering.

## ✨ Main Features

* 🤖 **Autonomous AI Pipeline:** A smart loop that evaluates its own generated images.

* 👁️ **Deterministic Image Cleaning:** Uses OpenCV to erase dimensions and texts before generation, preventing ControlNet "hallucinations."

* 📐 **Semantic-Geometric Translation:** LLMs convert exact measurements (e.g., `156x80cm`) into descriptive proportions in English (e.g., `wide horizontal proportion`), shielding the 3D engine's prompt.

* ⚖️ **QA Judge (Quality Assurance):** A rigorous Vision model that rejects the image if it doesn't reach a target photorealism score or if it detects residual text on the wood.

* 📉 **Dynamic Weight Decay:** If the Judge rejects the photorealism, the system automatically reduces the structural strength of the ControlNet (e.g., from `0.50` to `0.35`) and retries on its own.

## 🏗️ System Architecture

The project adopts a modular architecture based on single responsibilities:

1. `preprocess_module.py`: Responsible for applying morphological filters and *inpainting* algorithms (OpenCV) to clean scribbles and texts from the original technical drawing.

2. `vision_module.py`: Uses Gemini OCR to extract structural data and build optimized prompts for the SDXL engine.

3. `render_module.py`: Communicates with the Leonardo.ai API, configuring the ControlNet (Edge) and applying the Kino XL photographic model.

4. `judge_module.py`: Acts as the Art Director, evaluating the fidelity of the generated structure and the photorealism.

5. `app.py`: The Streamlit interface and the heart of the Autonomous Agent, orchestrating the retry loop.

## 🚀 How to Run Locally

### Prerequisites

* Python 3.10 or higher

* API Keys for [Google Gemini](https://aistudio.google.com/) and [Leonardo.ai](https://leonardo.ai/)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/woodai-studio.git
cd woodai-studio
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Configure Environment Variables:
   Create a `.env` file in the root of the project with the following content:

```env
GEMINI_API_KEY=your_gemini_key_here
LEONARDO_API_KEY=your_leonardo_key_here
```

5. Start the application:

```bash
streamlit run app.py
```

## 💻 Code Example: The Autonomous Loop

One of the highlights of this project is the retry and weight decay loop (implemented in `app.py`). If the AI generates a bad image, the system self-corrects:

```python
# Simplified excerpt of the Autonomous Agent flow
controlnet_weight = 0.50
max_attempts = 3

for attempt in range(max_attempts):
    # 1. Generate the image
    render_path = render_carpentry_design(
        clean_img_path, prompt, controlnet_weight
    )
    
    # 2. The Judge evaluates the quality
    evaluation = judge_render(render_path, structural_data)
    
    if evaluation.is_approved:
        display_success(render_path)
        break
    else:
        # 3. Weight decay and automatic prompt correction
        controlnet_weight -= 0.15
        prompt = update_prompt_with_judge_feedback(
            prompt, evaluation.prompt_corrections
        )
        print(f"Attempt failed. Reducing weight to {controlnet_weight}...")
```

## 🔮 Next Steps (Roadmap)

* [x] Decouple geometry from text (OpenCV).

* [x] Implement Quality Judge (Vision QA).

* [ ] Support multiple cameras and angles in the same request.

* [ ] Cloud deployment using Docker and Google Cloud Run.

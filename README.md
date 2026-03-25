# simple-manga-translator

Translate manga pages from Japanese (or any language) to English using AI — just point it at a folder and get clean, readable translations back.

Supports DeepL, GPT-4o, Gemini, DeepSeek, Groq, and several offline models out of the box.

---

## Features

- Translates entire folders of manga pages in one command
- Speech bubble detection using a deep learning model + contour analysis
- Automatic text inpainting (removes original text cleanly)
- Multiple translator backends — free and paid
- GPU accelerated

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator) installed as a sibling folder

### Folder structure

```
D:/
├── simple-manga-translator/   ← this repo
└── manga-image-translator/    ← engine (install separately)
```

---

## Installation

**1. Clone this repo**
```bash
git clone https://github.com/YOUR_USERNAME/simple-manga-translator
cd simple-manga-translator
```

**2. Install manga-image-translator (the engine)**
```bash
cd ..
git clone https://github.com/zyddnys/manga-image-translator
cd manga-image-translator
pip install -e .
pip install -r requirements.txt
```

**3. Install simple-manga-translator dependencies**
```bash
cd ../simple-manga-translator
pip install -r requirements.txt
```

**4. Download the bubble detector model**

Download `detector.onnx` from [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) and place it at:
```
simple-manga-translator/models/detector.onnx
```

**5. Set up API keys (if using DeepL, GPT-4o, or other APIs)**

Create a `.env` file in the project root and add whichever keys you need:
```
DEEPL_AUTH_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
GROQ_API_KEY=
```

---

## Usage

**Translate a folder of manga pages:**
```bash
python run.py -i "path/to/manga/folder" -o "path/to/output"
```

**Translate a single image:**
```bash
python run.py -i page.png -o output/
```

**Switch translator:**
```bash
python run.py -i manga/ -o out/ -t deepl
python run.py -i manga/ -o out/ -t chatgpt
python run.py -i manga/ -o out/ -t gemini
```

**Change target language:**
```bash
python run.py -i manga/ -o out/ --lang DEU
```

**List all available translators:**
```bash
python run.py --list
```

**Run without GPU:**
```bash
python run.py -i manga/ -o out/ --no-gpu
```

---

## Translators

| Name | Type | Notes |
|------|------|-------|
| `deepl` | API | Best quality, 500k chars/month free |
| `chatgpt` | API | GPT-4o, excellent quality |
| `gemini` | API | Free tier available |
| `deepseek` | API | Cheap, good quality |
| `groq` | API | Fast, free tier |
| `sugoi` | Offline | Japanese → English only |
| `m2m100` | Offline | Multilingual |
| `m2m100_big` | Offline | Multilingual, higher quality |
| `nllb` | Offline | Meta, multilingual |
| `nllb_big` | Offline | Meta, multilingual large |

---

## Configuration

Edit `settings.json` to change defaults:

```json
{
  "translator": {
    "translator": "deepl",
    "target_lang": "ENG"
  },
  "detector": {
    "detector": "ensemble",
    "detection_size": 2048
  },
  "render": {
    "renderer": "default",
    "font_size_offset": -2
  },
  "inpainter": {
    "inpainter": "lama_large",
    "inpainting_size": 2048
  }
}
```

---

## Credits

See [CREDITS.md](CREDITS.md) for full attribution.

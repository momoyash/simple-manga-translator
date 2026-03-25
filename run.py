"""
simple-manga-translator
-----------------------
Translate manga pages from any language to English (or other targets).
Supports DeepL, OpenAI, and offline models out of the box.

Usage:
  python run.py -i "D:/manga/series" -o "D:/output/series"
  python run.py -i page.png -o out/ -t deepl
  python run.py --list
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from core.renderer import fix_folder, clip_overflow

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
ENGINE   = ROOT.parent / "manga-image-translator"
SETTINGS = ROOT / "settings.json"

load_dotenv(ROOT / ".env")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Translators ────────────────────────────────────────────────────────────────
TRANSLATORS = {
    "sugoi":      {"offline": True,  "key": None,                "info": "Offline JP->EN (fast)"},
    "m2m100":     {"offline": True,  "key": None,                "info": "Offline multilingual"},
    "m2m100_big": {"offline": True,  "key": None,                "info": "Offline multilingual (better)"},
    "nllb":       {"offline": True,  "key": None,                "info": "Offline multilingual (Meta)"},
    "nllb_big":   {"offline": True,  "key": None,                "info": "Offline multilingual large"},
    "deepl":      {"offline": False, "key": "DEEPL_AUTH_KEY",    "info": "DeepL (500k free/mo)"},
    "chatgpt":    {"offline": False, "key": "OPENAI_API_KEY",    "info": "GPT-4o (best quality)"},
    "gemini":     {"offline": False, "key": "GEMINI_API_KEY",    "info": "Gemini (free tier available)"},
    "deepseek":   {"offline": False, "key": "DEEPSEEK_API_KEY",  "info": "DeepSeek (cheap, good)"},
    "groq":       {"offline": False, "key": "GROQ_API_KEY",      "info": "Groq (fast, free tier)"},
}


def show_translators():
    print("\n  Translators\n")
    print(f"  {'name':<14} {'type':<10} info")
    print(f"  {'-'*14} {'-'*10} {'-'*35}")
    for name, t in TRANSLATORS.items():
        kind   = "offline" if t["offline"] else "api"
        status = ""
        if not t["offline"] and t["key"]:
            status = "  [ready]" if os.getenv(t["key"]) else "  [no key]"
        print(f"  {name:<14} {kind:<10} {t['info']}{status}")
    print()


def count_images(folder: Path) -> int:
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def load_settings() -> dict:
    with open(SETTINGS) as f:
        return json.load(f)


def save_settings(s: dict):
    with open(SETTINGS, "w") as f:
        json.dump(s, f, indent=2)


def _clip_pass(src: Path, dst: Path):
    """Post-process: restore any translated text that leaked outside bubbles."""
    import cv2
    print("  [clip] removing overflow text...")
    if src.is_dir():
        fixed = fix_folder(str(src), str(dst), debug=False)
        print(f"  [clip] done  ({fixed} pages)")
    else:
        orig  = cv2.imread(str(src))
        trans = cv2.imread(str(dst / (src.stem + ".png")))
        if orig is not None and trans is not None:
            if orig.shape[:2] != trans.shape[:2]:
                orig = cv2.resize(orig, (trans.shape[1], trans.shape[0]))
            result = clip_overflow(orig, trans)
            cv2.imwrite(str(dst / (src.stem + ".png")), result)
            print("  [clip] done")


def run(input_path: str, output_path: str, translator: str = None,
        lang: str = None, gpu: bool = True, fmt: str = "png",
        clip: bool = False):

    src = Path(input_path)
    dst = Path(output_path)

    if not src.exists():
        print(f"[smt] not found: {src}"); sys.exit(1)
    if not ENGINE.exists():
        print(f"[smt] engine missing at {ENGINE}"); sys.exit(1)

    cfg = load_settings()

    if translator:
        if translator not in TRANSLATORS:
            print(f"[smt] unknown translator '{translator}' — run --list to see options")
            sys.exit(1)
        cfg["translator"]["translator"] = translator
        save_settings(cfg)

    if lang:
        cfg["translator"]["target_lang"] = lang.upper()
        save_settings(cfg)

    active = cfg["translator"]["translator"]
    target = cfg["translator"].get("target_lang", "ENG")
    t_info = TRANSLATORS.get(active, {})

    if not t_info.get("offline") and t_info.get("key"):
        if not os.getenv(t_info["key"]):
            print(f"[smt] {active} needs {t_info['key']} in .env"); sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    label = f"{count_images(src)} pages" if src.is_dir() else src.name
    print(f"\n  {label}")
    print(f"  translator : {active}")
    print(f"  language   : {target}")
    print(f"  output     : {dst}\n")

    cmd = [
        sys.executable, "-m", "manga_translator",
        "--use-gpu" if gpu else "",
        "local",
        "-i", str(src),
        "-o", str(dst),
        "-f", fmt,
        "--overwrite",
        "--config-file", str(SETTINGS),
    ]
    cmd = [c for c in cmd if c]

    result = subprocess.run(cmd, cwd=str(ENGINE), env=os.environ.copy())
    print()
    if result.returncode == 0:
        print(f"  done -> {dst}")
        if clip:
            _clip_pass(src, dst)
    else:
        print(f"  finished with errors (code {result.returncode})")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="smt",
        description="simple-manga-translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  examples:
    python run.py -i manga/ -o out/
    python run.py -i page.png -o out/ -t deepl
    python run.py -i manga/ -o out/ -t chatgpt --lang ENG
    python run.py --list
        """
    )
    p.add_argument("-i", "--input",      help="Image or folder to translate")
    p.add_argument("-o", "--output",     help="Output folder")
    p.add_argument("-t", "--translator", help="Translator (overrides settings.json)")
    p.add_argument("--lang",             help="Target language code  (default: ENG)")
    p.add_argument("-f", "--format",     default="png", choices=["png", "jpg", "webp"])
    p.add_argument("--no-gpu",           action="store_true")
    p.add_argument("--clip",             action="store_true", help="Clip text overflow outside bubbles (experimental)")
    p.add_argument("--list",             action="store_true", help="Show available translators")

    args = p.parse_args()

    if args.list:
        show_translators(); sys.exit(0)

    if not args.input or not args.output:
        p.print_help(); sys.exit(1)

    run(args.input, args.output,
        translator=args.translator,
        lang=args.lang,
        gpu=not args.no_gpu,
        fmt=args.format,
        clip=args.clip)

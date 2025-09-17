#!/usr/bin/env python3
"""
Script to download the multilingual-e5-small model from HuggingFace.
Reads model configuration from settings.toml via dynaconf.
"""

import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from dynaconf import Dynaconf

# Load settings directly in this script
SCRIPT_DIR = Path(__file__).parent

settings = Dynaconf(
    envvar_prefix="TENABLE",
    settings_files=[SCRIPT_DIR / "settings.toml"],
    environments=True,
    load_dotenv=True,
)


def check_model_exists(model_name: str, local_path: Path) -> bool:
    """
    Check if the model already exists and can be loaded successfully.

    Args:
        model_name: HuggingFace model name
        local_path: Local cache directory path

    Returns:
        True if model exists and loads successfully, False otherwise
    """
    try:
        # Check if cache directory exists and has content
        if not local_path.exists() or not any(local_path.iterdir()):
            return False

        # Try to load tokenizer and model from cache
        print("🔍 Checking existing model cache...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(local_path),
            local_files_only=True
        )

        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(local_path),
            local_files_only=True
        )

        # Test model functionality
        test_input = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**test_input)

        print(f"✅ Model exists and is functional (output shape: {outputs.last_hidden_state.shape})")
        return True

    except Exception as e:
        print(f"⚠️ Model cache exists but failed to load: {e}")
        return False


def download_model(force: bool = False) -> None:
    """Download and cache the model and tokenizer from HuggingFace."""
    # Get data directory from settings
    data_dir = settings.get('data_dir', '../data')
    data_path = SCRIPT_DIR / data_dir

    # Handle both old and new settings format
    model_config = settings.get('MODEL', settings.get('model', {}))
    model_name = model_config.get('name', 'intfloat/multilingual-e5-small')

    # Use local_path from settings or construct default path
    local_path_setting = model_config.get('local_path', 'models/multilingual-e5-small')
    if local_path_setting.startswith('../data/'):
        # If path is relative to data dir, use data_path as base
        local_path = data_path / local_path_setting.lstrip('../data/')
    else:
        # Otherwise, treat as relative to script directory
        local_path = SCRIPT_DIR / local_path_setting

    print(f"📋 Model: {model_name}")
    print(f"📁 Cache path: {local_path}")

    # Check if model already exists (unless force download is requested)
    if not force and check_model_exists(model_name, local_path):
        print("✅ Model already exists and is functional - no download needed!")
        print("💡 Use --force to force re-download if needed")
        return

    if force and local_path.exists():
        print("🔄 Force download requested - proceeding with download...")

    # Create the directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download and cache the tokenizer
        print("⬇️ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(local_path)
        )

        # Download and cache the model
        print("⬇️ Downloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(local_path)
        )

        print(f"✅ Successfully downloaded {model_name}")
        print(f"📁 Model cached at: {local_path}")

        # Test loading to verify everything works
        print("🔍 Testing model loading...")
        test_input = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**test_input)

        print("✅ Model test successful!")
        print(f"📊 Model output shape: {outputs.last_hidden_state.shape}")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download multilingual-e5-small model from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_model.py                # Download only if not cached
  python download_model.py --force        # Force re-download even if cached
        """
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model is already cached"
    )

    args = parser.parse_args()

    print("🤖 Multilingual E5 Small Model Downloader")
    print("=" * 50)

    download_model(force=args.force)


if __name__ == "__main__":
    main()
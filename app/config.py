from dynaconf import Dynaconf
from pathlib import Path

# Get the directory where this config.py file is located
CONFIG_DIR = Path(__file__).parent

settings = Dynaconf(
    envvar_prefix="TENABLE",
    settings_files=[CONFIG_DIR / "settings.toml"],
    environments=True,
    load_dotenv=True,
)
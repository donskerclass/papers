from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "profile"
DATA = ROOT / "data"
LOCAL = ROOT / "local"          # gitignored: raw library export, background corpus
SITE = ROOT / "site"            # build output, deployed to GitHub Pages

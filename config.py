"""Central config / secrets. Populate NEWS_API_KEY via environment variable, not source code."""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"
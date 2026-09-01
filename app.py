"""Root FastAPI entry point used by Vercel's zero-configuration runtime."""
from backend.api import app

__all__ = ["app"]

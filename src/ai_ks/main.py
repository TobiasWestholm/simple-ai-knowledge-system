from fastapi import FastAPI

from ai_ks.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive", "environment": settings.app_env}

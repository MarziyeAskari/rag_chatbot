import uvicorn
from app.main import app
from src.config_loader import get_setting

settings = get_setting()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )

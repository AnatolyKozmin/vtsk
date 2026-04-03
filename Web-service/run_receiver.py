import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print("=" * 50)
    print("RECEIVER - Protected Target Service")
    print("=" * 50)
    print(f"Host: 127.0.0.1:{settings.receiver_port}")
    print("Protection: Nemesida WAF + pfSense simulation")
    print("=" * 50)
    uvicorn.run("app.receiver:app", host="127.0.0.1", port=settings.receiver_port, reload=True, access_log=False)

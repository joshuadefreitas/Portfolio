from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
def health_status():
    return {"status": "ok", "message": "API is running"}
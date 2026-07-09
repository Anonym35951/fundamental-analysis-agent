from fastapi import APIRouter, Depends

from api.core.dependencies import get_current_user
from api.models.user import User
from api.services.data_source_status_service import get_data_source_status

router = APIRouter(prefix="/status", tags=["status"])


@router.get("/data-sources")
def data_sources_status(current_user: User = Depends(get_current_user)):
    return {"sources": get_data_source_status()}

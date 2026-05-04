from fastapi import APIRouter

from ..dto.user.create_user import CreateUserDto
from ..services import user_service

router = APIRouter(prefix='/users')

@router.get('')
async def get_users():
    return await user_service.get_users()

@router.post('')
async def create_user(user: CreateUserDto):
    return await user_service.create_user(user)
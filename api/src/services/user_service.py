from ..repositories import user_repository
from ..dto.user.create_user import CreateUserDto

async def create_user(user: CreateUserDto):
    return await user_repository.create_user(user)

async def get_user_by_email(email: str):
    return await user_repository.get_user_by_email(email)

async def get_users():
    return await user_repository.get_users()
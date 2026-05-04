from ..prisma import prisma
from ..dto.user.create_user import CreateUserDto


async def create_user(user: CreateUserDto):
    return await prisma.user.create(data=user.model_dump())

async def get_user_by_email(email: str):
    return await prisma.user.find_unique(where={"email": email})

async def get_users():
    return await prisma.user.find_many()
from datetime import datetime

from pydantic import BaseModel


class User(BaseModel):
    id: str
    name: str
    email: str
    password: str
    createdAt: datetime
    updatedAt: datetime

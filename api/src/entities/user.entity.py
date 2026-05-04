from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    id: str
    name: str
    email: str
    password: str
    createdAt: datetime
    updatedAt: datetime
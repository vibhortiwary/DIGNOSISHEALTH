from pydantic import BaseModel

class TabularInput(BaseModel):
    data: dict

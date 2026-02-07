from pydantic import BaseModel
from typing import List

class Output(BaseModel):
    DateTime: str
    latitude: int
    longitude: int
    predicted_congestion_level: float

class OutputList(BaseModel):
    outputs: List[Output]
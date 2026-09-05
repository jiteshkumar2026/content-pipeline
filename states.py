#so we are creating a graph 
#create a state first
import os

#1 typed dict (most common)
from typing import TypedDict

class state(TypedDict):
    topic : str
    summary :str
    score: int

#2 pydantic approach
#it is good at data validation and type checking

from pydantic import BaseModel,field_validator

class state(BaseModel):
    topic:str
    summary:str
    score:int

    @field_validator
    def score_positive(cls,v):
        if v<0:
            raise ValueError("score musgt be positive")

#3 python dataclasses
#standard python data class but it is used rarely

from dataclasses import dataclass,field

@dataclass
class state:
    topic:str
    summary:str
    score:int
    messages:list=field(default_factory=list)

    
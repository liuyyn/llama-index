from pydantic import BaseModel, Field
from typing import List


class Attraction(BaseModel):
    """Output containing the acttraction name, location and description"""

    name: str = Field(description="The name of the attraction.")
    location: str = Field(description="The address of the attraction.")
    description: str = Field(description="The description of the attraction.")


class Itinerary(BaseModel):
    """Output of the itinerary given the destination"""

    attractions: List[Attraction] = Field(
        description="List of attractions from the itinerary"
    )

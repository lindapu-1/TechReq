from pydantic import BaseModel, Field
from typing import List

criteria = """
Label the reported customer need as a latent need (latent = True) if either of the following conditions is met:

1. The need represents a significant change to the product design and does not fall into any of the following categories: size, shape, weight, material, safety, durability, aesthetics, ergonomics, cost, setup, or transport.
2. The need reflects an exceptionally innovative and clearly expressed insight regarding the product and/or how it is used.

EXTREMELY important apply this criterion very strictly. If the need does not meet either of these conditions, label it as NOT latent (latent = False).
"""

class LatentNeed(BaseModel):
    need: str = Field(..., description="A brief explanation of the need.")
    chain_of_thought: str = Field(..., description=criteria)
    latent: bool = Field(..., description=criteria)


class LatentNeeds(BaseModel):
    latent_needs: List[LatentNeed] = Field(
        ..., description="List of all needs and if they're latent or not."
    )

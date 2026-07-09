from typing import Literal

from pydantic import BaseModel, EmailStr, Field

# Muss mit den Kategorien im Frontend synchron bleiben:
# frontend/src/components/support/SupportForm.tsx (SUPPORT_CATEGORIES)
SUPPORT_CATEGORIES = (
    "Allgemeine Frage",
    "Technisches Problem",
    "Abrechnung & Abo",
    "Feedback",
    "Sonstiges",
)


class SupportRequest(BaseModel):
    category: Literal[
        "Allgemeine Frage",
        "Technisches Problem",
        "Abrechnung & Abo",
        "Feedback",
        "Sonstiges",
    ]
    email: EmailStr
    message: str = Field(min_length=10, max_length=5000)

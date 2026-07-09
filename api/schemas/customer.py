from datetime import datetime

from pydantic import BaseModel


class CustomerListItem(BaseModel):
    id: int
    email: str
    username: str | None
    first_name: str | None
    last_name: str | None
    plan: str
    billing_status: str
    created_at: datetime
    monthly_request_count: int
    monthly_request_limit: int | None

    class Config:
        from_attributes = True


class CustomerDetail(CustomerListItem):
    age: int | None
    email_verified: bool
    current_period_end: datetime | None
    stripe_customer_id: str | None


class UpdateCustomerPlanRequest(BaseModel):
    new_plan: str


class CustomerNoteCreate(BaseModel):
    note: str


class CustomerNoteResponse(BaseModel):
    id: int
    admin_author_id: int | None
    note: str
    created_at: datetime

    class Config:
        from_attributes = True

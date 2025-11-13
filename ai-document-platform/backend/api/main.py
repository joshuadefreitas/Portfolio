from typing import List, Optional, Dict

from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import func

# Reuse DB session + models from your pipeline
from backend.pipeline.load import SessionLocal, Invoice, Vendor


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="AI Document Intelligence Platform API",
    version="1.0.0",
    description="API v1 for invoices, vendors and basic analytics.",
)


# -----------------------------------------------------------------------------
# DB dependency
# -----------------------------------------------------------------------------

def get_db():
    """Create a new SQLAlchemy session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------------
# Pydantic models (response schemas)
# -----------------------------------------------------------------------------

class VendorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    tax_id: Optional[str] = None
    country_code: Optional[str] = None
    default_currency: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class InvoiceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    invoice_number: str
    vendor_id: Optional[int] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: str
    # monetary fields as strings
    amount_gross: str
    amount_net: Optional[str] = None
    amount_vat: Optional[str] = None
    vat_rate: Optional[str] = None
    status: str
    payment_date: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VendorListOut(BaseModel):
    items: List[VendorOut]
    total: int
    limit: int
    offset: int


class InvoiceListOut(BaseModel):
    items: List[InvoiceOut]
    total: int
    limit: int
    offset: int


class TopVendorOut(BaseModel):
    vendor_id: int
    name: str
    total_spend: str


class AnalyticsSummaryOut(BaseModel):
    total_invoices: int
    total_gross: str
    status_counts: Dict[str, int]
    top_vendors: List[TopVendorOut]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def decimal_to_str(value) -> str:
    """Convert Decimal/None to a string, defaulting to '0'."""
    if value is None:
        return "0"
    return str(value)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/api/v1/health", tags=["health"])
def health():
    return {"status": "ok"}


# ----------------------------- Vendors ---------------------------------------


@app.get("/api/v1/vendors", response_model=VendorListOut, tags=["vendors"])
def list_vendors(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    country_code: Optional[str] = Query(None),
):
    query = db.query(Vendor)

    if country_code:
        query = query.filter(Vendor.country_code == country_code)

    total = query.count()
    vendors = (
        query.order_by(Vendor.name)
        .offset(offset)
        .limit(limit)
        .all()
    )

    return VendorListOut(
        items=vendors,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/vendors/{vendor_id}", response_model=VendorOut, tags=["vendors"])
def get_vendor(vendor_id: int, db: Session = Depends(get_db)):
    vendor = db.query(Vendor).filter(Vendor.id == vendor_id).one_or_none()
    if not vendor:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "VENDOR_NOT_FOUND",
                "message": f"Vendor with id={vendor_id} was not found",
                "details": None,
            },
        )
    return vendor


# ----------------------------- Invoices --------------------------------------


@app.get("/api/v1/invoices", response_model=InvoiceListOut, tags=["invoices"])
def list_invoices(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="paid|unpaid|cancelled"),
    min_amount_gross: Optional[str] = Query(None),
    vendor_id: Optional[int] = Query(None),
    issue_date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    issue_date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    query = db.query(Invoice)

    if status:
        query = query.filter(Invoice.status == status)

    if min_amount_gross is not None:
        query = query.filter(Invoice.amount_gross >= min_amount_gross)

    if vendor_id is not None:
        query = query.filter(Invoice.vendor_id == vendor_id)

    if issue_date_from is not None:
        query = query.filter(Invoice.issue_date >= issue_date_from)

    if issue_date_to is not None:
        query = query.filter(Invoice.issue_date <= issue_date_to)

    total = query.count()
    invoices = (
        query.order_by(Invoice.issue_date.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Pydantic v2 + from_attributes will handle ORM → response_model
    # and convert Decimal to str for the string fields.
    return InvoiceListOut(
        items=invoices,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/invoices/{invoice_id}", response_model=InvoiceOut, tags=["invoices"])
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).one_or_none()
    if not invoice:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "INVOICE_NOT_FOUND",
                "message": f"Invoice with id={invoice_id} was not found",
                "details": None,
            },
        )
    return invoice


# ----------------------------- Analytics -------------------------------------


@app.get(
    "/api/v1/analytics/summary",
    response_model=AnalyticsSummaryOut,
    tags=["analytics"],
)
def analytics_summary(
    db: Session = Depends(get_db),
    issue_date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    issue_date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    vendor_id: Optional[int] = Query(None),
):
    base_query = db.query(Invoice)

    if issue_date_from:
        base_query = base_query.filter(Invoice.issue_date >= issue_date_from)
    if issue_date_to:
        base_query = base_query.filter(Invoice.issue_date <= issue_date_to)
    if vendor_id:
        base_query = base_query.filter(Invoice.vendor_id == vendor_id)

    # Total invoices
    total_invoices = base_query.count()

    # Total gross
    total_gross_val = (
        base_query.with_entities(func.coalesce(func.sum(Invoice.amount_gross), 0))
        .scalar()
    )

    # Status counts
    status_rows = (
        base_query.with_entities(Invoice.status, func.count(Invoice.id))
        .group_by(Invoice.status)
        .all()
    )
    status_counts: Dict[str, int] = {
        status: count for status, count in status_rows
    }

    # Top vendors by spend (respecting filters)
    vendor_query = (
        db.query(
            Vendor.id.label("vendor_id"),
            Vendor.name,
            func.coalesce(func.sum(Invoice.amount_gross), 0).label("total_spend"),
        )
        .join(Invoice, Invoice.vendor_id == Vendor.id)
    )

    if issue_date_from:
        vendor_query = vendor_query.filter(Invoice.issue_date >= issue_date_from)
    if issue_date_to:
        vendor_query = vendor_query.filter(Invoice.issue_date <= issue_date_to)
    if vendor_id:
        vendor_query = vendor_query.filter(Vendor.id == vendor_id)

    vendor_rows = (
        vendor_query.group_by(Vendor.id)
        .order_by(func.sum(Invoice.amount_gross).desc())
        .limit(5)
        .all()
    )

    top_vendors = [
        TopVendorOut(
            vendor_id=row.vendor_id,
            name=row.name,
            total_spend=decimal_to_str(row.total_spend),
        )
        for row in vendor_rows
    ]

    return AnalyticsSummaryOut(
        total_invoices=total_invoices,
        total_gross=decimal_to_str(total_gross_val),
        status_counts=status_counts,
        top_vendors=top_vendors,
    )
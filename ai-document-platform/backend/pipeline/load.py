import os
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    String,
    Numeric,
    Date,
    DateTime,
    Boolean,
    ForeignKey,
    func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# -------------------------------------------------------------------
# Database setup
# -------------------------------------------------------------------

# Default: local SQLite for easy dev.
# For Postgres, set DATABASE_URL env var like:
# export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/aidoc"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_doc.db")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()

# -------------------------------------------------------------------
# ORM Models (aligned with schema.sql, simplified for cross-DB)
# -------------------------------------------------------------------

class Vendor(Base):
    __tablename__ = "vendors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vendor_code = Column(String(50), unique=True, nullable=True)
    name = Column(String, nullable=False)
    tax_id = Column(String(50), nullable=True)
    country_code = Column(String(2), nullable=True)
    default_currency = Column(String(3), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    invoices = relationship("Invoice", back_populates="vendor")


class Invoice(Base):
    __tablename__ = "invoices"

    # Use Integer for SQLite autoincrement compatibility
    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_number = Column(String(100), nullable=False)
    vendor_id = Column(Integer, ForeignKey("vendors.id", onupdate="CASCADE"))
    issue_date = Column(Date, nullable=True)
    due_date = Column(Date, nullable=True)
    currency = Column(String(3), nullable=False)
    amount_gross = Column(Numeric(18, 2), nullable=False)
    amount_net = Column(Numeric(18, 2), nullable=True)
    amount_vat = Column(Numeric(18, 2), nullable=True)
    vat_rate = Column(Numeric(5, 2), nullable=True)
    status = Column(String(30), nullable=False, default="unpaid")
    payment_date = Column(Date, nullable=True)
    category = Column(String, nullable=True)  # can be filled by AI later
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    vendor = relationship("Vendor", back_populates="invoices")


# -------------------------------------------------------------------
# Load CSV into database
# -------------------------------------------------------------------

def get_or_create_vendor(session, name, tax_id, country_code, default_currency="EUR"):
    vendor = (
        session.query(Vendor)
        .filter(
            Vendor.name == name,
            Vendor.tax_id == tax_id,
        )
        .one_or_none()
    )

    if vendor is None:
        vendor = Vendor(
            name=name,
            tax_id=tax_id,
            country_code=country_code,
            default_currency=default_currency,
        )
        session.add(vendor)
        session.flush()  # assign id
    return vendor


def load_invoices_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    with SessionLocal() as session:
        for _, row in df.iterrows():
            vendor = get_or_create_vendor(
                session=session,
                name=row["vendor_name"],
                tax_id=row.get("vendor_tax_id", None),
                country_code=row.get("vendor_country_code", None),
                default_currency=row.get("currency", "EUR"),
            )

            invoice = Invoice(
                invoice_number=row["invoice_number"],
                vendor_id=vendor.id,
                issue_date=pd.to_datetime(row["issue_date"]).date() if not pd.isna(row["issue_date"]) else None,
                due_date=pd.to_datetime(row["due_date"]).date() if not pd.isna(row["due_date"]) else None,
                currency=row["currency"],
                amount_gross=row["amount_gross"],
                amount_net=row.get("amount_net", None),
                amount_vat=row.get("amount_vat", None),
                vat_rate=row.get("vat_rate", None),
                status=row.get("status", "unpaid"),
                payment_date=pd.to_datetime(row["payment_date"]).date()
                if ("payment_date" in row and isinstance(row["payment_date"], str) and row["payment_date"])
                else None,
            )

            session.add(invoice)

        session.commit()


def main():
    # Create tables
    print(f"Using DATABASE_URL={DATABASE_URL}")
    Base.metadata.create_all(bind=engine)

    # Locate CSV
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "sample_invoices.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")

    print(f"Loading invoices from {csv_path} ...")
    load_invoices_from_csv(csv_path)
    print("Done. Invoices and vendors loaded into the database.")


if __name__ == "__main__":
    main()
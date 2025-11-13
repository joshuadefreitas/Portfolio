-- schema.sql
-- Base schema for a tenant in the AI Document Intelligence Platform
-- Designed for PostgreSQL

-- ==========================
-- Vendors
-- ==========================
CREATE TABLE vendors (
    id              SERIAL PRIMARY KEY,
    vendor_code     VARCHAR(50) UNIQUE,
    name            TEXT NOT NULL,
    tax_id          VARCHAR(50),
    country_code    CHAR(2),
    default_currency CHAR(3),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optional lookup table for currencies (can also be static in code)
CREATE TABLE currencies (
    code    CHAR(3) PRIMARY KEY,
    name    TEXT
);

-- Seed a few common currencies (optional)
INSERT INTO currencies (code, name) VALUES
    ('EUR', 'Euro'),
    ('USD', 'US Dollar'),
    ('GBP', 'British Pound')
ON CONFLICT (code) DO NOTHING;

-- ==========================
-- Source documents
-- ==========================
-- Stores the original file + OCR/LLM extraction metadata
CREATE TABLE source_documents (
    id              BIGSERIAL PRIMARY KEY,
    external_id     TEXT,               -- e.g. original system ID or hash
    file_name       TEXT NOT NULL,
    mime_type       TEXT,
    storage_path    TEXT NOT NULL,      -- e.g. S3 path or local file path
    uploaded_by     TEXT,               -- user identifier / email
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ocr_text        TEXT,               -- raw OCR text
    llm_model       TEXT,               -- e.g. "gpt-4.1-mini"
    llm_raw_json    JSONB,              -- full extraction payload
    parsed_success  BOOLEAN NOT NULL DEFAULT FALSE,
    error_message   TEXT
);

-- ==========================
-- Invoices (header)
-- ==========================
CREATE TABLE invoices (
    id              BIGSERIAL PRIMARY KEY,
    invoice_number  VARCHAR(100) NOT NULL,
    vendor_id       INTEGER REFERENCES vendors(id) ON UPDATE CASCADE,
    issue_date      DATE,
    due_date        DATE,
    currency        CHAR(3) NOT NULL REFERENCES currencies(code),
    amount_gross    NUMERIC(18, 2) NOT NULL,
    amount_net      NUMERIC(18, 2),
    amount_vat      NUMERIC(18, 2),
    vat_rate        NUMERIC(5, 2),
    status          VARCHAR(30) NOT NULL DEFAULT 'unpaid',  -- unpaid / paid / cancelled / draft
    payment_date    DATE,
    category        TEXT,               -- AI-assigned category (e.g. "Travel", "IT", "Consulting")
    source_document_id BIGINT REFERENCES source_documents(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optional: derived "is_late" flag (Postgres generated column)
ALTER TABLE invoices
ADD COLUMN is_late BOOLEAN GENERATED ALWAYS AS (
    payment_date IS NOT NULL
    AND due_date IS NOT NULL
    AND payment_date > due_date
) STORED;

-- ==========================
-- Invoice line items (details)
-- ==========================
CREATE TABLE invoice_line_items (
    id              BIGSERIAL PRIMARY KEY,
    invoice_id      BIGINT NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    line_number     INTEGER,
    description     TEXT,
    quantity        NUMERIC(18, 4),
    unit_price      NUMERIC(18, 4),
    amount          NUMERIC(18, 2),
    category        TEXT,           -- optional: AI-assigned line item category
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ==========================
-- Indexes for analytics & querying
-- ==========================

CREATE INDEX idx_invoices_vendor_id
    ON invoices (vendor_id);

CREATE INDEX idx_invoices_invoice_number_vendor
    ON invoices (vendor_id, invoice_number);

CREATE INDEX idx_invoices_issue_date
    ON invoices (issue_date);

CREATE INDEX idx_invoices_due_date
    ON invoices (due_date);

CREATE INDEX idx_invoices_status
    ON invoices (status);

CREATE INDEX idx_invoices_is_late
    ON invoices (is_late);

CREATE INDEX idx_source_documents_uploaded_at
    ON source_documents (uploaded_at);

CREATE INDEX idx_invoice_line_items_invoice_id
    ON invoice_line_items (invoice_id);
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

def generate_sample_invoices(n_invoices: int = 80):
    vendors = [
        {"vendor_name": "ACME Consulting", "vendor_tax_id": "NL123456789B01", "vendor_country_code": "NL"},
        {"vendor_name": "Iberia Tech Solutions", "vendor_tax_id": "ESB12345678", "vendor_country_code": "ES"},
        {"vendor_name": "Nordic Cloud AB", "vendor_tax_id": "SE5566778899", "vendor_country_code": "SE"},
        {"vendor_name": "Atlantic Travel", "vendor_tax_id": "GB123456789", "vendor_country_code": "GB"},
        {"vendor_name": "Green Office Supplies", "vendor_tax_id": "DE123456789", "vendor_country_code": "DE"},
    ]

    currencies = ["EUR", "EUR", "EUR", "USD"]  # slightly biased to EUR
    statuses = ["unpaid", "paid", "paid", "cancelled"]

    base_date = datetime(2024, 1, 1)

    rows = []
    for i in range(1, n_invoices + 1):
        vendor = random.choice(vendors)
        issue_date = base_date + timedelta(days=random.randint(0, 180))
        due_date = issue_date + timedelta(days=random.choice([14, 30, 45]))
        currency = random.choice(currencies)

        amount_gross = round(random.uniform(100, 5000), 2)
        vat_rate = random.choice([0.0, 9.0, 21.0])
        if vat_rate > 0:
            amount_net = round(amount_gross / (1 + vat_rate / 100), 2)
            amount_vat = round(amount_gross - amount_net, 2)
        else:
            amount_net = amount_gross
            amount_vat = 0.0

        status = random.choice(statuses)

        if status == "paid":
            payment_delay_days = random.randint(-5, 20)
            payment_date = due_date + timedelta(days=payment_delay_days)
        else:
            payment_date = None

        rows.append({
            "invoice_number": f"INV-{2024}-{i:04d}",
            "vendor_name": vendor["vendor_name"],
            "vendor_tax_id": vendor["vendor_tax_id"],
            "vendor_country_code": vendor["vendor_country_code"],
            "issue_date": issue_date.date().isoformat(),
            "due_date": due_date.date().isoformat(),
            "currency": currency,
            "amount_gross": amount_gross,
            "amount_net": amount_net,
            "amount_vat": amount_vat,
            "vat_rate": vat_rate,
            "status": status,
            "payment_date": payment_date.date().isoformat() if payment_date else "",
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_sample_invoices()
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "sample_invoices.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} invoices at {out_path}")
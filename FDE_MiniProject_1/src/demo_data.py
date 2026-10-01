from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .reason_registry import business_category_for

SAMPLES = [
    ("FIT", "TOO_SMALL", "size bahut chota hai"),
    ("FIT", "TOO_TIGHT", "XL mangaya fir bhi chest me tight"),
    ("FIT", "TOO_LOOSE", "waist me bahut loose hai"),
    ("FIT", "TOO_LONG", "sleeves bahut long hai"),
    ("FIT", "TOO_SHORT", "length mere liye too short hai"),
    ("SIZE_INFORMATION", "SIZE_CHART_MISMATCH", "size chart says 40 but actual smaller"),
    ("SIZE_INFORMATION", "SIZE_NOT_AS_EXPECTED", "usual M leti hu par ye M bilkul alag fit hua"),
    ("QUALITY", "FABRIC_QUALITY", "kapda quality bahut cheap lag raha"),
    ("QUALITY", "STITCHING", "stitching nikal rahi hai"),
    ("QUALITY", "MATERIAL_EXPECTATION", "photo mein cotton jaisa tha actual material different lag raha"),
    ("PRODUCT_MISMATCH", "COLOUR_MISMATCH", "colour pic se totally different"),
    ("PRODUCT_MISMATCH", "IMAGE_MISMATCH", "product photo jaisa bilkul nahi hai"),
    ("DAMAGED_OR_DEFECTIVE", "BROKEN", "zip broken hai"),
    ("DAMAGED_OR_DEFECTIVE", "STAINED", "dress pe stain tha"),
    ("DAMAGED_OR_DEFECTIVE", "TORN", "kurti mein hole hai"),
    ("WRONG_ITEM", "WRONG_SIZE_SENT", "M order kiya tha L deliver hua"),
    ("WRONG_ITEM", "WRONG_COLOUR_SENT", "blue order kiya red mila"),
    ("CUSTOMER_PREFERENCE", "DID_NOT_LIKE", "acha nahi laga"),
    ("CUSTOMER_PREFERENCE", "CHANGED_MIND", "ab nahi chahiye changed my mind"),
    ("DELIVERY_OR_PACKAGING", "PACKAGING_DAMAGED", "packaging pura damaged tha"),
    ("DELIVERY_OR_PACKAGING", "LATE_DELIVERY", "bahut late deliver hua event miss ho gaya"),
    ("DELIVERY_OR_PACKAGING", "NOT_DELIVERED", "parcel delivered show ho raha but mujhe mila hi nahi"),
    ("PRODUCT_MISMATCH", "COLOUR_MISMATCH", "fit good hai but colour different hai"),
    ("FIT", "TOO_TIGHT", "size tight hai aur stitching bhi kharab hai"),
    ("UNCERTAIN", "", "return karna hai"),
    ("UNCERTAIN", "", "problem hai"),
    ("UNCERTAIN", "", "not good"),
    ("UNCERTAIN", "", ""),
]


def generate_demo_frame(rows: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    categories = ["Kurti", "Dress", "Top", "Kidswear", "Shirt"]
    skus = ["K102", "D221", "T332", "K450", "S118", "D904", "T110"]
    vendor_by_sku = {
        "K102": "Aarav Textiles",
        "D221": "Meera Fashions",
        "T332": "Urban Loom",
        "K450": "Aarav Textiles",
        "S118": "Northstar Apparel",
        "D904": "Meera Fashions",
        "T110": "Urban Loom",
    }
    output = []
    for index in range(rows):
        primary, sub, comment = rng.choice(SAMPLES)
        structured_reason = business_category_for(primary, sub)
        if structured_reason in {"Quality", "Product Mismatch", "Customer Preference", "Delivery Or Packaging", "Uncertain"}:
            structured_reason = None
        # Keep a meaningful share under Other so the AI workflow is exercised.
        return_reason = structured_reason if structured_reason and rng.random() < 0.7 else "Other"
        sku = rng.choice(skus)
        output.append({
            "return_id": f"R{index + 1:04d}",
            "sku_id": sku,
            "vendor": vendor_by_sku[sku],
            "category": rng.choice(categories),
            "return_reason": return_reason,
            "return_comment": comment,
            "expected_primary": primary,
            "expected_sub": sub,
        })
    return pd.DataFrame(output)


def write_demo_csv(path: Path, rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_frame(rows).to_csv(path, index=False)

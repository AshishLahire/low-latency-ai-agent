"""
Customer & Order DB helpers
===========================
All reads are simple primary-key or indexed lookups — no full scans.
Results are formatted as plain text so they can be dropped directly
into the LLM system prompt without extra serialisation overhead.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.db.client import get_supabase

logger = logging.getLogger(__name__)


async def get_customer_by_email(email: str) -> Optional[dict]:
    db = get_supabase()
    row = (
        db.table("customers")
        .select("id, name, email, phone, tier")
        .eq("email", email)
        .maybe_single()
        .execute()
    )
    return row.data


async def get_customer_by_id(customer_id: str) -> Optional[dict]:
    db = get_supabase()
    row = (
        db.table("customers")
        .select("id, name, email, phone, tier")
        .eq("id", customer_id)
        .maybe_single()
        .execute()
    )
    return row.data


async def get_recent_orders(customer_id: str, limit: int = 3) -> list:
    """Fetch the most recent N orders for a customer."""
    db = get_supabase()
    rows = (
        db.table("orders")
        .select("id, status, total_amount, currency, items, tracking_number, created_at")
        .eq("customer_id", customer_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return rows.data or []


def format_customer_info(customer: dict, orders: list) -> str:
    """
    Render customer + order data as a compact text block
    for injection into the system prompt.
    """
    if not customer:
        return ""

    lines = [
        f"Name: {customer['name']}",
        f"Email: {customer['email']}",
        f"Tier: {customer['tier'].upper()}",
    ]
    if customer.get("phone"):
        lines.append(f"Phone: {customer['phone']}")

    if orders:
        lines.append("\nRecent orders:")
        for o in orders:
            items_summary = ", ".join(
                f"{i.get('name', 'item')} x{i.get('qty', 1)}"
                for i in (o.get("items") or [])
            )
            tracking = f" | Tracking: {o['tracking_number']}" if o.get("tracking_number") else ""
            lines.append(
                f"  • Order {str(o['id'])[:8]}… — {o['status'].upper()} "
                f"₹{o['total_amount']} ({items_summary}){tracking}"
            )
    else:
        lines.append("\nNo recent orders found.")

    return "\n".join(lines)

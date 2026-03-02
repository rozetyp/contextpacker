"""
Stripe webhook handler for credit purchases.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .config import settings
from .auth import CREDITS_PER_PURCHASE


router = APIRouter()


@router.post("/webhooks/stripe", include_in_schema=False)
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events for credit purchases.
    
    Listens for checkout.session.completed events from Stripe Payment Links.
    Adds 1000 credits per successful $9 payment.
    """
    import stripe
    from .db import get_user_by_email, get_user_by_stripe_customer, add_credits, update_user_stripe_customer
    
    if not settings.STRIPE_SECRET_KEY or not settings.STRIPE_WEBHOOK_SECRET:
        return JSONResponse(status_code=500, content={"error": "Stripe not configured"})
    
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    # Get raw body for signature verification
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, settings.STRIPE_WEBHOOK_SECRET)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid payload"})
    except stripe.error.SignatureVerificationError:
        return JSONResponse(status_code=400, content={"error": "Invalid signature"})
    
    # Handle checkout.session.completed (Payment Link success)
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        
        customer_email = session.get("customer_details", {}).get("email")
        stripe_customer_id = session.get("customer")
        payment_id = session.get("payment_intent") or session.get("id")
        
        if not customer_email:
            return JSONResponse(status_code=200, content={"status": "ignored", "reason": "no email"})
        
        # Find user by Stripe customer ID or email
        user = None
        if stripe_customer_id:
            user = await get_user_by_stripe_customer(stripe_customer_id)
        if not user:
            user = await get_user_by_email(customer_email)
        
        if not user:
            print(f"[STRIPE] Payment from unknown user: {customer_email}")
            return JSONResponse(status_code=200, content={"status": "pending", "reason": "user not found, will be credited on signup"})
        
        # Link Stripe customer if not already linked
        if stripe_customer_id and not user.get("stripe_customer_id"):
            await update_user_stripe_customer(user["id"], stripe_customer_id)
        
        # Add credits
        new_balance = await add_credits(
            user_id=user["id"],
            amount=CREDITS_PER_PURCHASE,
            tx_type="purchase",
            stripe_payment_id=payment_id,
            note="Payment via Stripe checkout"
        )
        
        print(f"[STRIPE] Added {CREDITS_PER_PURCHASE} credits to {customer_email}, new balance: {new_balance}")
        return JSONResponse(status_code=200, content={"status": "success", "credits_added": CREDITS_PER_PURCHASE})
    
    return JSONResponse(status_code=200, content={"status": "ignored"})

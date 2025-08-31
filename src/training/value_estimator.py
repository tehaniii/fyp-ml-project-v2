def estimate_value(metadata):
    # Base price by denomination
    base_by_denom = {
        "One Rupee": (0.5, 0.7),
        "Two Rupees": (0.5, 15.0),
        "Five Rupees": (10.0, 12.0),
        "Ten Rupees": (10.0, 12.0),
        "Massa": (30.0, 60.0),                   # For NissankaMalla_CopperMassa
        "One Rixdollar": (50.0, 150.0),          # For George_IV_Rixdollar_1821
        "One Stiver": (0.0, 2.0),                # For One_Stiver_1815_Replica (replica)
        "Token": (5.0, 25.0)                     # For Wekanda_Mills_Token_1881
    }

    denom = metadata.get("Denomination", "")
    condition = metadata.get("Condition", "New")

    # Default price range for unknown denominations
    low, high = base_by_denom.get(denom, (5.0, 20.0))

    # Use higher price for old/historic coins, else lower
    price = high if condition == "Old" else low

    # Format the result
    buy_price = round(price, 2)
    sell_price = round(price * 1.3, 2)  # ~30% markup

    return {"buy": f"${buy_price}", "sell": f"${sell_price}"}

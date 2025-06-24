def get_nutrition_from_db(item_name: str, nutrition_db: dict) -> list:
    """
    Returns nutrient data from the unified USDA nutrition database for a given food item.

    If the item is a direct match, returns its nutrient dictionary.
    If the item is a dish with ingredients, returns a dictionary mapping each available ingredient
    to its nutrient dictionary.

    Args:
        item_name (str): Food name from detection/classification.
        nutrition_db (dict): Merged JSON database.

    Returns:
        dict: Nutrient info, either directly or per-ingredient (if composite).
    """
    key = item_name.lower().strip()

    if key not in nutrition_db:
        return []

    item = nutrition_db[key]

    # Case 1: Direct nutrient info (not a composite)
    if "nutrients" in item:
        return [{"description": item["description"], "nutrients": item["nutrients"]}]

    # Case 2: Composite dish with ingredients
    elif "ingredients" in item:
        return [
            {"description": ing, "nutrients": data["nutrients"]}
            for ing, data in item["ingredients"].items()
            if "nutrients" in data
        ]

    return []

if __name__ == "__main__":
    import json
    with open("utils/local_nutrition_db.json") as f:
        nutrition_db = json.load(f)

    # Direct USDA item
    print(get_nutrition_from_db("apple", nutrition_db))
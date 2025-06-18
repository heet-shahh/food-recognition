import json

# This is a mock database. In a real application, this data would come from
# a proper database, an API, or a more comprehensive data source.
# Keys should be lowercase for easy, case-insensitive matching.
CALORIE_DATA = {
    # Direct OD results
    "apple": {"calories": 95, "serving_size": "1 medium apple (182g)"},
    "banana": {"calories": 105, "serving_size": "1 medium banana (118g)"},
    "orange": {"calories": 62, "serving_size": "1 medium orange (131g)"},
    "pizza": {"calories": 285, "serving_size": "1 large slice (102g)"},
    "sushi": {"calories": 45, "serving_size": "1 piece"},
    "taco": {"calories": 226, "serving_size": "1 medium taco"},
    "sandwich": {"calories": 350, "serving_size": "1 sandwich"},
    "tomato": {"calories": 22, "serving_size": "1 medium tomato (123g)"},
    "bell pepper": {"calories": 24, "serving_size": "1 medium pepper (119g)"},

    # Custom Kuwaiti classification results
    "balaleet": {"calories": 350, "serving_size": "1 serving (200g)"},
    "mallooba(maqluba)": {"calories": 450, "serving_size": "1 serving (350g)"},
    "majboos_dajaj": {"calories": 550, "serving_size": "1 plate (400g)"},
    "jireesh": {"calories": 400, "serving_size": "1 bowl"},
    "murabyan": {"calories": 480, "serving_size": "1 serving"},
    "samosa": {"calories": 110, "serving_size": "1 piece"},
    "warak_enab": {"calories": 35, "serving_size": "1 piece"},
    "hummus": {"calories": 166, "serving_size": "100g serving"},
    "kebab": {"calories": 230, "serving_size": "1 skewer"},
}

def get_calories(item_name: str) -> str:
    """
    Looks up the calorie count for a given food item from the mock database.

    Args:
        item_name: The name of the food item.

    Returns:
        A formatted string describing the calorie count and serving size.
    """
    # Clean up the item name for matching (lowercase, handle spaces/underscores)
    item_key = item_name.lower().strip().replace("_", " ")

    # Find the calorie data
    item_data = CALORIE_DATA.get(item_key)

    if item_data:
        calories = item_data.get("calories", "N/A")
        serving = item_data.get("serving_size", "N/A")
        return f"{calories} kcal (per {serving})"
    
    return "Not found"

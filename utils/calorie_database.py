# food_pipeline/utils/calorie_database.py

# This is a mock database. In a real application, this data would come from
# a proper database, an API, or a more comprehensive data source.
# Keys should be lowercase for easy, case-insensitive matching.
CALORIE_DATA = {
    # Direct OD results
    "apple": 95,
    "banana": 105,
    "orange": 62,
    "pizza": 285,  # Per slice
    "sushi": 45,   # Per piece
    "taco": 226,
    "sandwich": 350,
    "tomato": 22,
    "bell pepper": 24,

    # Custom Kuwaiti classification results
    "balaleet": 350,
    "majboos_dajaj": 550,
    "jireesh": 400,
    "murabyan": 480,
    "harees": 320,
    "samosa": 110,  # Per piece
    "warak_enab": 35, # Per piece
    "hummus": 166, # Per 100g
    "fattoush": 150,
    "tabouleh": 130,
    "labneh": 59,  # Per 100g
    "kebab": 230,
    "luqaimat": 50, # Per piece
    "gers_ogaily": 280,
    
    # Generic categories (as a fallback)
    "food": 250,  # A generic estimate
    "drink": 150,
}

def get_calories(item_name: str) -> str:
    """
    Looks up the calorie count for a given food item.

    Args:
        item_name: The name of the food item.

    Returns:
        A string describing the calorie count or 'Not found'.
    """
    # Clean up the item name for matching (lowercase, handle spaces)
    item_key = item_name.lower().strip().replace("_", " ")

    # Find the calorie data, return a default if not found
    calories = CALORIE_DATA.get(item_key, "Not found")

    if isinstance(calories, int):
        return f"{calories} kcal"
    return calories
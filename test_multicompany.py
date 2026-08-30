
from rag_core import detect_companies

assert detect_companies("Compare Ford, Tesla and BMW revenue in 2022.") == [
    "BMW", "Tesla", "Ford"
]
assert detect_companies("compare these three in 2022") == [
    "BMW", "Tesla", "Ford"
]
assert detect_companies("What was Ford revenue in 2020?") == ["Ford"]

print("Multi-company detection tests passed.")

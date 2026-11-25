"""Test label_sentiment function for bias."""

from utils import label_sentiment

test_reviews = [
    'very bad service',
    'excellent experience', 
    'not good at all',
    'this is okay',
    'worst ever',
    'amazing',
    'terrible quality',
    'average',
    'poor connection',
    'perfect speed',
    'hate this service',
    'love it',
    'dont like',
    'very satisfied',
]

print("Testing label_sentiment function:\n")
for review in test_reviews:
    label = label_sentiment(review)
    print(f"{review:30} -> {label}")

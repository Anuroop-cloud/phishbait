"""
Simple test script for inference.predict().
"""

from inference import predict


def main() -> None:
    test_samples = [
        "Your account will be suspended immediately. Click http://bit.ly/abc",
        "आपका बैंक खाता बंद कर दिया जाएगा तुरंत सत्यापित करें",
        "Hey bro let's meet at 6pm tomorrow",
        "Congratulations! You won 5 lakh rupees. Verify now!",
        "Lunch at 2?",
    ]

    for msg in test_samples:
        print("-" * 60)
        print(f"Message: {msg}")
        result = predict(msg)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    main()

import unittest
from scripts.evaluate_bias import evaluate_bias

class TestBiasMetrics(unittest.TestCase):
    def test_bias_evaluation(self):
        result = evaluate_bias("bert-base-uncased", "data/processed/stereoset_cleaned.csv")
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()

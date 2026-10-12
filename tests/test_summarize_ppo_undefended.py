import unittest

from evaluation.summarize_ppo_undefended_seeds import summarize


class SummarizePPOUndefendedTests(unittest.TestCase):
    def test_multiseed_summary_matches_raw_runs(self):
        result = summarize(bootstrap_samples=100)
        self.assertEqual(result["seeds"], [42, 43, 44])
        self.assertEqual(result["total_paired_cases"], 225)
        self.assertEqual(
            result["ppo_successes"],
            sum(row["ppo_successes"] for row in result["runs"]),
        )
        self.assertEqual(
            result["greedy_insertion_successes"],
            sum(row["greedy_insertion_successes"] for row in result["runs"]),
        )
        for row in result["runs"]:
            self.assertEqual(row["cases"], 75)
            self.assertEqual(
                row["paired_difference_count"],
                row["ppo_successes"] - row["greedy_insertion_successes"],
            )


if __name__ == "__main__":
    unittest.main()

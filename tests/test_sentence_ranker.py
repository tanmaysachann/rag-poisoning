import unittest

from generation.sentence_ranker import FEATURE_NAMES, _contains_alias


class SentenceRankerTests(unittest.TestCase):
    def test_alias_matches_whole_normalized_span(self):
        self.assertTrue(_contains_alias("The capital is Paris, France.", ["Paris France"]))
        self.assertTrue(_contains_alias("The capital is Paris, France.", ["Paris, France"]))
        self.assertFalse(_contains_alias("The capital is Paris, France.", ["Paris Germany"]))
        self.assertFalse(_contains_alias("Parisian art is famous.", ["Paris"]))

    def test_feature_schema_is_versioned_by_name(self):
        self.assertEqual(len(FEATURE_NAMES), 14)


if __name__ == "__main__":
    unittest.main()

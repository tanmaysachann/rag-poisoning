import unittest

from data.build_multisupport_benchmark import collect_groups


class MultisupportBuilderTests(unittest.TestCase):
    def test_keeps_two_selected_answer_passages_and_excludes_prior_sources(self):
        def row(source_id):
            return {
                "query_id": source_id, "query": f"Where is landmark {source_id}?",
                "answers": ["Paris"],
                "passages": {
                    "passage_text": [
                        f"Landmark {source_id} is situated in Paris, near the river and the central district, according to this local guide.",
                        f"The city of Paris is the location of landmark {source_id}, a well known attraction visited by many people every year.",
                    ],
                    "is_selected": [1, 1],
                },
            }

        groups, scanned = collect_groups([row(1), row(2), row(3), row(4)], target=3,
                                         excluded_source_ids={1})
        self.assertEqual(scanned, 4)
        self.assertEqual([group["source_query_id"] for group in groups], [2, 3, 4])
        self.assertTrue(all(len(group["passages"]) == 2 for group in groups))


if __name__ == "__main__":
    unittest.main()

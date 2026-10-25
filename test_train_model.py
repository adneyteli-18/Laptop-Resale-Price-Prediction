import unittest

import numpy as np

from train_model import FEATURES, SPLIT_COLUMN, TARGET_COLUMN, build_pipeline, load_dataset


class TrainingPipelineTests(unittest.TestCase):
    def test_dataset_has_train_and_holdout_rows_without_target_encoding_features(self):
        dataset = load_dataset()

        self.assertEqual(dataset[SPLIT_COLUMN].nunique(), 2)
        self.assertTrue((dataset[TARGET_COLUMN] > 0).all())
        self.assertTrue(set(FEATURES).issubset(dataset.columns))
        self.assertFalse(any("target_enc" in column for column in dataset.columns))
        self.assertNotIn("price_normalized", dataset.columns)
        self.assertNotIn("description", dataset.columns)

    def test_pipeline_fits_and_predicts_from_the_declared_features(self):
        dataset = load_dataset()
        sample = dataset[dataset[SPLIT_COLUMN]].head(80)
        pipeline = build_pipeline()
        pipeline.fit(sample[FEATURES], sample[TARGET_COLUMN])

        predictions = pipeline.predict(sample[FEATURES].head(5))

        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions > 0).all())


if __name__ == "__main__":
    unittest.main()

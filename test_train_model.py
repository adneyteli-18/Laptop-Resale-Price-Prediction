import unittest

import numpy as np

from app import _format_hardware_option, _model_series_options, _options
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
        self.assertNotIn("state", FEATURES)
        self.assertNotIn("condition_degradation_index", FEATURES)
        self.assertEqual(len(FEATURES), 8)

    def test_dashboard_catalog_exposes_all_models_and_graphics(self):
        dataset = load_dataset()
        training_rows = dataset[dataset[SPLIT_COLUMN]]
        model_options = {
            model
            for brand in training_rows["clean_brand"].unique()
            for model in _model_series_options(training_rows, brand)
        }
        graphics_options = _options(training_rows, "clean_gpu_composite")

        self.assertEqual(model_options, set(training_rows["clean_model_series"].unique()))
        self.assertEqual(len(graphics_options), training_rows["clean_gpu_composite"].nunique())
        self.assertIn("RTX4090 16GB", [_format_hardware_option(option) for option in graphics_options])
        self.assertEqual(
            _format_hardware_option("INTEL_CORE_I5_SKU11_TH"),
            "Intel Core i5 SKU11 TH",
        )

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

import unittest

import numpy as np
import pandas as pd

from app import (
    _dedicated_gpu_options,
    _format_hardware_option,
    _model_series_options,
    load_reference_data,
)
from train_model import (
    FEATURES,
    INTEGRATED_GRAPHICS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    build_pipeline,
    load_dataset,
    normalize_graphics,
)


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
        self.assertIn("is_discrete", FEATURES)
        self.assertEqual(len(FEATURES), 9)

    def test_dashboard_catalog_exposes_all_models_and_only_dedicated_graphics(self):
        training_rows = load_reference_data()
        model_options = {
            model
            for brand in training_rows["clean_brand"].unique()
            for model in _model_series_options(training_rows, brand)
        }
        graphics_options = _dedicated_gpu_options(training_rows)
        dedicated_rates = training_rows.groupby("clean_gpu_composite")["is_discrete"].mean()

        self.assertEqual(model_options, set(training_rows["clean_model_series"].unique()))
        self.assertEqual(
            set(graphics_options),
            set(dedicated_rates[dedicated_rates > 0.5].index),
        )
        self.assertNotIn(INTEGRATED_GRAPHICS, graphics_options)
        self.assertFalse(any("IGPU" in option for option in graphics_options))
        self.assertIn("RTX4090 16GB", [_format_hardware_option(option) for option in graphics_options])
        self.assertEqual(
            _format_hardware_option("INTEL_CORE_I5_SKU11_TH"),
            "Intel Core i5 SKU11 TH",
        )

    def test_integrated_graphics_are_normalized_to_no_dedicated_gpu(self):
        sample = pd.DataFrame(
            {
                "is_discrete": [0, 1],
                "clean_gpu_composite": [
                    "INTEL_UHD_SYSRAM",
                    "RTX4090_16GB",
                ],
            }
        )

        normalized = normalize_graphics(sample)

        self.assertEqual(
            normalized["clean_gpu_composite"].tolist(),
            [INTEGRATED_GRAPHICS, "RTX4090_16GB"],
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

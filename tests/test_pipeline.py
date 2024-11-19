import unittest
import sys
import os
# from ..src/pipeline/data_pipeline import DataPipeline
# Add src folder to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pipeline.new_data_pipeline import DataPipeline # type: ignore
# from  src.pipeline.data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DataPipeline()

    def test_generate_sample_data(self):
        df = self.pipeline.generate_sample_data()
        self.assertFalse(df.empty)
        self.assertIn('cpu_usage', df.columns)

    def test_preprocess_data(self):
        df = self.pipeline.generate_sample_data()
        processed = self.pipeline.preprocess_data(df)
        self.assertIn('features', processed)
        self.assertEqual(len(processed['features']), len(df))

if __name__ == "__main__":
    unittest.main()

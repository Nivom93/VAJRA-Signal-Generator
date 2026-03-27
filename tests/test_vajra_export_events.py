import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from vajra_export_events import fetch_delta_oi
from vajra_engine_ultra_v6_final import ExchangeWrapper, EngineConfig

class TestFetchDeltaOI(unittest.TestCase):
    def setUp(self):
        self.mock_exw = MagicMock(spec=ExchangeWrapper)
        self.mock_client = MagicMock()
        self.mock_exw.client = self.mock_client
        self.mock_exw.client.id = "binance"
        self.mock_exw.client.has = {'fetchOpenInterestHistory': True}

        # 10 timestamps, 1 minute apart starting from 1700000000000 ms
        self.ltf_timestamps = pd.Series([1700000000000 + i * 60000 for i in range(10)])

    def test_fetch_delta_oi_happy_path(self):
        # Setup mock data from exchange
        # Note: fetch_delta_oi expects 'timestamp' in ms
        mock_oi_data = [
            {'timestamp': 1700000000000, 'openInterestValue': '100.0'},
            {'timestamp': 1700000060000, 'openInterestValue': '105.0'}, # 5% increase
            {'timestamp': 1700000120000, 'openInterestValue': '102.9'}, # 2% decrease
            {'timestamp': 1700000180000, 'openInterestValue': '102.9'}, # 0% change
            {'timestamp': 1700000240000, 'openInterestValue': '113.19'}, # 10% increase
        ]
        self.mock_client.fetch_open_interest_history.return_value = mock_oi_data

        result = fetch_delta_oi(self.mock_exw, "BTC/USDT", "1m", self.ltf_timestamps)

        # Let's see what the dataframe would actually look like in fetch_delta_oi
        # df['timestamp'] = pd.to_datetime(...)
        # df.set_index('timestamp', inplace=True)
        # aligned = df['delta_oi'].shift(1).reindex(pd.to_datetime(ltf_timestamps, unit='ms', utc=True)).ffill().fillna(0.0)
        #
        # Values:
        # idx 0: 1700000000000 -> 0.0
        # idx 1: 1700000060000 -> 5.0
        # idx 2: 1700000120000 -> -2.0
        # idx 3: 1700000180000 -> 0.0
        # idx 4: 1700000240000 -> 10.0
        #
        # shift(1) values:
        # idx 0: 1700000000000 -> NaN
        # idx 1: 1700000060000 -> 0.0
        # idx 2: 1700000120000 -> 5.0
        # idx 3: 1700000180000 -> -2.0
        # idx 4: 1700000240000 -> 0.0
        #
        # The reindex is on pd.to_datetime(ltf_timestamps, unit='ms', utc=True)
        # ltf_timestamps values are identical to the first 10 minute intervals starting at 1700000000000
        # so:
        # result[0] = NaN -> fillna(0.0) -> 0.0
        # result[1] = 0.0
        # result[2] = 5.0
        # result[3] = -2.0
        # result[4] = 0.0
        # result[5] = 0.0 (because ffill from result[4] which is 0.0) -> Wait, df['delta_oi'] has no index 5!
        # Ah, reindex will forward-fill the last available value.
        # The last value in shift(1) is 0.0 (from idx 4).
        # Let's verify this logic.

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.ltf_timestamps))

        # Check call arguments
        self.mock_client.fetch_open_interest_history.assert_called_once_with("BTC/USDT", "1m", limit=1000)

        # Expected based on the implementation:
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 5.0)
        self.assertAlmostEqual(result[3], -2.0)
        self.assertAlmostEqual(result[4], 0.0)

        # Everything after 1700000240000 in the shifted series is missing, so it will forward-fill
        # the last value which is 0.0.
        for i in range(5, 10):
            self.assertAlmostEqual(result[i], 0.0)

    def test_fetch_delta_oi_no_capability(self):
        # Exchange doesn't support fetchOpenInterestHistory
        self.mock_exw.client.has = {'fetchOpenInterestHistory': False}
        result = fetch_delta_oi(self.mock_exw, "BTC/USDT", "1m", self.ltf_timestamps)

        self.assertEqual(len(result), len(self.ltf_timestamps))
        self.assertTrue(np.all(result == 0.0))
        self.mock_client.fetch_open_interest_history.assert_not_called()

    def test_fetch_delta_oi_empty_response(self):
        self.mock_client.fetch_open_interest_history.return_value = []
        result = fetch_delta_oi(self.mock_exw, "BTC/USDT", "1m", self.ltf_timestamps)

        self.assertEqual(len(result), len(self.ltf_timestamps))
        self.assertTrue(np.all(result == 0.0))

    def test_fetch_delta_oi_missing_keys(self):
        # Response missing openInterestValue
        self.mock_client.fetch_open_interest_history.return_value = [{'timestamp': 1700000000000}]
        result = fetch_delta_oi(self.mock_exw, "BTC/USDT", "1m", self.ltf_timestamps)

        self.assertEqual(len(result), len(self.ltf_timestamps))
        self.assertTrue(np.all(result == 0.0))

    def test_fetch_delta_oi_bybit_symbol_formatting(self):
        self.mock_exw.client.id = "bybit"
        self.mock_client.fetch_open_interest_history.return_value = []

        fetch_delta_oi(self.mock_exw, "BTC/USDT", "1m", self.ltf_timestamps)

        # Should transform BTC/USDT to BTC/USDT:USDT for Bybit
        self.mock_client.fetch_open_interest_history.assert_called_once_with("BTC/USDT:USDT", "1m", limit=1000)

    @patch("vajra_export_events.log")
    def test_fetch_delta_oi_exception(self, mock_log):
        self.mock_client.fetch_open_interest_history.side_effect = Exception("API Error")

        result = fetch_delta_oi(self.mock_exw, "BTC/USDT", "1m", self.ltf_timestamps)

        self.assertEqual(len(result), len(self.ltf_timestamps))
        self.assertTrue(np.all(result == 0.0))
        mock_log.warning.assert_called_once()
        self.assertIn("CCXT fetch_oi failed", mock_log.warning.call_args[0][0])

    def test_fetch_delta_oi_single_value(self):
        # Edge case: only one value returned, can't compute delta
        self.mock_client.fetch_open_interest_history.return_value = [
            {'timestamp': 1700000000000, 'openInterestValue': '100.0'}
        ]

        result = fetch_delta_oi(self.mock_exw, "BTC/USDT", "1m", self.ltf_timestamps)

        self.assertEqual(len(result), len(self.ltf_timestamps))
        self.assertTrue(np.all(result == 0.0))

if __name__ == "__main__":
    unittest.main()

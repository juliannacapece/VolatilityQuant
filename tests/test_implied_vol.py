import unittest
import numpy as np
import pandas as pd
from modules.engine import black76_price, implied_vol, implied_forward, prepare_chain, load_aapl_snapshot


class ImpliedVolTests(unittest.TestCase):
    def test_round_trip_recovers_volatility(self):
        F, T, DF = 100., .5, np.exp(-.04 * .5)
        for kind in ('Call', 'Put'):
            for K in (70., 90., 100., 115., 140.):
                for sigma in (.08, .25, .6, 1.2):
                    price = black76_price(F, K, T, DF, sigma, kind)
                    if price < 1e-6:  # too far OTM to carry volatility information
                        continue
                    self.assertAlmostEqual(implied_vol(price, F, K, T, DF, kind), sigma, places=5)

    def test_prices_outside_no_arbitrage_bounds_give_nan(self):
        F, K, T, DF = 100., 90., .25, .99
        intrinsic = DF * (F - K)
        self.assertTrue(np.isnan(implied_vol(intrinsic * .99, F, K, T, DF, 'Call')))  # below intrinsic
        self.assertTrue(np.isnan(implied_vol(DF * F * 1.01, F, K, T, DF, 'Call')))   # above upper bound
        self.assertTrue(np.isnan(implied_vol(0., F, K, T, DF, 'Put')))
        self.assertTrue(np.isnan(implied_vol(1., F, K, 0., DF, 'Put')))               # expired

    def test_forward_from_put_call_parity(self):
        F, T, DF, sigma = 103.7, .4, np.exp(-.05 * .4), .3
        rows = [{'strike': K, 'optionType': kind, 'mid': black76_price(F, K, T, DF, sigma, kind)}
                for K in np.arange(80., 125., 5.) for kind in ('Call', 'Put')]
        forward, source = implied_forward(pd.DataFrame(rows), spot=100., DF=DF)
        self.assertAlmostEqual(forward, F, places=6)
        self.assertEqual(source, 'put-call parity')

    def test_forward_falls_back_without_pairs(self):
        calls_only = pd.DataFrame({'strike': [95., 100.], 'optionType': 'Call', 'mid': [6., 3.]})
        forward, source = implied_forward(calls_only, spot=100., DF=.98)
        self.assertAlmostEqual(forward, 100. / .98)
        self.assertIn('no dividends', source)

    def test_prepare_chain_recovers_flat_smile(self):
        # Synthetic chain priced at 30% vol with a 2% dividend yield: every
        # computed IV should come back at 30% and calls/puts should agree.
        spot, rate, q, sigma = 100., .04, .02, .30
        as_of = pd.Timestamp('2026-01-02 16:00', tz='America/New_York')
        rows = []
        for exp in ('2026-02-20', '2026-06-19'):
            T = (pd.Timestamp(exp + ' 16:00', tz='America/New_York') - as_of).total_seconds() / (365 * 86400)
            DF, F = np.exp(-rate * T), spot * np.exp((rate - q) * T)
            for K in np.arange(75., 130., 5.):
                for kind in ('Call', 'Put'):
                    mid = black76_price(F, K, T, DF, sigma, kind)
                    rows.append({'strike': K, 'bid': mid * .99, 'ask': mid * 1.01, 'optionType': kind,
                                 'expiration': exp, 'impliedVolatility': .5})
        d = prepare_chain(pd.DataFrame(rows), spot, as_of, rate)
        self.assertFalse(d.empty)
        np.testing.assert_allclose(d.IV_Percent, 30., atol=1e-4)
        self.assertTrue((d.forwardSource == 'put-call parity').all())
        np.testing.assert_allclose(d.YahooIV_Percent, 50.)  # kept for comparison only


class SnapshotTests(unittest.TestCase):
    def test_snapshot_loads_with_consistent_calls_and_puts(self):
        spot, d, as_of = load_aapl_snapshot()
        self.assertGreater(len(d), 100)
        self.assertGreater(d.expiration.nunique(), 5)
        self.assertTrue(d.IV_Percent.between(5, 150).all())
        # Put-call parity: at the same strike and expiry, computed call and put
        # IVs should be close (Yahoo's reported values differ by ~7 vol points).
        pairs = d.pivot_table(index=['expiration', 'strike'], columns='optionType', values='IV_Percent').dropna()
        self.assertLess((pairs.Call - pairs.Put).abs().median(), 2.)


if __name__ == '__main__': unittest.main()

import unittest
import numpy as np
import pandas as pd
from modules.engine import historical_episode, build_features

class HistoricalTests(unittest.TestCase):
    def test_past_only_and_shrinking_horizon(self):
        prices=pd.Series(100*np.exp(np.arange(400)*.0002),index=pd.bdate_range('2024-01-01',periods=400))
        calls=[]
        def forecast(p,horizon):
            calls.append((p.index[-1],horizon))
            return dict(Historical=30.,GARCH=30.,XGBoost=30.)
        r=historical_episode(prices,forecaster=forecast)
        self.assertEqual(calls,list(zip(prices.index[-6:-1],[5,4,3,2,1])))
        self.assertEqual(len(r['path']),6)
        self.assertEqual(r['summary']['Terminal P&L ($)'].nunique(),1)
        self.assertTrue((r['ledger'].groupby('Strategy').tail(1)['Shares after trade']==0).all())

    def test_horizon_labels_are_mature(self):
        returns=pd.Series(np.arange(1,101,dtype=float))
        for h in range(1,6):
            f=build_features(returns,h)
            self.assertTrue(f.target.iloc[-h:].isna().all())
            self.assertAlmostEqual(f.target.iloc[-h-1],np.mean(returns.iloc[-h:]**2))

if __name__=='__main__':unittest.main()

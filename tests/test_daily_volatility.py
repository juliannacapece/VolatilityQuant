import unittest
import numpy as np
from modules.engine import simulate_path, hedge, volatility_schedule

class DailyVolatilityTests(unittest.TestCase):
    def test_constant_compatibility(self):
        p=simulate_path(.35)
        np.testing.assert_allclose(p,simulate_path([.35]*5))
        a=hedge(p,100,5/252,.04,.25,.35)
        b=hedge(p,100,5/252,.04,[.25]*5,[.35]*5)
        np.testing.assert_allclose(a.PortfolioValue,b.PortfolioValue)

    def test_remaining_variance_prices_deterministic_schedule(self):
        # Different schedules with identical integrated variance have the same
        # opening price/delta when the strategy uses their equivalent flat vol.
        vols=np.array([.2,.3,.4,.5,.6]); equivalent=np.sqrt(np.mean(vols**2))
        p=np.full(6,100.)
        a=hedge(p,100,5/252,.04,equivalent,vols,cost=0)
        b=hedge(p,100,5/252,.04,equivalent,equivalent,cost=0)
        np.testing.assert_allclose(a.PortfolioValue.iloc[[0,-1]],b.PortfolioValue.iloc[[0,-1]])

    def test_hedge_changes_do_not_change_path(self):
        p=simulate_path([.2,.3,.4,.5,.6])
        a=hedge(p,100,5/252,.04,.2,.4)
        b=hedge(p,100,5/252,.04,[.6,.5,.4,.3,.2],.4)
        np.testing.assert_array_equal(a.Stock,b.Stock)
        self.assertFalse(np.allclose(a.Delta,b.Delta))

    def test_invalid_schedule(self):
        for value in ([.2,.3],[.2]*4+[np.nan],0):
            with self.assertRaises(ValueError):volatility_schedule(value,5)

if __name__=='__main__':unittest.main()

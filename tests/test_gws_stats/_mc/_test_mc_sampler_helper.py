
import arviz as az
from gws_core import BaseTestCaseLight
from gws_stats import MCLinRegData, MCLinRegSampler
from pandas import DataFrame


class TestMCSampler(BaseTestCaseLight):

    def test_mc_sampler(self):
        sampler = MCLinRegSampler()

        sampler.set_slope_priors([
            {"func": "Normal", "mu": 0.8, "sigma": 1.0},
            {"func": "Normal", "mu": 0.8, "sigma": 1.0}
        ])
        sampler.set_sigma_prior({"func": "HalfCauchy", "beta": 10})

        x_data = DataFrame({
            "var1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0],
            "var2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0]
        })
        y_data = DataFrame({"y": [1.0, 2.0, 3.0, 10.0, 5.0, 6.0, 7.0, 20.0, 9.0, 9.0, 10.0]})
        data = MCLinRegData(x_data=x_data, y_data=y_data)

        sampler.set_data(data)
        result = sampler.trace(random_seed=[42, 43])
        traces = result.get_traces()

        slope = result.get_slope_traces()

        m = slope.mean()
        self.assertAlmostEqual(m.iat[0], 0.547745, places=3)
        self.assertAlmostEqual(m.iat[1], 0.477969, places=3)

        preds = result.get_predictions(num_samples=100)

        pred_stats = result.get_prediction_stats(num_samples=100)

    def test_mc_sampler_with_mixtures(self):
        sampler = MCLinRegSampler()

        sampler.set_slope_priors([
            {"func": "Normal", "mu": 0.8, "sigma": 1.0},
            {"func": "Mixture", "dist": [
                {"func": "Uniform", "lower": -100, "upper": 100},
                {"func": "Normal", "mu": 0.8, "sigma": 1.0},
            ]}
        ])
        sampler.set_sigma_prior({"func": "HalfCauchy", "beta": 10})

        x_data = DataFrame({
            "var1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0],
            "var2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0]
        })
        y_data = DataFrame({"y": [1.0, 2.0, 3.0, 10.0, 5.0, 6.0, 7.0, 20.0, 9.0, 9.0, 10.0]})
        data = MCLinRegData(x_data=x_data, y_data=y_data)

        sampler.set_data(data)
        result = sampler.trace(random_seed=[42, 43])
        traces = result.get_traces()

        slope = result.get_slope_traces()

        m = slope.mean()
        self.assertAlmostEqual(m.iat[0], 0.523255, places=3)
        self.assertAlmostEqual(m.iat[1], 0.503547, places=3)

        preds = result.get_predictions(num_samples=100)

        pred_stats = result.get_prediction_stats(num_samples=100)

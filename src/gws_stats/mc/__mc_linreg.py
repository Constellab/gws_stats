
from gws_core import (ConfigParams, FloatParam, InputSpec, OutputSpec,
                      ParamSet, StrParam, Table, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)

from ..base.helper.simple_design_helper import SimpleDesignHelper
from .sampler.mc_linreg_sampler import MCLinRegData, MCLinRegSampler

# *****************************************************************************
#
# NormalTestResultTable
#
# *****************************************************************************


@resource_decorator("MCLinearRegressorTable", human_name="MC linear regressor table", hide=True)
class MCLinearRegressorTable(Table):
    """ MCLinearRegressorTable """

# *****************************************************************************
#
# MCLinRegressor
#
# *****************************************************************************


@task_decorator("MCLinearRegressor", human_name="MC linear regressor",
                short_description="Monte-Carlo linear regressior")
class MCLinearRegressor(Task):
    """
    Robust Monte-Carlo based linear regression

    Based on pyMC package.

    Usual distribution functions are
    * `Normal` with paramerers `mu, sigma`
    * `TruncatedNormal` with paramerers `mu, sigma, lower, upper`
    * `HalfNormal` with paramerers `mu, sigma`
    * `HalfCauchy` with paramerers `beta`

    See also https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-robust.html
    """

    input_specs = InputSpecs({'table': InputSpec(
        Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(MCLinearRegressorTable, human_name="result",
                                                     short_description="The output result")})
    config_specs = {
        'training_design': SimpleDesignHelper.create_training_design_param_set(),
        # 'intercept': ParamSet({
        #     "func": StrParam(allowed_values=["Normal", "TruncatedNormal", "HalfNormal"], human_name="Distribution function"),
        #     "mu": FloatParam(value, human_name="Distribution function"),
        #     "params": StrParam(human_name="Parameters", short_description="The parameters (e.g. mu=1.2, sigma=3). See documentation."),
        # }, human_name="Intercept"),
        'slope': ParamSet({
            "func": StrParam(allowed_values=["Normal", "TruncatedNormal", "HalfNormal"], human_name="Function", short_description="Distribution function"),
            "mu": FloatParam(human_name="Mean", short_description="Distribution mean"),
            "sigma": FloatParam(human_name="Sigma", short_description="Distribution sigma"),
            "lb": FloatParam(optional=True, human_name="Lower bound", short_description="The upper bound of the distribution. It is required for TruncatedNormal and used as constrain in any case."),
            "ub": FloatParam(optional=True, human_name="Upper bound", short_description="The lower bound of the distribution. It is required for TruncatedNormal and used as constrain in any case."),
        }, human_name="Slope", max_number_of_occurrences=1),
        'sigma': ParamSet({
            "func": StrParam(allowed_values=["Normal", "TruncatedNormal", "HalfNormal", "HalfCauchy"], human_name="Distribution function"),
            "mu": FloatParam(optional=True, default=0.0, human_name="Mean", short_description="Distribution mean. Only for Normal-type distribution"),
            "sigma": FloatParam(optional=True, default=10, human_name="Sigma", short_description="Distribution sigma. Only for Normal-type distribution"),
            "beta": FloatParam(optional=True, default=10, human_name="Sigma", short_description="Distribution beta. Only for HalfCauchy distribution"),
        }, human_name="Sigma", max_number_of_occurrences=1)
    }

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs["table"]
        training_design = params["training_design"]

        x_true, y_true = SimpleDesignHelper.create_training_matrices(
            table, training_design)
        data = MCLinRegData(x_data=x_true, y_data=y_true)
        sampler = MCLinRegSampler()
        sampler.set_data(data)

        slope_params = params["slope"]

        intercept = params["intercept"]
        slope = params["slope"]
        sigma = params["sigma"]
        sampler.set_intercept_priors(
            intercept["func"], self._parse_params(intercept["params"]))
        sampler.set_slope_priors(
            slope["func"], self._parse_params(slope["params"]))
        sampler.set_sigma_prior(
            sigma["func"], self._parse_params(sigma["params"]))

        result = sampler.sample()

        return {"result": result}

    @staticmethod
    def _parse_params(params):
        kwargs = {}
        tab = params.split(",")
        for val in tab:
            kv = val.split("=")
            k = kv[0].strip()
            v = float(kv[1].strip())
            kwargs[k] = v

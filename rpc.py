import arviz as az
import matplotlib .pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from ranking_dataset import RankingDataset
from typing import List
from hurdle_ordered_logistic import HurdleOrderedLogistic, ZeroInflatedOrderedLogistic, OrderedLogistic
import xarray as xr


class RPC:
    coeff_mat = {
        1: pt.as_tensor(
            [[ 1., 0.],
             [-1., 1.]]
        ),
        2: pt.as_tensor(
            [[ 1., 0., 0.],
             [-2., 2., 0.],
             [ 1., -2., 1.]]
        ),
        3: pt.as_tensor(
            [[ 1.,  0.,  0., 0.],
             [-3.,  3.,  0., 0.],
             [ 3., -6.,  3., 0.],
             [-1.,  3., -3., 1.]]
        ),
        4: pt.as_tensor(
            [[ 1.,   0.,   0.,  0., 0.],
             [-4.,   4.,   0.,  0., 0.],
             [ 6., -12.,   6.,  0., 0.],
             [-4.,  12., -12.,  4., 0.],
             [ 1.,  -4.,   6., -4., 1.]]
        ),
        5: pt.as_tensor(
            [[  1.,   0.,   0.,   0.,  0., 0.],
             [ -5.,   5.,   0.,   0.,  0., 0.],
             [ 10., -20.,  10.,   0.,  0., 0.],
             [-10.,  30., -30.,  10.,  0., 0.],
             [  5., -20.,  30., -20.,  5., 0.],
             [ -1.,   5., -10.,  10., -5., 1.]]
        ),
        6: pt.as_tensor(
            [[  1.,   0.,   0.,   0.,   0.,  0., 0.],
             [ -6.,   6.,   0.,   0.,   0.,  0., 0.],
             [ 15., -30.,  15.,   0.,   0.,  0., 0.],
             [-20.,  60., -60.,  20.,   0.,  0., 0.],
             [ 15., -60.,  90., -60.,  15.,  0., 0.],
             [ -6.,  30., -60.,  60., -30.,  6., 0.],
             [  1.,  -6.,  15., -20.,  15., -6., 1.]]
        ),
        7: pt.as_tensor(
            [[  1.,    0.,    0.,    0.,    0.,   0.,  0., 0.],
             [ -7.,    7.,    0.,    0.,    0.,   0.,  0., 0.],
             [ 21.,  -42.,   21.,    0.,    0.,   0.,  0., 0.],
             [-35.,  105., -105.,   35.,    0.,   0.,  0., 0.],
             [ 35., -140.,  210., -140.,   35.,   0.,  0., 0.],
             [-21.,  105., -210.,  210., -105.,  21.,  0., 0.],
             [  7.,  -42.,  105., -140.,  105., -42.,  7., 0.],
             [ -1.,    7.,  -21.,   35.,  -35.,  21., -7., 1.]]
        )
    }

    def __init__(
            self,
            dataset: RankingDataset,
            order: int = 3,
            prior_sigma: float = 4.,
            prior_cutpoints_sigma = 1e-12,
            near_zero = 1e-15,
            use_latent_cutpoints=False, 
            use_hurdle=False,
            use_zero_inflation=False,
            extend_dataset=True,
            rng_seed: int = 0
    ):
        assert not (use_zero_inflation and use_hurdle)
        assert order in self.coeff_mat.keys()

        self.dataset = dataset

        self.near_zero = near_zero
        self.use_latent_cutpoints = use_latent_cutpoints
        self.use_hurdle = use_hurdle
        self.use_zero_inflation = use_zero_inflation
        self.extend_dataset = extend_dataset
        self.attribute_labels = self.dataset.attribute_labels
        self.num_attributes = len(self.attribute_labels)

        self.dataframe = self.dataset.dataframe[self.attribute_labels + [self.dataset.normalized_attribute_sum_name]].copy()

        self.unique_data_array, self.data_indices = np.unique(
            self.dataframe.values,
            axis=0,
            return_inverse=True
        )
        self.normalized_attribute_sum_array = self.unique_data_array[:, self.num_attributes]
        self.unique_data_array = self.unique_data_array[:, :self.num_attributes]
        self.num_scores = len(self.unique_data_array)

        unique_data_mins = np.expand_dims(np.min(self.unique_data_array, axis=0), axis=0)
        self.observed_data_array = self.unique_data_array - unique_data_mins
        self.input_data_array = self._min_max_scale(self.observed_data_array)

        self.order = order
        self.order_range = list((range(1, self.order)))

        self.prior_sigma = prior_sigma
        self.rng = np.random.default_rng(rng_seed)

        self.point_sigma_name = "point_sigma"
        self.alpha_name = "alpha"
        self.score_sigma_name = "score_sigma"
        self.score_name = "score"
        self.rank_name = "rank"
        self.tangent_weight_name = "tangent_weight"

        self.priors = {
            self.point_sigma_name: np.repeat([self.prior_sigma], self.num_attributes),
            self.score_sigma_name: self.prior_sigma,
            self.score_name: self._min_max_scale(self.normalized_attribute_sum_array)
        }

        if self.order == 1:
            self.priors[self.alpha_name] = self.rng.random(self.num_attributes)
        else:
            self.control_point_priors = {
                f"control_point_{_}": self.rng.random(self.num_attributes)
                for _ in self.order_range
            }
            self.priors.update(
                self.control_point_priors
            )
        
        self.attribute_mins = {label: self.dataframe[label].min() for label in self.attribute_labels}
        self.attribute_maxs = {label: self.dataframe[label].max() for label in self.attribute_labels}

        if self.use_latent_cutpoints:
            self.prior_cutpoints_sigma = prior_cutpoints_sigma

            self.cutpoint_priors = {
                f"{label}_latent_cutpoints": np.arange(
                    self.attribute_mins[label] + 1.,
                    self.attribute_maxs[label] - 1.
                ) + .5
                for label in self.attribute_labels
            }
            self.priors.update(
                self.cutpoint_priors
            )
            self.cutpoint_std_priors = {
                f"{label}_latent_cutpoints_sigma": self.prior_cutpoints_sigma for label in self.attribute_labels
            }
            self.priors.update(
                self.cutpoint_std_priors
            )

        self.inference_data = None
        self.scores = None
        self.tangent_weights = None

        self.loo_cross_validation_metrics = None
        self.loo_cross_validation_score = None

    def _min_max_scale(self, x, _min=None, _max=None):
        if _min is None:
            _min = self.near_zero
        
        if _max is None:
            _max = 1. - self.near_zero

        x_min = np.expand_dims(np.min(x, axis=0), axis=0)
        x_max = np.expand_dims(np.max(x, axis=0), axis=0)

        x_scaled = (x - x_min) / (x_max - x_min)
        
        return x_scaled * (_max - _min) + _min

    def build(self):
        with pm.Model() as self.model:
            point_sigma = pm.HalfNormal(
                name=self.point_sigma_name,
                sigma=self.priors[self.point_sigma_name],
                # initval="prior"
            )
            if self.order == 1:
                alpha = pm.TruncatedNormal(
                    name=self.alpha_name,
                    mu=self.priors[self.alpha_name],
                    sigma=point_sigma,
                    lower=0.,
                    upper=1.,
                    # initval="prior"
                )
                control_points = [
                    pm.Deterministic(name="control_point_0", var=.5 * (1. - alpha)),
                    pm.Deterministic(name=f"control_point_{self.order}", var=.5 * (1. + alpha))
                ]
            else:
                control_points = [
                    pt.as_tensor(np.zeros(self.num_attributes))
                ] + [
                    pm.TruncatedNormal(
                        name=name,
                        mu=mu,
                        sigma=point_sigma,
                        lower=0.,
                        upper=1.,
                        # initval="prior"
                    ) for name, mu in self.control_point_priors.items()
                ] + [
                    pt.as_tensor(np.ones(self.num_attributes))
                ]
            
            score_sigma = pm.HalfNormal(
                name=self.score_sigma_name,
                sigma=self.priors[self.score_sigma_name],
                # initval="prior"
            )
            score = pm.TruncatedNormal(
                name=self.score_name,
                mu=self.priors[self.score_name],
                sigma=score_sigma,
                lower=0.,
                upper=1.,
                # initval="prior"
            )
            Z = pm.Deterministic(
                name="Z",
                var=pt.transpose(
                    pm.math.stack(
                        [pt.ones((self.num_scores,)), score] + [
                            score ** (order + 1.) for order in self.order_range
                        ],
                        axis=0
                    )
                )
            )
            M = self.coeff_mat[self.order]

            P = pm.Deterministic(
                name="P",
                var=pm.math.stack(control_points, axis=0)
            )
            X_reconstruction_mu = pm.math.matmul(pm.math.matmul(Z, M), P)

            for index, column in enumerate(self.attribute_labels):
                observed = pt.as_tensor(self.observed_data_array[:, index])
            
                score_min = self.attribute_mins[column]
                score_max = self.attribute_maxs[column]

                if self.use_latent_cutpoints:
                    attribute_cutpoints_std = pm.HalfNormal(
                        name=f"{column}_latent_cutpoints_sigma",
                        sigma=self.priors[f"{column}_latent_cutpoints_sigma"],
                        # initval="prior"
                    )
                    attribute_cutpoints = pm.math.concatenate([
                        np.ones(1) * score_min + .5,
                        pm.LogNormal(
                            name=f"{column}_latent_cutpoints",
                            mu=self.priors[f"{column}_latent_cutpoints"],
                            sigma=attribute_cutpoints_std,
                            transform=pm.distributions.transforms.Ordered(),
                            # initval="prior"
                        ),
                        np.ones(1) * score_max - .5
                    ])
                else:
                    attribute_cutpoints = np.arange(score_min, score_max) + .5
            
                attribute_eta = pm.Deterministic(
                    name=f"{column}_eta",
                    var=X_reconstruction_mu[:, index] * (score_max - score_min) + score_min
                )

                if self.use_hurdle:
                    attribute_psi = pm.Deterministic(
                        name=f"{column}_psi",
                        var=pt.true_div(pt.sum(pt.where(
                            observed >= attribute_cutpoints[0], 1., 0.
                            )), observed.shape[0])
                    )

                    attribute_response = HurdleOrderedLogistic(
                        name=f"{column}_response",
                        psi=attribute_psi,
                        eta=attribute_eta,
                        cutpoints=attribute_cutpoints,
                        observed=observed
                    )
                else:
                    attribute_response = pm.OrderedLogistic(
                        f"{column}_response",
                        eta=attribute_eta,
                        cutpoints=attribute_cutpoints,
                        observed=observed
                    )

    def plot_prior_predictive(self, draws=500):
        assert self.model is not None, "model not yet built"

        with self.model:
            prior_predictions = pm.sample_prior_predictive(
                draws=draws,
                random_seed=self.rng
            )
        
        if self.inference_data is None:
            self.inference_data = prior_predictions
        else:
            self.inference_data.extend(prior_predictions, join="right")
        
        az.plot_ppc(prior_predictions, group="prior")

    def fit(
            self,
            nuts_sampler="pymc",
            nuts_sampler_kwargs={"chain_method": "vectorized"},
            target_accept=.5,
            tune=1000,
            draws=1000,
            chains=4,
            cores=4,
            initvals=None
    ):
        assert self.model is not None, "model not yet built"

        if initvals is None:
            initvals = self.priors
        
        with self.model:
            posterior_samples = pm.sample(
                nuts_sampler=nuts_sampler,
                nuts_sampler_kwargs=nuts_sampler_kwargs,
                target_accept=target_accept,
                random_seed=self.rng,
                tune=tune,
                draws=draws,
                chains=chains,
                cores=cores,
                initvals=initvals,
                discard_tuned_samples=True
            )

            if self.inference_data is None:
                self.inference_data = posterior_samples
            else:
                self.inference_data.extend(posterior_samples, join="right")
            
            self.scores = np.mean(self.inference_data.posterior[self.score_name], axis=1)

            self.tangent_weights = self._add_tangent_weights_to_inference_data()

            if self.extend_dataset:
                self._add_scores_and_weighted_sums_to_dataset()
    
    def _compute_log_likelihood(self):
        assert self.inference_data is not None, "model not yet fit"

        with self.model:
            pm.compute_log_likelihood(self.inference_data)

    def get_loo_cross_validation(self):
        assert self.inference_data is not None, "model not yet fit"

        if "log_likelihood" not in self.inference_data:
            self._compute_log_likelihood()

        if self.loo_cross_validation_metrics is None:
            self.loo_cross_validation_metrics = {
                f"{label}_loo": az.loo(self.inference_data, var_name=f"{label}_response") 
                for label in self.attribute_labels
            }

        if self.loo_cross_validation_score is None:
            self.loo_cross_validation_score = sum([metric.elpd_loo for metric in self.loo_cross_validation_metrics.values()])

        return self.loo_cross_validation_score

    def _add_tangent_weights_to_inference_data(self):
        assert (self.inference_data is not None 
                and "posterior" in self.inference_data 
                and self.score_name in self.inference_data.posterior)

        Z = self.inference_data.posterior["Z"].values
        M = self.coeff_mat[self.order]
        P = self.inference_data.posterior["P"].values

        Xr = pm.math.matmul(pm.math.matmul(Z, M), P)

        Z_prime = Z[:, :, :, :self.order]
        M_prime = self.coeff_mat[self.order - 1]
        P_prime = self.order * np.diff(P, axis=2)

        Xr_prime = pm.math.matmul(pm.math.matmul(Z_prime, M_prime), P_prime)

        X_min = np.expand_dims(np.min(self.unique_data_array, axis=0), axis=[0, 1])
        X_max = np.expand_dims(np.max(self.unique_data_array, axis=0), axis=[0, 1])

        X = Xr * (X_max - X_min) + X_min

        X_weighted = (Xr_prime * X).eval()

        X_weighted_dataset = xr.Dataset(
            data_vars={
                self.tangent_weight_name: ([
                    "chain", 
                    "draw", 
                    f"{self.tangent_weight_name}_dim_0", 
                    f"{self.tangent_weight_name}_dim_1"
                    ], X_weighted)
                }, 
            coords={
                "chain": (["chain"], np.arange(X_weighted.shape[0])),
                "draw": (["draw"], np.arange(X_weighted.shape[1])),
                f"{self.tangent_weight_name}_dim_0": ([f"{self.tangent_weight_name}_dim_0"], np.arange(X_weighted.shape[2])),
                f"{self.tangent_weight_name}_dim_1": ([f"{self.tangent_weight_name}_dim_1"], np.arange(X_weighted.shape[3]))
            }
        )
        self.inference_data.add_groups({"analytics": X_weighted_dataset})

        posterior_tangent_weight = self.inference_data.analytics[self.tangent_weight_name]
        mean_tangent_weight = posterior_tangent_weight.mean(("chain", "draw"))
        sum_tangent_weight = mean_tangent_weight.sum((f"{self.tangent_weight_name}_dim_1"))

        return self._min_max_scale(
            sum_tangent_weight.values, 
            np.min(self.scores).item(),
            np.max(self.scores).item()
        )
    
    def _add_scores_and_weighted_sums_to_dataset(self):
        scores_mean = np.mean(self.scores, axis=0)
        data_with_scores = np.concatenate(
            (self.unique_data_array, np.expand_dims(scores_mean, axis=1)), 
            axis=1
            )
        self.dataset.dataframe.insert(
            len(self.dataset.dataframe.columns), 
            self.score_name, 
            data_with_scores[self.data_indices][:, -1]
            )
        self.dataset.dataframe.sort_values(self.score_name, inplace=True, ascending=False)

        unique_scores, nonunique_indices = np.unique(
            self.dataset.dataframe[self.score_name].values, 
            return_inverse=True
            )
        unique_ranks = np.arange(len(unique_scores))
        nonunique_indices = np.abs(nonunique_indices - nonunique_indices[0])
        nonunique_ranks = unique_ranks[nonunique_indices] + 1

        self.dataset.dataframe.insert(
            len(self.dataset.dataframe.columns), self.rank_name, nonunique_ranks)
    
    def plot_posterior_tangent_weight_traces(self):
        if "analytics" not in self.inference_data or self.tangent_weight_name not in self.inference_data.analytics:
            self._add_tangent_weights_to_inference_data()

        posterior_tangent_weight = self.inference_data.analytics[self.tangent_weight_name]
        mean_tangent_weight = posterior_tangent_weight.mean(("chain", "draw"))
        sum_tangent_weight = mean_tangent_weight.sum((f"{self.tangent_weight_name}_dim_1"))

        y = np.linspace(0, 1, len(sum_tangent_weight))
        hdi = az.hdi(posterior_tangent_weight).sortby(sum_tangent_weight)

        plt.plot(sum_tangent_weight.sortby(sum_tangent_weight), y)
        plt.fill_betweenx(
            y, 
            hdi[self.tangent_weight_name].sum(f"{self.tangent_weight_name}_dim_1").values[:, 0], 
            hdi[self.tangent_weight_name].sum(f"{self.tangent_weight_name}_dim_1").values[:, 1], 
            alpha=0.3
            )
    
    def plot_posterior_predictive(self):
        assert self.inference_data is not None, "model not yet fit"

        with self.model:
            posterior_predictive_idata = pm.sample_posterior_predictive(
                self.inference_data, extend_inferencedata=True, random_seed=self.rng)
        az.plot_ppc(posterior_predictive_idata)
    
    def plot_posterior_score_traces(self):
        assert self.inference_data is not None, "model not yet fit"
        
        posterior_score = self.inference_data.posterior[self.score_name]
        mean_score = posterior_score.mean(("chain", "draw"))

        y = np.linspace(0, 1, len(mean_score))
        hdi = az.hdi(posterior_score).sortby(mean_score)

        plt.plot(mean_score.sortby(mean_score), y)
        plt.fill_betweenx(
            y,
            hdi[self.score_name].values[:, 0],
            hdi[self.score_name].values[:, 1],
            alpha=.3
        )
        plot_trace = az.plot_trace(
            self.inference_data,
            var_names=[self.score_sigma_name]
        )
    
    def plot_posterior_point_traces(self):
        assert self.inference_data is not None, "model not yet fit"
        
        plot_trace = az.plot_trace(
            self.inference_data,
            var_names=[
                key for key in self.priors.keys() 
                if key not in [
                    self.score_name, 
                    self.score_sigma_name, 
                    self.tangent_weight_name
                    ]
                ]
        )
    

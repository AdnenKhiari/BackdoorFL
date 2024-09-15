from flwr.server.strategy import FedMedian

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

import numpy as np
from scipy.spatial.distance import cdist


class GeoMedianStrategy(FedMedian):
    
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedGeoMedian(accept_failures={self.accept_failures})"
        return rep
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using geometric median."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results to list of arrays
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # Flatten the weights of each client
        flat_weights = [np.concatenate([layer.flatten() for layer in weights]) for weights in weights_results]

        # Use geometric median to aggregate flattened weights
        aggregated_flat_weights = geometric_median(flat_weights)

        # Unflatten the result back to original shape
        layer_shapes = [layer.shape for layer in weights_results[0]]
        aggregated_weights = []
        idx = 0
        for shape in layer_shapes:
            size = np.prod(shape)
            aggregated_weights.append(aggregated_flat_weights[idx:idx + size].reshape(shape))
            idx += size

        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated



def geometric_median(points, options={}):
    """
    Calculates the geometric median of an array of points using Weiszfeld's algorithm.
    """

    points = np.asarray(points)

    if len(points.shape) == 1:
        raise ValueError("Expected 2D array")

    return weiszfeld_method(points, options)


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm to compute the geometric median.
    """

    default_options = {'maxiter': 1000, 'tol': 1e-7}
    default_options.update(options)
    options = default_options

    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)

    iters = 0

    while iters < options['maxiter']:
        distances = distance_func(guess).T

        # Handle divide by zero
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points/distances).sum(axis=0) / (1./distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next)**2).sum())

        guess = guess_next

        if guess_movement <= options['tol']:
            break

        iters += 1

    return np.array(guess)

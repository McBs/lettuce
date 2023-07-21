"""
Turbulence functions.
"""

import torch
from typing import Tuple

__all__ = [
    "kolmogorov_scales", "characteristic_length_lambda", "reynolds_number_lambda"
]

def kolmogorov_scales(dissipation: float, viscosity: float) -> Tuple[float, float, float]:
    """
    Calculates the Kolmogorov scales for a given dissipation and viscosity.

    :param dissipation: The dissipation value.
    :param viscosity: The viscosity value.
    :return: A tuple containing the calculated values for eta, tau, and u_eta.
    """
    eta = (viscosity**3/dissipation)**(1/4)
    tau = (viscosity/dissipation)**(1/2)
    u_eta = (viscosity*dissipation)**(1/4)
    return eta, tau, u_eta

def characteristic_length_lambda(dissipation: torch.float, viscosity: torch.float, u_rms: torch.float) -> torch.float:
    """

    :param dissipation: The dissipation value.
    :param viscosity: The viscosity value.
    :param u_rms: The root mean square velocity value.
    :return: The calculated characteristic length scale lambda.
    """
    return torch.sqrt(15 * viscosity * u_rms**2 / dissipation)

def reynolds_number_lambda(lamda: torch.float, viscosity: torch.float, u_rms: torch.float) -> torch.float:
    """
    Calculates the Reynolds number for a given characteristic length scale lambda, viscosity, and u_rms.

    :param lamda: The characteristic length scale value.
    :param viscosity: The viscosity value.
    :param u_rms: The root mean square velocity value.
    :return: The calculated Reynolds number.
    """
    return u_rms * lamda / viscosity


__all__ = ["RevNetBlock"]

from typing import Tuple

import torch

from .backprop import InvertibleBlock


class RevNetBlock(InvertibleBlock):
    def __init__(self, left2right: torch.nn.Module, right2left: torch.nn.Module):
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        left2right : torch.nn.Module
            [TODO:description]
        right2left : torch.nn.Module
            [TODO:description]
        """
        super().__init__()
        self._left2right = left2right
        self._right2left = right2left

    def _forward(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        x_left : torch.Tensor
            [TODO:description]
        x_right : torch.Tensor
            [TODO:description]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]:
            [TODO:description]
        """
        y_left = x_left + self._right2left(x_right)
        y_right = x_right + self._left2right(y_left)
        return y_left, y_right

    def _inverse(
        self, y_left: torch.Tensor, y_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        y_left : torch.Tensor
            [TODO:description]
        y_right : torch.Tensor
            [TODO:description]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]:
            [TODO:description]
        """
        x_right = y_right - self._left2right(y_left)
        x_left = y_left - self._right2left(x_right)
        return x_left, x_right

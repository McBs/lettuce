__all__ = ["InvertibleBlock", "MemoryEfficientNet"]

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union

import torch


class MemoryFreeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, block: "InvertibleBlock", n_elems: int, *zs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        ctx : [TODO:type]
            [TODO:description]
        block : "InvertibleBlock"
            [TODO:description]
        n_elems : int
            [TODO:description]
        zs : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Tuple[torch.Tensor]:
            [TODO:description]
        """
        xs = zs[:n_elems]
        dummy = zs[n_elems : 2 * n_elems]
        params = zs[2 * n_elems :]
        with torch.no_grad():
            ys = block._forward(*xs)
        ctx.block = block
        ctx.n_elems = n_elems
        return (*ys, *dummy, *params)

    @staticmethod
    def backward(ctx, *out: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        ctx : [TODO:type]
            [TODO:description]
        out : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Tuple[torch.Tensor]:
            [TODO:description]
        """
        grads_out = out[: ctx.n_elems]
        ys = out[ctx.n_elems : 2 * ctx.n_elems]
        grads = ctx.block.grad(ys, grads_out)
        return (None, None, *grads)


class MemoizeOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *xs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        ctx : [TODO:type]
            [TODO:description]
        xs : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Tuple[torch.Tensor]:
            [TODO:description]
        """
        xs = xs[: len(xs) // 2]
        ctx.save_for_backward(*xs)
        return xs

    @staticmethod
    def backward(ctx, *grad_out: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        ctx : [TODO:type]
            [TODO:description]
        grad_out : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Tuple[torch.Tensor]:
            [TODO:description]
        """
        xs = ctx.saved_tensors
        grad = [g * torch.ones_like(x) for (g, x) in zip(grad_out, xs)]
        return (*grad, *xs)


class InvertibleBlock(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(
        self, xs: Sequence[torch.Tensor], dummies: Sequence[torch.Tensor]
    ) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        xs : Sequence[torch.Tensor]
            [TODO:description]
        dummies : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
            [TODO:description]
        """
        n_elems = len(xs)
        out = MemoryFreeFunction.apply(self, n_elems, *xs, *dummies, *self.parameters())
        ys = out[:n_elems]
        dummies = out[n_elems : 2 * n_elems]
        return ys, dummies

    @abstractmethod
    def _forward(self, *xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        self : [TODO:type]
            [TODO:description]
        xs : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Sequence[torch.Tensor]:
            [TODO:description]
        """
        pass

    @abstractmethod
    def _inverse(self, *xs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        self : [TODO:type]
            [TODO:description]
        xs : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Sequence[torch.Tensor]:
            [TODO:description]
        """
        pass

    def grad(
        self, ys: Sequence[torch.Tensor], grads_out: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        ys : Sequence[torch.Tensor]
            [TODO:description]
        grads_out : Sequence[torch.Tensor]
            [TODO:description]

        Returns
        -------
        Tuple[torch.Tensor]:
            [TODO:description]
        """
        with torch.no_grad():
            xs = self._inverse(*ys)
        with torch.enable_grad():
            xs = [x.detach().requires_grad_(True) for x in xs]
            in_vars = xs + list(self.parameters())
            ys = self._forward(*xs)
            g = torch.autograd.grad(ys, in_vars, grads_out, allow_unused=True)
            return g[: len(ys)] + tuple(xs) + g[len(ys) :]


class MemoryEfficientNet(torch.nn.Module):
    def __init__(self, blocks: Sequence[InvertibleBlock]):
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        blocks : Sequence[InvertibleBlock]
            [TODO:description]
        """
        super().__init__()
        self._blocks = torch.nn.ModuleList(blocks)

    def forward(
        self, *xs: Sequence[torch.nn.Module]
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        [TODO:summary]

        [TODO:description]

        Parameters
        ----------
        xs : Sequence[torch.nn.Module]
            [TODO:description]

        Returns
        -------
        Union[torch.Tensor, Sequence[torch.Tensor]]:
            [TODO:description]
        """
        dummy = [torch.empty_like(x) for x in xs]
        for block in self._blocks:
            xs, dummy = block(xs, dummy)
        ys = MemoizeOutput.apply(*xs, *dummy)
        if len(ys) == 1:
            ys = ys[0]
        return ys

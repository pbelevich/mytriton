import inspect

from .ssa import SSALowering
from .trace import constexpr, trace


class CompiledKernel:
    def __init__(self, fn):
        self.fn = fn
        signature = inspect.signature(fn)
        annotations = inspect.get_annotations(fn, eval_str=True)
        parameters = [
            parameter.replace(
                annotation=annotations.get(name, parameter.annotation),
            )
            for name, parameter in signature.parameters.items()
        ]
        self.signature = signature.replace(parameters=parameters)

    def __getitem__(self, grid):

        def launch(*args, **kwargs):
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()

            meta = {
                name: bound.arguments[name]
                for name, parameter in self.signature.parameters.items()
                if parameter.annotation is constexpr
            }

            launch_grid = grid(meta) if callable(grid) else grid
            if isinstance(launch_grid, int):
                launch_grid = (launch_grid,)
            launch_grid = tuple(launch_grid)

            if not launch_grid or any(
                not isinstance(x, int) or x <= 0 for x in launch_grid
            ):
                raise ValueError(f"invalid launch grid: {launch_grid}")

            ops = trace(self.fn, self.signature, bound.arguments)

            ssa = SSALowering().lower(ops)

            return ops, ssa

        return launch


def jit(fn):
    return CompiledKernel(fn)

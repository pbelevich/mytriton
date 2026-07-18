from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Mapping
from typing import Any

from . import language as tl
from .trace import (
    Builder,
    PointerType,
    Ptr,
    Value,
    is_constexpr_annotation,
    make_runtime_params,
)


class ASTFrontendError(TypeError):
    pass


def _find_function_def(fn) -> ast.FunctionDef:
    source = inspect.getsource(fn)
    tree = ast.parse(textwrap.dedent(source))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            return node

    raise ASTFrontendError(f"could not find function definition for {fn.__name__}")


def _make_symbolic_env(
    signature, bound_args: Mapping[str, Any], runtime_params
) -> dict[str, Any]:
    env: dict[str, Any] = {}

    params_by_name = {param.name: param for param in runtime_params}

    for name, parameter in signature.parameters.items():
        value = bound_args[name]

        if is_constexpr_annotation(parameter.annotation):
            env[name] = value
            continue

        param = params_by_name[name]

        if isinstance(param.ty, PointerType):
            env[name] = Ptr(param)
        else:
            env[name] = Value(param)

    return env


class ASTTracer(ast.NodeVisitor):
    def __init__(self, env: dict[str, Any], external_env: Mapping[str, Any]) -> None:
        self.env = env
        self.external_env = external_env

    def visit_stmt_list(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self.visit(stmt)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ASTFrontendError(f"unsupported AST node: {type(node).__name__}")

    # ----------------------------
    # Statements
    # ----------------------------

    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise ASTFrontendError("only single-target assignment is supported")

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise ASTFrontendError("only assignment to a simple name is supported")

        self.env[target.id] = self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = node.target
        if not isinstance(target, ast.Name):
            raise ASTFrontendError("only assignment to a simple name is supported")

        if node.value is not None:
            self.env[target.id] = self.visit(node.value)
        else:
            self.env[target.id] = None

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise ASTFrontendError(
                "only augmented assignment to a simple name is supported"
            )

        name = node.target.id
        if name not in self.env:
            raise NameError(name)

        lhs = self.env[name]
        rhs = self.visit(node.value)

        if isinstance(node.op, ast.Add):
            self.env[name] = lhs + rhs
            return

        if isinstance(node.op, ast.Sub):
            self.env[name] = lhs - rhs
            return

        if isinstance(node.op, ast.Mult):
            self.env[name] = lhs * rhs
            return

        if isinstance(node.op, ast.Div):
            self.env[name] = lhs / rhs
            return

        raise ASTFrontendError(
            f"unsupported augmented assignment: {type(node.op).__name__}"
        )

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            raise ASTFrontendError(
                "only for loops with a simple induction variable are supported"
            )

        if node.orelse:
            raise ASTFrontendError("for/else is not supported")

        loop_range = self._eval_range_iterator(node.iter)

        name = node.target.id
        old_value = self.env.get(name)
        had_old_value = name in self.env

        for value in loop_range:
            self.env[name] = value
            self.visit_stmt_list(node.body)

        if had_old_value:
            self.env[name] = old_value
        else:
            del self.env[name]

    def visit_Return(self, node: ast.Return) -> None:
        raise ASTFrontendError("return statements are not supported in kernels")

    # ----------------------------
    # Expressions
    # ----------------------------

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.env:
                return self.env[node.id]

            if node.id in self.external_env:
                return self.external_env[node.id]

            raise NameError(node.id)

        if isinstance(node.ctx, ast.Store):
            return node.id

        raise ASTFrontendError(f"unsupported name context: {type(node.ctx).__name__}")

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        return getattr(value, node.attr)

    def visit_Call(self, node: ast.Call) -> Any:
        fn = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]

        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ASTFrontendError("**kwargs are not supported")
            kwargs[keyword.arg] = self.visit(keyword.value)

        return fn(*args, **kwargs)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return lhs + rhs

        if isinstance(node.op, ast.Sub):
            return lhs - rhs

        if isinstance(node.op, ast.Mult):
            return lhs * rhs

        if isinstance(node.op, ast.Div):
            return lhs / rhs

        if isinstance(node.op, ast.BitAnd):
            return lhs & rhs

        raise ASTFrontendError(f"unsupported binary operator: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return value

        raise ASTFrontendError(f"unsupported unary operator: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ASTFrontendError(
                "only simple comparisons are supported; "
                "when combining comparisons with &, wrap each comparison in parentheses"
            )

        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        op = node.ops[0]

        if isinstance(op, ast.Lt):
            return lhs < rhs

        if isinstance(op, ast.Is):
            return lhs is rhs

        raise ASTFrontendError(f"unsupported comparison operator: {type(op).__name__}")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        index = self._eval_index(node.slice)
        return value[index]

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        test = self.visit(node.test)
        if not isinstance(test, bool):
            raise ASTFrontendError("only constexpr conditions in IfExp are supported")
        return self.visit(node.body) if test else self.visit(node.orelse)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _eval_index(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_index(elt) for elt in node.elts)

        if isinstance(node, ast.Slice):
            if node.lower is None and node.upper is None and node.step is None:
                return slice(None)

            raise ASTFrontendError("only ':' slices are supported")

        if isinstance(node, ast.Constant) and node.value is None:
            return None

        return self.visit(node)

    def _eval_range_iterator(self, node: ast.AST) -> range:
        if not isinstance(node, ast.Call):
            raise ASTFrontendError("only for ... in range(...) is supported")

        iterator = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]

        if iterator is not range and iterator is not tl.static_range:
            raise ASTFrontendError(
                "only range(...) and tl.static_range(...) loops are supported"
            )

        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop = args
            step = 1
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise ASTFrontendError("range expects 1, 2, or 3 arguments")

        for name, value in (
            ("start", start),
            ("stop", stop),
            ("step", step),
        ):
            if type(value) is not int:
                raise ASTFrontendError(
                    "dynamic range bounds are not supported by AST frontend MVP; "
                    f"{name} is {value!r}"
                )

        return range(start, stop, step)


def trace(fn, signature, bound_args, runtime_params=None):
    if runtime_params is None:
        runtime_params = make_runtime_params(signature, bound_args)

    function_def = _find_function_def(fn)
    env = _make_symbolic_env(signature, bound_args, runtime_params)
    closure_vars = inspect.getclosurevars(fn)
    external_env = {
        **closure_vars.builtins,
        **closure_vars.globals,
        **closure_vars.nonlocals,
    }

    with Builder() as builder:
        ASTTracer(env, external_env).visit_stmt_list(function_def.body)

    return builder.ops, runtime_params

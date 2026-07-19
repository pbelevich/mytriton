from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Mapping
from typing import Any

from . import language as tl
from .trace import (
    Builder,
    Expression,
    ForRange,
    LoopCarry,
    LoopIndex,
    LoopResult,
    PointerType,
    Ptr,
    Value,
    is_constexpr_annotation,
    make_runtime_params,
    unwrap,
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


class AssignedNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self._seen: set[str] = set()

    def add_name(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.names.append(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.add_name(target.id)
            else:
                raise ASTFrontendError(
                    "loop MVP supports only assignment to simple names"
                )
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise ASTFrontendError(
                "loop MVP supports only annotated assignment to simple names"
            )
        self.add_name(node.target.id)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise ASTFrontendError(
                "loop MVP supports only augmented assignment to simple names"
            )
        self.add_name(node.target.id)
        self.generic_visit(node.value)


def _assigned_names(body: list[ast.stmt]) -> list[str]:
    collector = AssignedNameCollector()
    for stmt in body:
        collector.visit(stmt)
    return collector.names


class ASTTracer(ast.NodeVisitor):
    def __init__(
        self,
        env: dict[str, Any],
        external_env: Mapping[str, Any],
        capture_env: Mapping[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.external_env = external_env
        self._capture_ids = {
            id(unwrap(value))
            for value in (capture_env or {}).values()
            if isinstance(value, (Ptr, Value))
        }
        self._seen_capture_ids: set[int] = set()
        self.captures: list[Expression] = []

    def _record_capture(self, value: Any) -> None:
        if not isinstance(value, (Ptr, Value)):
            return

        expr = unwrap(value)
        key = id(expr)
        if key not in self._capture_ids or key in self._seen_capture_ids:
            return

        self._seen_capture_ids.add(key)
        self.captures.append(expr)

    def _propagate_capture(self, expr: Expression) -> None:
        key = id(expr)
        if key not in self._capture_ids or key in self._seen_capture_ids:
            return

        self._seen_capture_ids.add(key)
        self.captures.append(expr)

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

        start, stop, step = self._eval_range_parts(node.iter)

        # Fast path: all bounds are Python ints => frontend unroll.
        if type(start) is int and type(stop) is int and type(step) is int:
            name = node.target.id
            old_value = self.env.get(name)
            had_old_value = name in self.env

            for value in range(start, stop, step):
                self.env[name] = value
                self.visit_stmt_list(node.body)

            if had_old_value:
                self.env[name] = old_value
            else:
                del self.env[name]

            return

        # Runtime loop MVP.
        self._visit_runtime_for(node, start=start, stop=stop, step=step)

    def _visit_runtime_for(
        self,
        node: ast.For,
        *,
        start: Any,
        stop: Any,
        step: Any,
    ) -> None:
        target = node.target
        if not isinstance(target, ast.Name):
            raise ASTFrontendError(
                "only for loops with a simple induction variable are supported"
            )

        if type(step) is not int:
            raise ASTFrontendError("runtime for MVP requires integer constant step")

        if step <= 0:
            raise ASTFrontendError("runtime for MVP supports only positive step")

        assigned = _assigned_names(node.body)

        if target.id in assigned:
            raise ASTFrontendError(
                "assignment to runtime loop induction variable is not supported"
            )

        # Only variables that existed before the loop and are reassigned inside
        # become loop-carried variables. This captures `acc` in matmul.
        carried_names = tuple(name for name in assigned if name in self.env)

        loop_index = LoopIndex(target.id)
        carried_inputs = tuple(unwrap(self.env[name]) for name in carried_names)
        carried_args = tuple(
            LoopCarry(index=i, initial=initial)
            for i, initial in enumerate(carried_inputs)
        )

        body_env = dict(self.env)
        body_env[target.id] = Value(loop_index)

        for name, carried_arg in zip(carried_names, carried_args, strict=True):
            body_env[name] = Value(carried_arg)

        capture_env = {
            name: value
            for name, value in self.env.items()
            if name not in carried_names and name != target.id
        }

        with Builder() as body_builder:
            body_tracer = ASTTracer(
                body_env,
                self.external_env,
                capture_env=capture_env,
            )
            body_tracer.visit_stmt_list(node.body)

        for capture in body_tracer.captures:
            self._propagate_capture(capture)

        carried_outputs = tuple(unwrap(body_env[name]) for name in carried_names)

        loop = ForRange(
            index=loop_index,
            start=unwrap(start),
            stop=unwrap(stop),
            step=unwrap(step),
            captures=tuple(body_tracer.captures),
            body=body_builder.ops,
            carried_inputs=carried_inputs,
            carried_args=carried_args,
            carried_outputs=carried_outputs,
        )

        results = tuple(
            LoopResult(loop=loop, index=i) for i in range(len(carried_names))
        )
        loop.results = results

        Builder.current().ops.append(loop)

        # After the loop, carried variables refer to loop results.
        for name, result in zip(carried_names, results, strict=True):
            self.env[name] = Value(result)

        # Names created only inside the loop do not leak in this MVP.

    def visit_Return(self, node: ast.Return) -> None:
        raise ASTFrontendError("return statements are not supported in kernels")

    # ----------------------------
    # Expressions
    # ----------------------------

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.env:
                value = self.env[node.id]
                self._record_capture(value)
                return value

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

    def _eval_range_parts(self, node: ast.AST) -> tuple[Any, Any, Any]:
        if not isinstance(node, ast.Call):
            raise ASTFrontendError("only for ... in range(...) is supported")

        iterator = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]

        if iterator is not range and iterator is not tl.static_range:
            raise ASTFrontendError(
                "only range(...) and tl.static_range(...) loops are supported"
            )

        if len(args) == 1:
            return 0, args[0], 1

        if len(args) == 2:
            return args[0], args[1], 1

        if len(args) == 3:
            return args[0], args[1], args[2]

        raise ASTFrontendError("range expects 1, 2, or 3 arguments")


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

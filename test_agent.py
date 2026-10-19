"""早期天气 Agent 演示（已退役，不参与 Web 应用 / 不在 tests 目录内）。

原来的模块级 ``input()`` 会让导入即阻塞，``eval()`` 也有安全风险；这里改成
可手动运行的安全计算演示，供参考，不再作为运行时入口。
"""

from __future__ import annotations

import ast
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("只支持数字与 + - * / // % ** 运算")


def safe_calculate(expression: str) -> str:
    """安全计算算术表达式（不使用 eval）。"""
    try:
        tree = ast.parse(expression, mode="eval")
        return f"{expression} = {_evaluate(tree)}"
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return f"计算失败：{error}"


def main() -> None:
    print("早期演示已退役。当前应用入口：uv run python web_app.py")
    print(safe_calculate("(1 + 2) * 3"))


if __name__ == "__main__":
    main()

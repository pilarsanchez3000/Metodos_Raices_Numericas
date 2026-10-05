"""Implementación del método de falsa posición modificada (Illinois).

Este módulo convierte expresiones matemáticas en funciones evaluables,
valida un intervalo con cambio de signo y aplica iteraciones del método de
Illinois para aproximar una raíz real. También genera una tabla de iteraciones
y una gráfica de la función con la solución encontrada.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "normalize_expression",
    "parse_function",
    "print_iteration_table",
    "plot_illinois",
    "illinois",
    "main",
]


def normalize_expression(expression: str) -> str:
    """Normaliza una expresión matemática para aceptar notación común.

    La función elimina espacios, convierte notación matemática habitual a una
    sintaxis compatible con Python y añade multiplicación implícita donde hace
    falta, por ejemplo ``3x`` o ``2sin(x)``.

    Args:
        expression: Expresión escrita por el usuario, por ejemplo
            ``x^3 - 2x - 5`` o ``sin(x)``.

    Returns:
        La expresión transformada a una forma evaluable con ``eval``.

    Raises:
        TypeError: Si la entrada no es una cadena.
        ValueError: Si la expresión está vacía.
    """
    if not isinstance(expression, str):
        raise TypeError("La expresión debe ser una cadena de texto.")

    expr = expression.strip()
    if not expr:
        raise ValueError("La expresión no puede estar vacía.")

    expr = re.sub(r"\s+", "", expr)
    expr = expr.replace("^", "**").replace("π", "pi").replace("×", "*").replace("÷", "/")
    expr = expr.replace("−", "-")
    expr = re.sub(r"(?i)\bln\b", "log", expr)
    expr = re.sub(r"(?i)\bsen\b", "sin", expr)

    tokens = re.findall(
        r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]+|\*\*|[()+\-*/%,]",
        expr,
    )
    if "".join(tokens) != expr:
        raise ValueError("La expresión contiene caracteres no permitidos.")

    constants = {"x", "pi", "e", "tau"}
    functions = {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "log10",
        "log2",
        "sqrt",
        "abs",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
    }

    normalized: list[str] = []
    for index, token in enumerate(tokens):
        if index and token:
            previous = tokens[index - 1]
            previous_ends_value = (
                re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", previous)
                or previous in constants
                or previous == ")"
            )
            token_starts_value = (
                re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token)
                or token in constants
                or token in functions
                or token == "("
            )
            if previous_ends_value and token_starts_value and not (
                previous in functions and token == "("
            ):
                normalized.append("*")
        normalized.append(token)

    return "".join(normalized)


def _validate_expression(node: ast.AST, allowed_names: set[str]) -> None:
    """Valida que la expresión utilice solo operaciones y nombres permitidos.

    Args:
        node: Árbol sintáctico generado por ``ast.parse``.
        allowed_names: Nombres permitidos en la expresión, incluyendo funciones y
            constantes.

    Raises:
        ValueError: Si se detecta una operación o nombre no permitido.
    """
    permitted_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.UAdd,
        ast.USub,
    )

    for child in ast.walk(node):
        if not isinstance(child, permitted_nodes):
            raise ValueError("La expresión contiene una operación no permitida.")

        if isinstance(child, ast.Name) and child.id not in allowed_names:
            raise ValueError(f"Nombre o función no permitida: {child.id!r}.")

        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name):
                raise ValueError("Solo se permiten llamadas a funciones matemáticas.")
            if child.keywords:
                raise ValueError("No se permiten argumentos con nombre en las funciones.")


def parse_function(expression: str) -> Callable[[float], float]:
    """Convierte una expresión matemática en una función segura de ``x``.

    Args:
        expression: Expresión escrita por el usuario.

    Returns:
        Una función ``f(x)`` que evalúa la expresión en un valor dado.

    Raises:
        ValueError: Si la sintaxis o los nombres usados no son válidos.
    """
    expr = normalize_expression(expression)

    allowed = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sqrt": math.sqrt,
        "abs": abs,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    try:
        syntax_tree = ast.parse(expr, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Expresión inválida: {expression!r}.") from error

    _validate_expression(syntax_tree, set(allowed) | {"x"})
    compiled_expression = compile(syntax_tree, "<expresión>", "eval")

    def f(x: float) -> float:
        """Evalúa la expresión para el valor indicado de ``x``.

        Args:
            x: Valor en el que se evalúa la función.

        Returns:
            Resultado de ``f(x)``.

        Raises:
            ValueError: Si la función no puede evaluarse para ese valor.
        """
        try:
            result = eval(
                compiled_expression,
                {"__builtins__": {}},
                {"x": x, **allowed},
            )
            return float(result)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as error:
            raise ValueError(f"La función no está definida para x = {x}.") from error

    return f


def print_iteration_table(table: Iterable[dict[str, float | int]]) -> None:
    """Imprime la tabla de iteraciones del método de Illinois.

    Args:
        table: Secuencia de diccionarios con la información de cada iteración.
    """
    rows = list(table)
    if not rows:
        print("No hay iteraciones para mostrar.")
        return

    headers = ["Iter", "a", "b", "c", "f(c)", "Ancho"]
    formatted_rows = [
        [
            str(row["iter"]),
            f'{row["a"]:.8f}',
            f'{row["b"]:.8f}',
            f'{row["c"]:.8f}',
            f'{row["fc"]:.8e}',
            f'{row["error"]:.8e}',
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in formatted_rows)) for i in range(len(headers))]

    print("\nITERACIONES — MÉTODO DE ILLINOIS")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))

    for row in formatted_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def plot_illinois(
    expression: str,
    table: list[dict[str, float | int]],
    root: float,
    a_initial: float,
    b_initial: float,
) -> None:
    """Grafica la función, las aproximaciones y la raíz calculada.

    Args:
        expression: Expresión matemática a representar.
        table: Tabla de iteraciones generada por el método.
        root: Aproximación final de la raíz.
        a_initial: Límite inicial inferior del intervalo.
        b_initial: Límite inicial superior del intervalo.
    """
    f = parse_function(expression)
    margin = max(abs(b_initial - a_initial) * 0.25, 0.5)
    x_values = np.linspace(
        min(a_initial, b_initial) - margin,
        max(a_initial, b_initial) + margin,
        1000,
    )

    def safe_evaluate(x: float) -> float:
        """Evalúa la función con manejo de valores no finitos."""
        try:
            value = f(float(x))
            return value if math.isfinite(value) else np.nan
        except (ValueError, OverflowError):
            return np.nan

    y_values = [safe_evaluate(x) for x in x_values]
    approximations = [float(row["c"]) for row in table]
    function_values = [safe_evaluate(x) for x in approximations]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=f"f(x) = {expression}", color="blue")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvspan(
        a_initial,
        b_initial,
        color="orange",
        alpha=0.15,
        label="Intervalo inicial",
    )
    plt.scatter(
        approximations,
        function_values,
        color="darkorange",
        label="Aproximaciones",
        zorder=3,
    )

    for row in table:
        x_value = float(row["c"])
        plt.annotate(
            str(row["iter"]),
            (x_value, safe_evaluate(x_value)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.scatter(
        [root],
        [safe_evaluate(root)],
        color="red",
        s=75,
        label=f"Raíz ≈ {root:.8f}",
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Falsa posición modificada — método de Illinois")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("illinois_iteraciones.png", dpi=300)
    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close()


def illinois(
    expression: str,
    a: float,
    b: float,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> tuple[float, list[dict[str, float | int]]]:
    """Calcula una raíz con el método de falsa posición modificada Illinois.

    Args:
        expression: Función en términos de ``x``, por ejemplo ``x^3 - 2*x - 5``.
        a: Extremo izquierdo del intervalo inicial.
        b: Extremo derecho del intervalo inicial.
        tolerance: Tolerancia para el residuo o el ancho del intervalo.
        max_iterations: Número máximo de iteraciones permitidas.

    Returns:
        Una tupla con la raíz aproximada y la tabla de iteraciones realizadas.

    Raises:
        ValueError: Si los datos son inválidos o el intervalo no encierra una raíz.
        ArithmeticError: Si la aproximación siguiente no puede calcularse.
    """
    if tolerance <= 0:
        raise ValueError("La tolerancia debe ser mayor que cero.")
    if max_iterations < 1:
        raise ValueError("El máximo de iteraciones debe ser al menos 1.")

    if a > b:
        a, b = b, a

    f = parse_function(expression)
    fa, fb = f(a), f(b)

    if not math.isfinite(fa) or not math.isfinite(fb):
        raise ValueError("La función no está definida en uno de los extremos.")

    a_initial, b_initial = a, b

    if math.isclose(fa, 0.0, abs_tol=1e-12):
        root = a
        table = [{
            "iter": 0,
            "a": a,
            "b": b,
            "c": root,
            "fc": 0.0,
            "error": 0.0,
        }]
        print_iteration_table(table)
        print(f"\nRaíz exacta en un extremo: {root:.10f}")
        plot_illinois(expression, table, root, a_initial, b_initial)
        return root, table

    if math.isclose(fb, 0.0, abs_tol=1e-12):
        root = b
        table = [{
            "iter": 0,
            "a": a,
            "b": b,
            "c": root,
            "fc": 0.0,
            "error": 0.0,
        }]
        print_iteration_table(table)
        print(f"\nRaíz exacta en un extremo: {root:.10f}")
        plot_illinois(expression, table, root, a_initial, b_initial)
        return root, table

    if fa * fb > 0:
        raise ValueError(
            f"El intervalo [{a}, {b}] no presenta cambio de signo: "
            f"f(a)={fa:.8g}, f(b)={fb:.8g}."
        )

    wa, wb = fa, fb
    previous_replaced_endpoint: str | None = None
    table: list[dict[str, float | int]] = []
    root: float | None = None
    c: float | None = None

    for iteration in range(1, max_iterations + 1):
        denominator = wb - wa
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            raise ArithmeticError("No se pudo calcular la siguiente aproximación.")

        c = (a * wb - b * wa) / denominator
        fc = f(c)

        if not math.isfinite(fc):
            raise ValueError(f"La función no está definida para x = {c}.")

        interval_width = abs(b - a)
        table.append({
            "iter": iteration,
            "a": a,
            "b": b,
            "c": c,
            "fc": fc,
            "error": interval_width,
        })

        print(
            f"Iteración {iteration:3d}: "
            f"a={a:.10f}, b={b:.10f}, c={c:.10f}, "
            f"f(c)={fc:.8e}, ancho={interval_width:.8e}"
        )

        if abs(fc) <= tolerance or interval_width <= tolerance:
            root = c
            break

        if fa * fc < 0:
            b, fb, wb = c, fc, fc
            wa = wa * 0.5 if previous_replaced_endpoint == "b" else fa
            previous_replaced_endpoint = "b"
        else:
            a, fa, wa = c, fc, fc
            wb = wb * 0.5 if previous_replaced_endpoint == "a" else fb
            previous_replaced_endpoint = "a"
    else:
        if c is None:
            raise RuntimeError("No se pudo calcular ninguna aproximación.")
        root = c
        print(f"\nAviso: se alcanzó el máximo de {max_iterations} iteraciones.")

    if root is None:
        raise RuntimeError("No se pudo calcular una raíz aproximada.")

    print_iteration_table(table)
    print(f"\nRaíz aproximada: {root:.10f}")
    print(f"Iteraciones realizadas: {len(table)}")

    plot_illinois(expression, table, root, a_initial, b_initial)
    print("Gráfica guardada como 'illinois_iteraciones.png'.")

    return root, table


def main() -> None:
    """Solicita los datos al usuario y ejecuta el método de Illinois."""
    expression = input(
        "Ingrese f(x) [ej. x^3 - 2x - 5 o sen(x)]: "
    ).strip() or "x^3 - 2x - 5"

    a = float(input("Límite inferior a: "))
    b = float(input("Límite superior b: "))
    tolerance = float(input("Tolerancia [1e-5]: ") or "1e-5")

    illinois(expression, a, b, tolerance=tolerance)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, ArithmeticError) as error:
        print(f"\nError: {error}")
        raise SystemExit(1) from error
        
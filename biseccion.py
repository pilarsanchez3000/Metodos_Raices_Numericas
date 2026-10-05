"""Herramient para resolver ecuaciones mediante el método de bisección.

Este módulo permite convertir expresiones matemáticas a una función evaluable,
validar un intervalo de búsqueda y aplicar iteraciones de bisección para aproximar
una raíz real. También genera una gráfica de la función con la solución encontrada.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "normalize_expression",
    "parse_function",
    "plot_bisection",
    "print_iteration_table",
    "bisection",
    "main",
]

_FUNCTION_NAMES = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "exp",
    "log",
    "ln",
    "log10",
    "log2",
    "sqrt",
    "abs",
    "pow",
}


def normalize_expression(expression: str) -> str:
    """Normaliza una expresión matemática a una sintaxis válida para Python.

    La función elimina espacios, reemplaza operadores comunes como ``^`` por
    ``**``, convierte constantes y añade multiplicación implícita cuando sea
    necesario para que la expresión pueda evaluarse con ``eval`` de forma segura.

    Args:
        expression: Expresión matemática escrita por el usuario, por ejemplo
            ``x^2 - 3*x + 2``.

    Returns:
        La expresión transformada para ser evaluada con ``eval``.

    Raises:
        TypeError: Si la entrada no es una cadena.
        ValueError: Si la expresión está vacía.
    """
    if not isinstance(expression, str):
        raise TypeError("La expresión debe ser una cadena de texto.")

    expr = expression.strip()
    if not expr:
        raise ValueError("La expresión no puede estar vacía.")

    expr = expr.replace(" ", "")
    expr = expr.replace("^", "**")
    expr = expr.replace("π", "pi").replace("Π", "pi")
    expr = re.sub(r"(?i)\bln\b", "log", expr)
    expr = re.sub(r"(?<![A-Za-z0-9_])e(?![A-Za-z0-9_])", "e_const", expr)
    expr = re.sub(r"(?<![A-Za-z0-9_])pi(?![A-Za-z0-9_])", "pi", expr)

    result: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        next_ch = expr[i + 1] if i + 1 < len(expr) else ""

        if ch.isdigit() and next_ch.isalpha():
            result.append(ch)
            result.append("*")
            i += 1
            continue

        if ch == ")" and next_ch == "(":
            result.append(ch)
            result.append("*")
            i += 1
            continue

        if ch.isalpha() or ch == "_":
            start = i
            while i < len(expr) and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            identifier = expr[start:i]
            if i < len(expr) and expr[i] == "(" and identifier not in _FUNCTION_NAMES:
                result.append(identifier)
                result.append("*")
                continue
            result.append(identifier)
            continue

        if ch == "(" and result and result[-1].isalpha():
            result.append(ch)
            i += 1
            continue

        result.append(ch)
        i += 1

    expr = "".join(result)
    return expr


def parse_function(expression: str) -> Callable[[float], float]:
    """Construye una función evaluable a partir de una expresión matemática.

    Args:
        expression: Cadena con la función a evaluar, por ejemplo ``sin(x) + x``.

    Returns:
        Una función que acepta un valor real ``x`` y devuelve ``f(x)``.

    Raises:
        ValueError: Si la expresión es inválida o contiene elementos no reconocidos.
    """
    expr = normalize_expression(expression)

    allowed = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "arcsin": math.asin,
        "arccos": math.acos,
        "arctan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        "exp": math.exp,
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sqrt": math.sqrt,
        "abs": abs,
        "pi": math.pi,
        "e_const": math.e,
        "e": math.e,
        "pow": pow,
    }

    def f(x: float) -> float:
        """Evalúa la función matemática en un valor específico de ``x``.

        Args:
            x: Valor independiente en el que se evalúa la función.

        Returns:
            Resultado de evaluar ``f(x)`` en el punto indicado. Si la expresión
            genera un valor no finito, devolvera ``nan``.
        """
        try:
            compiled = compile(expr, "<math-expression>", "eval")
            return eval(compiled, {"__builtins__": {}}, {"x": x, **allowed})
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("nan")
        except SyntaxError as exc:
            raise ValueError(
                f"Expresión inválida: '{expression}'. Revisa el formato matemático."
            ) from exc
        except NameError as exc:
            raise ValueError(
                f"La función '{expression}' contiene elementos no reconocidos."
            ) from exc

    return f


def plot_bisection(
    expression: str,
    iterations: list[dict[str, float | int]],
    root: float,
    a_init: float,
    b_init: float,
) -> None:
    """Genera la gráfica de la función y la raíz aproximada.

    Args:
        expression: Expresión matemática a representar.
        iterations: Lista con las iteraciones realizadas por la bisección.
        root: Aproximación de la raíz encontrada.
        a_init: Límite inferior inicial del intervalo.
        b_init: Límite superior inicial del intervalo.

    Raises:
        ValueError: Si no se proporcionan iteraciones válidas.
    """
    if not iterations:
        raise ValueError("La lista de iteraciones no puede estar vacía.")

    f = parse_function(expression)
    margin = max(abs(b_init - a_init) * 0.5, 1.0)
    x_min = min(a_init, b_init) - margin
    x_max = max(a_init, b_init) + margin

    x_vals = np.linspace(x_min, x_max, 1000)
    y_vals = np.array([f(xi) for xi in x_vals])

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {expression}", color="blue", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.3)
    plt.axvspan(
        a_init,
        b_init,
        color="orange",
        alpha=0.15,
        label=f"Intervalo inicial [{a_init}, {b_init}]",
    )

    m_points = [item["m"] for item in iterations]
    fm_points = [item["f(m)"] for item in iterations]
    if len(m_points) > 1:
        plt.scatter(
            m_points[:-1],
            fm_points[:-1],
            color="orange",
            s=30,
            alpha=0.6,
            label="Iteraciones (m)",
        )

    plt.scatter([root], [f(root)], color="red", s=90, zorder=5, label=f"Raíz = {root:.8f}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Método de Bisección: f(x) = {expression} ({len(iterations)} iteraciones)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("biseccion_iteraciones.png", dpi=300)
    print("\nGráfica guardada exitosamente como 'biseccion_iteraciones.png'")
    try:
        plt.show()
    except Exception:
        pass


def print_iteration_table(table: Iterable[dict[str, float | int]]) -> None:
    """Imprime una tabla formateada con las iteraciones del método.

    Args:
        table: Secuencia de diccionarios con la información de cada iteración.
    """
    rows = list(table)
    if not rows:
        print("No hay iteraciones para mostrar.")
        return

    headers = ["Iter", "a", "b", "m", "f(m)", "Error"]
    formatted_rows = [
        [
            r["iter"],
            f"{r['a']:.8f}",
            f"{r['b']:.8f}",
            f"{r['m']:.8f}",
            f"{r['f(m)']:.8f}",
            f"{r['error']:.8e}",
        ]
        for r in rows
    ]
    widths = [
        max(len(str(h)), max(len(str(r[i])) for r in formatted_rows))
        for i, h in enumerate(headers)
    ]

    print("\nTABLA DE ITERACIONES")
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in formatted_rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def bisection(
    expression: str,
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> tuple[float, list[dict[str, float | int]]]:
    """Busca una raíz real de una función usando el método de bisección.

    Args:
        expression: Expresión matemática a evaluar.
        a: Límite inferior del intervalo.
        b: Límite superior del intervalo.
        tol: Tolerancia aceptada para la convergencia.
        max_iter: Número máximo de iteraciones permitidas.

    Returns:
        Una tupla con la raíz aproximada y la tabla de iteraciones realizadas.

    Raises:
        ValueError: Si el intervalo no es válido, la tolerancia es inválida o no
            existe cambio de signo.
    """
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Los límites del intervalo deben ser valores finitos.")
    if a == b:
        raise ValueError("El intervalo debe tener dos valores distintos.")
    if tol <= 0:
        raise ValueError("La tolerancia debe ser un número positivo.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser un entero mayor que cero.")

    f = parse_function(expression)
    fa = f(a)
    fb = f(b)

    if not math.isfinite(fa) or not math.isfinite(fb):
        raise ValueError("La función no es evaluable en alguno de los extremos del intervalo.")

    if abs(fa) < 1e-12:
        print(f"\n¡Atención! El límite inferior a = {a} ya es una raíz exacta de la función (f({a}) = 0).")
        print("Total de iteraciones realizadas: 0")
        table = [{"iter": 0, "a": a, "b": b, "m": a, "f(m)": 0.0, "error": 0.0}]
        plot_bisection(expression, table, a, a, b)
        return a, table

    if abs(fb) < 1e-12:
        print(f"\n¡Atención! El límite superior b = {b} ya es una raíz exacta de la función (f({b}) = 0).")
        print("Total de iteraciones realizadas: 0")
        table = [{"iter": 0, "a": a, "b": b, "m": b, "f(m)": 0.0, "error": 0.0}]
        plot_bisection(expression, table, b, a, b)
        return b, table

    if fa * fb > 0:
        raise ValueError(
            f"No hay cambio de signo en el intervalo [{a}, {b}]: f({a}) = {fa:.4f} y f({b}) = {fb:.4f}.\n"
            "El método de Bisección requiere que f(a) y f(b) tengan signos opuestos."
        )

    table: list[dict[str, float | int]] = []
    root: float | None = None
    a_init, b_init = a, b

    for iteration in range(1, max_iter + 1):
        m = (a + b) / 2
        fm = f(m)
        error = abs(b - a) / 2

        table.append(
            {
                "iter": iteration,
                "a": a,
                "b": b,
                "m": m,
                "f(m)": fm,
                "error": error,
            }
        )

        if abs(fm) < tol or error < tol:
            root = m
            break

        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    else:
        root = (a + b) / 2

    if root is None:
        raise RuntimeError("No se pudo calcular una raíz aproximada.")

    print_iteration_table(table)
    print(f"\nRaíz aproximada: {root:.10f}")
    print(f"Total de iteraciones realizadas: {len(table)}")
    plot_bisection(expression, table, root, a_init, b_init)
    return root, table


def main() -> None:
    """Solicita la expresión y el intervalo al usuario y ejecuta la bisección."""
    try:
        expr_in = input(
            "Ingrese f(x) [ej. x^3 - 7x + 6, exp(-x) - sin(x), ln(x) - 2]: "
        ).strip()
        if not expr_in:
            expr_in = "x**3 - 7*x + 6"

        a_in = float(input("Límite inferior (a): "))
        b_in = float(input("Límite superior (b): "))

        bisection(expr_in, a_in, b_in)
    except ValueError as err:
        print(f"\nError: {err}")
    except Exception as err:
        print(f"\nError inesperado: {err}")


if __name__ == "__main__":
    main()

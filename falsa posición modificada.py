import math
import re
import sys
import matplotlib.pyplot as plt
import numpy as np


def normalize_expression(expression: str) -> str:
    """Convierte notación matemática común (ej. '7x' -> '7*x', '^' -> '**') a Python válido."""
    expr = expression.strip().replace(" ", "")
    expr = expr.replace("^", "**")
    expr = expr.replace("π", "pi")
    expr = expr.replace("ln", "log")  # ayuda con logaritmo natural
    expr = re.sub(r'(?<=[0-9x\)])(?=[0-9x(a-zA-Z])', '*', expr)
    return expr


def parse_function(expression: str):
    """Crea una función evaluable a partir de la expresión del usuario."""
    expr = normalize_expression(expression)
    allowed = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
        "pow": pow,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
    }

    def f(x):
        try:
            return eval(expr, {"__builtins__": {}}, {"x": x, **allowed})
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("nan")
        except SyntaxError as exc:
            raise ValueError(
                f"Expresión inválida: '{expression}'. Usa un formato matemático válido."
            ) from exc
        except NameError as exc:
            raise ValueError(
                f"La función '{expression}' no es válida. Revisa los nombres de las variables/funciones."
            ) from exc

    return f


def print_iteration_table(table: list):
    """Imprime la tabla de iteraciones en formato legible."""
    if not table:
        print("No hay iteraciones para mostrar.")
        return

    headers = ["Iter", "a", "b", "m", "f(m)", "Error"]
    rows = [
        [
            r["iter"],
            f"{r['a']:.8f}",
            f"{r['b']:.8f}",
            f"{r['m']:.8f}",
            f"{r['f(m)']:.8f}",
            f"{r['error']:.8e}",
        ]
        for r in table
    ]

    widths = [max(len(str(h)), max(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]

    print("\nTABLA DE ITERACIONES")
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def plot_bisection(expression: str, iterations: list, root: float, a_init: float, b_init: float):
    """Genera la gráfica de la función, las iteraciones y la raíz hallada."""
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
    plt.axvspan(a_init, b_init, color="orange", alpha=0.15, label=f"Intervalo inicial [{a_init}, {b_init}]")

    m_points = [item["m"] for item in iterations]
    fm_points = [item["f(m)"] for item in iterations]

    if m_points:
        plt.scatter(m_points, fm_points, color="orange", s=35, zorder=3, label="Iteraciones")
        for item in iterations:
            plt.annotate(
                str(item["iter"]),
                (item["m"], item["f(m)"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="darkorange",
            )

    plt.scatter([root], [f(root)], color="red", s=90, zorder=5, label=f"Raíz = {root:.8f}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Método de Bisección: f(x) = {expression}")
    plt.legend()
    plt.tight_layout()

    plt.savefig("biseccion_iteraciones.png", dpi=300)
    print("\nGráfica guardada exitosamente como 'biseccion_iteraciones.png'")
    try:
        plt.show()
    except Exception:
        pass


def bisection(expression: str, a: float, b: float, tol: float = 1e-5, max_iter: int = 1000):
    """Ejecuta la bisección mostrando cada iteración en consola."""
    f = parse_function(expression)
    fa = f(a)
    fb = f(b)

    if abs(fa) < 1e-12:
        print(f"\n¡Atención! El límite inferior a = {a} ya es una raíz exacta de la función (f({a}) = 0).")
        table = [{"iter": 0, "a": a, "b": b, "m": a, "f(m)": 0.0, "error": 0.0}]
        print_iteration_table(table)
        plot_bisection(expression, table, a, a, b)
        return a, table

    if abs(fb) < 1e-12:
        print(f"\n¡Atención! El límite superior b = {b} ya es una raíz exacta de la función (f({b}) = 0).")
        table = [{"iter": 0, "a": a, "b": b, "m": b, "f(m)": 0.0, "error": 0.0}]
        print_iteration_table(table)
        plot_bisection(expression, table, b, a, b)
        return b, table

    if fa * fb > 0:
        raise ValueError(
            f"No hay cambio de signo en el intervalo [{a}, {b}]: f({a}) = {fa:.4f} y f({b}) = {fb:.4f}.\n"
            "El método de Bisección requiere que f(a) y f(b) tengan signos opuestos."
        )

    table = []
    a_init, b_init = a, b

    for iteration in range(1, max_iter + 1):
        m = (a + b) / 2
        fm = f(m)
        error = abs(b - a) / 2

        table.append({
            "iter": iteration,
            "a": a,
            "b": b,
            "m": m,
            "f(m)": fm,
            "error": error,
        })

        print(
            f"Iter {iteration:2d}: "
            f"a={a:.8f}, b={b:.8f}, m={m:.8f}, f(m)={fm:.8f}, error={error:.8e}"
        )

        if abs(fm) < tol or error < tol:
            root = m
            print(f"\nCriterio de convergencia alcanzado en la iteración {iteration}.")
            break

        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    else:
        root = (a + b) / 2

    print_iteration_table(table)
    print(f"\nRaíz aproximada: {root:.10f}")

    plot_bisection(expression, table, root, a_init, b_init)
    return root, table


if __name__ == "__main__":
    try:
        expr_in = input("Ingrese f(x) [ej. x**3 - 7*x + 6 o x^3 - 7x + 6]: ").strip()
        if not expr_in:
            expr_in = "x**3 - 7*x + 6"

        a_in = float(input("Límite inferior (a): "))
        b_in = float(input("Límite superior (b): "))

        bisection(expr_in, a_in, b_in)

    except Exception as err:
        print(f"\nError: {err}")
        
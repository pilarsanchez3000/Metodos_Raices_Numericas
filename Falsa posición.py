import math
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def normalize_expression(expression: str) -> str:
    expr = expression.strip().replace(" ", "")
    expr = expr.replace("^", "**")
    expr = expr.replace("π", "pi")
    expr = expr.replace("sen", "sin")
    expr = re.sub(r'(?<=[0-9x\)])(?=[0-9x(])', '*', expr)
    return expr


def parse_function(expression: str):
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
        except SyntaxError as exc:
            raise ValueError(
                f"Expresión inválida: '{expression}'. Usa formato válido, por ejemplo: x**3 - 7*x + 6 o sin(x)"
            ) from exc
        except NameError as exc:
            raise ValueError(
                f"La función '{expression}' no es válida. Usa solo x y funciones como sin, cos, exp, log, sqrt."
            ) from exc

    return f


def find_bracket(expression: str, start: float = -50.0, end: float = 50.0, step: float = 0.25):
    f = parse_function(expression)

    for left, right in [(-50.0, 50.0), (-100.0, 100.0), (-200.0, 200.0)]:
        xs = []
        ys = []
        x = left
        while x <= right:
            xs.append(x)
            ys.append(f(x))
            x += step

        for i in range(len(xs) - 1):
            if abs(ys[i]) < 1e-12:
                left_candidate = xs[i] - step
                right_candidate = xs[i] + step
                if left_candidate >= left and right_candidate <= right and f(left_candidate) * f(right_candidate) < 0:
                    return left_candidate, right_candidate
                return xs[i] - 0.5, xs[i] + 0.5

            if ys[i] * ys[i + 1] < 0:
                return xs[i], xs[i + 1]

    raise ValueError("No se encontró un intervalo válido con cambio de signo. Prueba otra función o cambia el rango de búsqueda.")


def print_table(table):
    headers = ["Iter", "a", "b", "m", "f(m)", "Error"]
    rows = []
    for iteracion, a, b, m, fm, error in table:
        rows.append([
            iteracion,
            f"{a:.10f}",
            f"{b:.10f}",
            f"{m:.10f}",
            f"{fm:.10f}",
            f"{error:.10e}",
        ])

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    print("\nTabla de iteraciones")
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def plot_method(expression, table):
    f = parse_function(expression)

    xs = []
    for _, a, b, m, fm, _ in table:
        xs.extend([a, b, m])

    
    x = np.linspace(-10, 8, 1000)
    y = np.array([f(xi) for xi in x])

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f"f(x) = {expression}", color="blue", linewidth=2)
    plt.axhline(0, color="black", linewidth=1.0, linestyle="--")

    for _, a, b, m, fm, _ in table:
        plt.plot([a, b], [0, 0], color="gray", alpha=0.55, linewidth=1.2)
        plt.scatter([a, b], [0, 0], color="orange", s=25)
        plt.scatter([m], [fm], color="green", s=30, zorder=3)

    raiz = table[-1][3]
    plt.scatter([raiz], [0], color="red", s=60, zorder=5, label=f"Raíz ≈ {raiz:.8f}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de falsa posición")
    plt.legend()
    plt.tight_layout()
    plt.savefig("falsa_posicion_iteraciones.png")
    plt.show()
    plt.close()


def false_position(expression: str, a: float, b: float, tol: float = 1e-5, max_iter: int = 1000):
    f = parse_function(expression)
    fa = f(a)
    fb = f(b)

    if fa == 0:
        root = a
        table = [(0, a, b, a, 0.0, 0.0)]
        print_table(table)
        print(f"\nRaíz: {root:.10f}")
        print("Número de iteraciones necesarias: 0")
        plot_method(expression, table)
        print("Gráfica guardada como falsa_posicion_iteraciones.png")
        return root, table

    if fb == 0:
        root = b
        table = [(0, a, b, b, 0.0, 0.0)]
        print_table(table)
        print(f"\nRaíz: {root:.10f}")
        print("Número de iteraciones necesarias: 0")
        plot_method(expression, table)
        print("Gráfica guardada como falsa_posicion_iteraciones.png")
        return root, table

    if fa * fb > 0:
        raise ValueError("La función no cambia de signo en [a, b].")

    table = []
    root = None
    iter_count = 0

    for iteration in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(b - a)
        iter_count += 1

        table.append((iteration, a, b, c, fc, error))
        print(f"Iteración {iteration}: a={a:.10f}, b={b:.10f}, c={c:.10f}, f(c)={fc:.10f}, error={error:.10e}")

        if abs(fc) < tol or error < tol:
            root = c
            break

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    else:
        root = (a * fb - b * fa) / (fb - fa)

    print_table(table)
    print(f"\nRaíz aproximada: {root:.10f}")
    print(f"Tolerancia: {tol:.0e}")
    print(f"Número de iteraciones necesarias para llegar a la tolerancia: {iter_count}")
    plot_method(expression, table)
    print("Gráfica guardada como falsa_posicion_iteraciones.png")

    return root, table


if __name__ == "__main__":
    try:
        expression = input("Ingrese la función f(x): ").strip()
        if not expression:
            expression = "x**3 - 2*x - 5"

        f = parse_function(expression)
        print("\nIngrese el intervalo [a, b] a evaluar.")
        a = float(input("a = "))
        b = float(input("b = "))

        if f(a) * f(b) > 0:
            raise ValueError("El intervalo ingresado no cumple con el criterio de cambio de signo. Ingresa otro intervalo.")

        print(f"\nResolviendo f(x) = {expression} en [{a}, {b}] usando método de falsa posición")
        root, table = false_position(expression, a, b, tol=1e-5)

        print(f"\nValor de la raíz aproximada: {root:.10f}")
        print(f"Número de iteraciones hasta la tolerancia 10^-5: {len(table)}")
        print("La gráfica se guardó en falsa_posicion_iteraciones.png")
        print("La gráfica también se muestra en pantalla.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
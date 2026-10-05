import math
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def normalize_expression(expression: str) -> str:
    """Normaliza la expresión matemática ingresada por el usuario."""
    expr = expression.strip().replace(" ", "")
    expr = expr.replace("^", "**")
    expr = expr.replace("π", "pi")
    expr = expr.replace("sen", "sin")

    # Corregir precedencia de raíz cúbica: x**1/3 o x**(1/3) -> cbrt(x)
    expr = re.sub(r'([a-zA-Z0-9_\.\(\)]+)\*\*(?:1/3|1\.0/3\.0|\(1/3\))', r'cbrt(\1)', expr)
    
    # Insertar multiplicación implícita (ej. 3x -> 3*x, 2(x+1) -> 2*(x+1))
    expr = re.sub(r'(?<=[0-9x\)])(?=[0-9x(a-zA-Z])', '*', expr)
    return expr


def real_cbrt(x):
    """Calcula la raíz cúbica real aceptando escalares y arreglos NumPy."""
    if isinstance(x, np.ndarray):
        return np.cbrt(x)
    return math.copysign(abs(x) ** (1 / 3), x) if x != 0 else 0.0


def parse_function(expression: str):
    """Convierte la cadena de texto en una función ejecutable."""
    expr = normalize_expression(expression)

    allowed_scalar = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "cbrt": real_cbrt,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
        "pow": pow,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
    }

    allowed_array = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "cbrt": np.cbrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
    }

    def f(x):
        try:
            if isinstance(x, np.ndarray):
                return eval(expr, {"__builtins__": {}}, {"x": x, **allowed_array})
            return eval(expr, {"__builtins__": {}}, {"x": x, **allowed_scalar})
        except SyntaxError as exc:
            raise ValueError(f"Expresión inválida: '{expression}'. Revisa la sintaxis.") from exc
        except NameError as exc:
            raise ValueError(f"La expresión contiene símbolos no válidos. Usa 'x'.") from exc

    return f


def numerical_derivative(f, x: float) -> float:
    """Calcula la derivada numérica mediante diferencias finitas centradas."""
    step = math.ulp(1.0) ** (1 / 3) * max(1.0, abs(x))
    try:
        derivative = (f(x + step) - f(x - step)) / (2 * step)
    except Exception:
        try:
            derivative = (f(x + step) - f(x)) / step
        except Exception:
            try:
                derivative = (f(x) - f(x - step)) / step
            except Exception as exc:
                raise ValueError("No se pudo evaluar la derivada en el punto actual.") from exc

    if not math.isfinite(derivative):
        raise ValueError("La derivada calculada resulta infinita o indefinida.")
    return derivative


def print_table(table):
    """Imprime los resultados de las iteraciones en una tabla formateada."""
    headers = ["Iter", "x_n", "f(x_n)", "f'(x_n)", "x_(n+1)", "Error"]
    rows = []
    for iteracion, x, fx, derivative, next_x, next_fx, error in table:
        rows.append([
            iteracion,
            f"{x:.10g}",
            f"{fx:.6e}",
            f"{derivative:.6e}",
            f"{next_x:.10g}",
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


def plot_method(expression: str, table: list, status_text: str):
    """Genera la gráfica visualizando las iteraciones y el estado del método."""
    f = parse_function(expression)
    iteration_values = [value for row in table for value in (row[1], row[4]) if math.isfinite(value)]

    if iteration_values:
        lower, upper = min(iteration_values), max(iteration_values)
        center = (lower + upper) / 2.0
        half_range = max((upper - lower) / 2.0, 1.0) * 1.5
        x_min, x_max = center - half_range, center + half_range
    else:
        x_min, x_max = -10, 10

    x = np.linspace(x_min, x_max, 1000)

    with np.errstate(all="ignore"):
        try:
            y = f(x)
            if not isinstance(y, np.ndarray):
                y = np.array([f(xi) for xi in x])
        except Exception:
            y = np.array([f(xi) if math.isfinite(f(xi)) else np.nan for xi in x])

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f"f(x) = {expression}", color="royalblue", linewidth=2)
    plt.axhline(0, color="black", linewidth=1.0, linestyle="--")

    for _, current_x, current_fx, derivative, next_x, _, _ in table:
        plt.scatter([current_x], [current_fx], color="darkorange", s=30, zorder=3)
        if math.isfinite(derivative) and abs(derivative) > 1e-12 and math.isfinite(next_x):
            x_tangent = np.array([current_x, next_x])
            y_tangent = current_fx + derivative * (x_tangent - current_x)
            plt.plot(x_tangent, y_tangent, color="orange", linestyle=":", alpha=0.7)

    last_x = table[-1][4]
    if math.isfinite(last_x):
        plt.scatter([last_x], [0], color="crimson", s=80, zorder=5, label=f"Último punto ≈ {last_x:.6g}")

    plt.xlim(x_min, x_max)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Método de Newton-Raphson\n[{status_text}]")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("newton_raphson_iteraciones.png", dpi=150)
    plt.show()
    plt.close()


def newton_raphson(expression: str, x0: float, tol: float = 1e-6, max_iter: int = 50):
    """Ejecuta el método indicando claramente si converge o diverge y la cantidad de iteraciones."""
    f = parse_function(expression)
    
    if not math.isfinite(x0):
        raise ValueError("El valor inicial x0 debe ser un número finito.")

    x = x0
    table = []
    fx = f(x)
    
    if not math.isfinite(fx):
        raise ValueError("La función no se puede evaluar en el valor inicial proporcionado.")

    # Si x0 ya es raíz
    if abs(fx) <= tol:
        table.append((0, x, fx, 0.0, x, fx, 0.0))
        print_table(table)
        print("\n========================================")
        print("ESTADO: CONVERGE (El valor inicial x0 ya es una raíz exacto o dentro de tolerancia)")
        print("Número total de iteraciones: 0")
        print(f"Raíz encontrada: {x:.10g}")
        print("========================================")
        plot_method(expression, table, "CONVERGE - Raíz encontrada en x0")
        return x, table

    converged = False
    reason = "Alcanzado el número máximo de iteraciones sin convergencia"

    for iteration in range(1, max_iter + 1):
        try:
            derivative = numerical_derivative(f, x)
        except Exception as err:
            reason = f"Error al calcular la derivada en iteración {iteration}: {err}"
            break

        if abs(derivative) < 1e-14:
            reason = f"La derivada es casi cero (f'(x) ≈ 0) en x = {x:.10g}"
            break

        next_x = x - fx / derivative
        
        # Detección de divergencia explícita
        if abs(next_x) > 1e12 or (iteration > 3 and abs(next_x) > abs(x) * 2.0 and abs(fx) > tol):
            reason = f"DIVERGENCIA DETECTADA: El valor de x creció desproporcionadamente a {next_x:.4e}"
            break

        try:
            next_fx = f(next_x)
        except Exception:
            next_fx = float("nan")

        error = abs(next_x - x)
        table.append((iteration, x, fx, derivative, next_x, next_fx, error))

        x, fx = next_x, next_fx
        
        if math.isfinite(fx) and abs(fx) <= tol and error <= tol * max(1.0, abs(x)):
            converged = True
            reason = "Criterio de tolerancia alcanzado con éxito"
            break

    total_iterations = len(table)

    if total_iterations > 0:
        print_table(table)

    print("\n========================================")
    if converged:
        print("ESTADO: CONVERGE")
        print(f"Número total de iteraciones: {total_iterations}")
        print(f"Raíz encontrada: {x:.10g}")
        status_text = f"CONVERGE en {total_iterations} iteraciones | Raíz ≈ {x:.6g}"
    else:
        print("ESTADO: DIVERGE (No se encontró raíz)")
        print(f"Número total de iteraciones realizadas: {total_iterations}")
        print(f"Causa / Razón: {reason}")
        status_text = f"DIVERGE ({total_iterations} iteraciones) - {reason}"
    print("========================================")

    if total_iterations > 0:
        plot_method(expression, table, status_text)
        print("Gráfica guardada como 'newton_raphson_iteraciones.png'.")

    return x if converged else None, table


if __name__ == "__main__":
    try:
        expr_input = input("Ingrese la función f(x) (ej. x**(1/3) - 2, x**3 - 8): ")
        x0_input = float(input("Ingrese el valor inicial x0: "))
        tol_input = input("Ingrese tolerancia (Enter para 1e-6 por defecto): ").strip()
        
        tol = float(tol_input) if tol_input else 1e-6

        print(f"\nEjecutando Newton-Raphson para f(x) = {expr_input} con x0 = {x0_input}")
        newton_raphson(expr_input, x0_input, tol=tol)

    except Exception as e:
        print(f"\nError de ejecución: {e}")
        sys.exit(1)
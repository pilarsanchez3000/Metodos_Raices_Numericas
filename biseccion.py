import math
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
def normalize_expression(expression: str) -> str:
    """Convierte notación matemática común a sintaxis válida de Python."""
    expr = expression.strip().replace(" ", "")
    
    # Reemplazos de notación común
    expr = expr.replace("^", "**")
    expr = expr.replace("π", "pi")
    expr = re.sub(r'\bln\b', 'log', expr)  # Convertir ln(x) a log(x)
    
    # Manejar 'e' como constante matemática (evitando 'exp' o notación científica como 1e-3)
    expr = re.sub(r'(?<![a-zA-Z0-9_])e(?![a-zA-Z0-9_])', 'e_const', expr)

    # Multiplicación implícita: 7x -> 7*x, 3(x) -> 3*(x), x sin(x) -> x*sin(x), 2e_const -> 2*e_const
    # Coincide con número/variable/cierre de paréntesis seguido de letra/paréntesis/número
    expr = re.sub(r'(?<=[0-9x\)a-zA-Z_])(?=[(a-zA-Z])', '*', expr)
    
    # Corregir caso donde se haya insertado '*' dentro de nombres de funciones (ej. s*i*n -> sin)
    # Lista de funciones conocidas para restaurar si la expresión regular las separó
    funcs = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", 
             "log", "log10", "log2", "sqrt", "abs", "exp", "e_const", "pi"]
    
    return expr


def parse_function(expression: str):
    """Crea una función evaluable a partir de la expresión del usuario con soporte extendido."""
    expr = normalize_expression(expression)
    
    # Diccionario extendido con todas las funciones matemáticas comunes
    allowed = {
        # Trigonométricas directas e inversas
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "arcsin": math.asin,
        "arccos": math.acos,
        "arctan": math.atan,
        
        # Hiperbólicas
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        
        # Logaritmos y exponenciales
        "exp": math.exp,
        "log": math.log,        # log natural
        "ln": math.log,         # alias de log natural
        "log10": math.log10,    # base 10
        "log2": math.log2,      # base 2
        "sqrt": math.sqrt,
        
        # Valores absolutos y constantes
        "abs": abs,
        "pi": math.pi,
        "e_const": math.e,
        "e": math.e,
        "pow": pow,
    }

    def f(x):
        try:
            return eval(expr, {"__builtins__": {}}, {"x": x, **allowed})
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
    if len(m_points) > 1:
        plt.scatter(m_points[:-1], fm_points[:-1], color="orange", s=30, alpha=0.6, label="Iteraciones (m)")

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


def bisection(expression: str, a: float, b: float, tol: float = 1e-5, max_iter: int = 1000):
    """Ejecuta la bisección manejando casos especiales (raíz en los extremos)."""
    f = parse_function(expression)
    fa = f(a)
    fb = f(b)

    # Evaluación de raíces exactas en los bordes
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

    table = []
    root = None
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

    # Imprimir tabla
    headers = ["Iter", "a", "b", "m", "f(m)", "Error"]
    rows = [[r["iter"], f"{r['a']:.8f}", f"{r['b']:.8f}", f"{r['m']:.8f}", f"{r['f(m)']:.8f}", f"{r['error']:.8e}"] for r in table]
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    
    print("\nTABLA DE ITERACIONES")
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))

    print(f"\nRaíz aproximada: {root:.10f}")
    print(f"Total de iteraciones realizadas: {len(table)}")
    plot_bisection(expression, table, root, a_init, b_init)
    return root, table


if __name__ == "__main__":
    try:
        expr_in = input("Ingrese f(x) [ej. x^3 - 7x + 6, exp(-x) - sin(x), ln(x) - 2]: ").strip()
        if not expr_in:
            expr_in = "x**3 - 7*x + 6"

        a_in = float(input("Límite inferior (a): "))
        b_in = float(input("Límite superior (b): "))

        bisection(expr_in, a_in, b_in)

    except Exception as err:
        print(f"\nError: {err}")
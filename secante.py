import ast
import math
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def normalize_expression(expression: str) -> str:
    expr = re.sub(r"\s+", "", expression).replace("π", "pi").replace("×", "*").replace("÷", "/")
    expr = expr.replace("−", "-").replace("^", "**")
    expr = re.sub(r"(?<![A-Za-z_])sen(?=\s*\()", "sin", expr)
    expr = re.sub(r"(?<![A-Za-z_])ln(?=\s*\()", "log", expr)
    tokens = re.findall(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|\*\*|[()+\-*/%,]", expr)
    if "".join(tokens) != expr:
        raise ValueError("La expresión contiene caracteres no permitidos.")

    constants = {"x", "pi", "e", "tau"}
    functions = {
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh", "exp", "log", "log2", "log10", "sqrt", "cbrt",
        "abs", "floor", "ceil", "hypot", "pow",
    }
    normalized = []
    for index, token in enumerate(tokens):
        if index:
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
            if previous_ends_value and token_starts_value:
                normalized.append("*")
        normalized.append(token)
    return "".join(normalized)


def parse_function(expression: str):
    expr = normalize_expression(expression)
    allowed = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        "exp": math.exp,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "cbrt": np.cbrt,
        "abs": abs,
        "floor": math.floor,
        "ceil": math.ceil,
        "hypot": math.hypot,
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "pow": pow,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
    }
    function_names = set(allowed) - {"pi", "e", "tau"}
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Expresión matemática inválida: {expression}") from exc

    def validate(node):
        if isinstance(node, ast.Expression):
            validate(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Solo se permiten constantes numéricas.")
        elif isinstance(node, ast.Name):
            if node.id not in {"x", "pi", "e", "tau"}:
                raise ValueError(f"Nombre o función no reconocida: {node.id}")
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
        ):
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            validate(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in function_names or node.keywords:
                raise ValueError(f"Función no permitida: {node.func.id}")
            for argument in node.args:
                validate(argument)
        else:
            raise ValueError("Usa operaciones aritméticas y funciones matemáticas permitidas.")

    validate(tree)

    def f(x):
        try:
            return eval(compile(tree, "<funcion_ingresada>", "eval"), {"__builtins__": {}}, {"x": x, **allowed})
        except (NameError, TypeError) as exc:
            raise ValueError(f"No se pudo evaluar la función ingresada: {exc}") from exc

    return f


def print_table(table):
    headers = ["Iter", "x_(n-1)", "x_n", "f(x_n)", "x_(n+1)", "f(x_(n+1))", "Error"]
    rows = []
    for iteracion, x_previous, x_current, f_current, x_next, f_next, error in table:
        rows.append([
            iteracion,
            f"{x_previous:.10f}",
            f"{x_current:.10f}",
            f"{f_current:.3e}",
            f"{x_next:.10f}",
            f"{f_next:.3e}",
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


def plot_secant(expression, table, root):
    f = parse_function(expression)
    iterates = [table[0][1], table[0][2]] + [row[4] for row in table]
    low, high = min(iterates), max(iterates)
    padding = max((high - low) * 0.2, 1.0)
    x_values = np.linspace(low - padding, high + padding, 1000)
    y_values = []
    for value in x_values:
        try:
            result = f(value)
            y_values.append(result if not isinstance(result, complex) and math.isfinite(result) else np.nan)
        except (ValueError, OverflowError, ZeroDivisionError):
            y_values.append(np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=f"f(x) = {expression}", color="blue", linewidth=2)
    plt.axhline(0, color="black", linewidth=1.0, linestyle="--")

    for _, x_previous, x_current, f_current, x_next, _, _ in table:
        f_previous = f(x_previous)
        plt.plot([x_previous, x_current], [f_previous, f_current], color="darkorange", alpha=0.55)
        plt.scatter([x_previous, x_current], [f_previous, f_current], color="orange", s=25)
        plt.scatter([x_next], [0], color="green", s=25, zorder=3)

    plt.scatter([root], [0], color="red", s=60, zorder=5, label=f"Raíz ≈ {root:.8f}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Método de la secante")
    plt.legend()
    plt.tight_layout()
    plt.savefig("secante_iteraciones.png")
    plt.show()
    plt.close()


def secant(expression: str, x0: float, x1: float, tol: float = 1e-5, max_iter: int = 100):
    f = parse_function(expression)
    if not all(math.isfinite(value) for value in (x0, x1)):
        raise ValueError("Las aproximaciones iniciales deben ser números finitos.")
    if x0 == x1:
        raise ValueError("Ingresa dos aproximaciones iniciales distintas.")
    if tol <= 0 or not math.isfinite(tol):
        raise ValueError("La tolerancia debe ser un número positivo y finito.")
    if max_iter < 1:
        raise ValueError("El número máximo de iteraciones debe ser al menos 1.")

    try:
        f0, f1 = f(x0), f(x1)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"La función no está definida en una aproximación inicial: {exc}") from exc
    if not math.isfinite(f0) or not math.isfinite(f1):
        raise ValueError("La función debe producir valores reales y finitos en las aproximaciones iniciales.")
    if abs(f0) <= tol:
        print(f"La aproximación inicial x0={x0:.10g} ya es una raíz (|f(x0)| <= tolerancia).")
        return x0, []
    if abs(f1) <= tol:
        print(f"La aproximación inicial x1={x1:.10g} ya es una raíz (|f(x1)| <= tolerancia).")
        return x1, []

    table = []
    root = None
    for iteration in range(1, max_iter + 1):
        denominator = f1 - f0
        scale = max(1.0, abs(f0), abs(f1))
        if abs(denominator) <= math.ulp(1.0) * scale:
            raise ZeroDivisionError(
                "f(x1) y f(x0) son demasiado parecidos; la fórmula de la secante divide por cero. Prueba otros valores iniciales."
            )

        x_next = x1 - f1 * (x1 - x0) / denominator
        if not math.isfinite(x_next):
            raise ValueError("La secante produjo una aproximación no finita; prueba otros valores iniciales.")
        try:
            f_next = f(x_next)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError(f"La función no está definida en x={x_next:.10g}: {exc}") from exc
        if not math.isfinite(f_next):
            raise ValueError("La función produjo un resultado no finito durante las iteraciones.")

        error = abs(x_next - x1)
        table.append((iteration, x0, x1, f1, x_next, f_next, error))
        print(
            f"Iteración {iteration}: x_(n+1) = {x1:.10g} - ({f1:.10g})*({x1:.10g} - {x0:.10g}) "
            f"/ ({f1:.10g} - {f0:.10g}) = {x_next:.10g}; f(x_(n+1))={f_next:.3e}; error={error:.3e}"
        )

        if abs(f_next) <= tol or error <= tol:
            root = x_next
            break
        x0, f0 = x1, f1
        x1, f1 = x_next, f_next
    else:
        raise RuntimeError(f"El método no convergió en {max_iter} iteraciones. Prueba otros valores iniciales.")

    print_table(table)
    print(f"\nRaíz aproximada: {root:.10f}")
    print(f"Tolerancia: {tol:.0e}")
    print(f"Número de iteraciones: {len(table)}")
    plot_secant(expression, table, root)
    print("Gráfica guardada como secante_iteraciones.png")

    return root, table


if __name__ == "__main__":
    try:
        expression = input("Ingresa f(x) (ejemplo: x^3 - 2x - 5, sin(x)-0.5, ln(x)-1): ").strip()
        if not expression:
            expression = "x**3 - 2*x - 5"

        x0_text = input("Primera aproximación x0 [2]: ").strip()
        x1_text = input("Segunda aproximación x1 [3]: ").strip()
        x0 = float(x0_text) if x0_text else 2.0
        x1 = float(x1_text) if x1_text else 3.0

        print(f"\nResolviendo f(x) = {expression} con el método de la secante")
        root, table = secant(expression, x0, x1, tol=1e-5)

        print(f"\nValor de la raíz aproximada: {root:.10f}")
        print(f"Número de iteraciones hasta la tolerancia 10^-5: {len(table)}")
        print("La gráfica se guardó en secante_iteraciones.png")
        print("La gráfica también se muestra en pantalla.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
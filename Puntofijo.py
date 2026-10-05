import ast
import math
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
def normalize_expression(expression: str) -> str:
    expr = expression.strip().replace(" ", "")
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = expr.replace("^", "**").replace("π", "pi")

    # Conversión de notación exponencial e^-x o e^(-x) a exp(-x)
    expr = re.sub(r"\be\*\*\((.*?)\)", r"exp(\1)", expr)
    expr = re.sub(r"\be\*\*(-?\w+)", r"exp(\1)", expr)

    # Normalización de funciones trigonométricas y logarítmicas
    expr = re.sub(r"\bsen(?=\()", "sin", expr)
    expr = re.sub(r"\bln(?=\()", "log", expr)
    expr = re.sub(r"\barcsin(?=\()", "asin", expr)
    expr = re.sub(r"\barccos(?=\()", "acos", expr)
    expr = re.sub(r"\barctan(?=\()", "atan", expr)

    function_names = (
        r"asin|acos|atan|atan2|asinh|acosh|atanh|sin|cos|tan|sinh|cosh|tanh|"
        r"sqrt|cbrt|exp|log10|log2|log|abs|floor|ceil|hypot|pow"
    )
    expr = re.sub(rf"(?<=[0-9x)])(?=(?:{function_names})\()", "*", expr)
    expr = re.sub(r"(?<=[0-9x)])(?=(?:pi|e|tau)\b)", "*", expr)
    expr = re.sub(r"(?<=[0-9x\)])(?=[0-9x(])", "*", expr)
    return expr


def generate_candidate_g_functions(f_expr_str: str, x0: float):
    """
    Genera automáticamente varias formas de g(x) despejando x de f(x) = 0,
    agregando términos de relajación o despejando funciones trascendentes.
    Evalúa |g'(x0)| para determinar la convergencia.
    """
    x = sp.Symbol("x")
    f_sym = sp.sympify(normalize_expression(f_expr_str))

    candidates = []

    # 1. Despejes automáticos directos de f(x) = 0
    try:
        solutions = sp.solve(sp.Eq(f_sym, 0), x)
        for sol in solutions:
            if sol.has(x):
                candidates.append(sol)
    except Exception:
        pass

    # 2. Métodos de relajación e igualación implícita: x = x + f(x) y x = x - f(x)
    candidates.append(x + f_sym)
    candidates.append(x - f_sym)

    # 3. Despejes algebraicos si es un polinomio
    poly = sp.Poly(f_sym, x) if f_sym.is_polynomial(x) else None
    if poly:
        degree = poly.degree()
        coeff_x1 = poly.coeff_monomial(x**1)
        if coeff_x1 != 0:
            g_linear = x - (f_sym / coeff_x1)
            candidates.append(sp.simplify(g_linear))
        if degree > 1:
            coeff_hn = poly.coeff_monomial(x**degree)
            rest = f_sym - coeff_hn * (x**degree)
            g_pow = sp.root(-rest / coeff_hn, degree)
            candidates.append(sp.simplify(g_pow))

    # Eliminar candidatos duplicados
    unique_candidates = []
    seen = set()
    for g_sym in candidates:
        g_str = str(g_sym)
        if g_str not in seen:
            seen.add(g_str)
            unique_candidates.append(g_sym)

    # Analizar criterio de convergencia |g'(x0)| < 1
    results = []
    for g_sym in unique_candidates:
        dg_sym = sp.diff(g_sym, x)
        try:
            val_dg = float(dg_sym.evalf(subs={x: x0}))
            converges = abs(val_dg) < 1.0
            results.append(
                {
                    "g_sym": g_sym,
                    "g_str": str(g_sym).replace("**", "^"),
                    "derivative_val": val_dg,
                    "converges": converges,
                }
            )
        except Exception:
            results.append(
                {
                    "g_sym": g_sym,
                    "g_str": str(g_sym).replace("**", "^"),
                    "derivative_val": float("nan"),
                    "converges": False,
                }
            )

    return results


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

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Expresión matemática inválida: {expression}") from exc

    function_names = set(allowed) - {"pi", "e", "tau"}

    def validate(node):
        if isinstance(node, ast.Expression):
            validate(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Usa solo constantes numéricas.")
        elif isinstance(node, ast.Name):
            if node.id not in {"x", "pi", "e", "tau"}:
                raise ValueError(f"Nombre no reconocido: {node.id}")
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
            raise ValueError("Operación matemática no válida.")

    validate(tree)

    def f(val_x):
        return eval(
            compile(tree, "<funcion_ingresada>", "eval"),
            {"__builtins__": {}},
            {"x": val_x, **allowed},
        )

    return f


def print_table(table):
    headers = ["Iter", "x_n", "g(x_n)", "f(x_n)", "Error"]
    rows = []
    for iteracion, x, next_x, residual, error in table:
        rows.append(
            [
                iteracion,
                f"{x:.10f}",
                f"{next_x:.10f}",
                f"{residual:.3e}",
                f"{error:.10e}",
            ]
        )

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    print("\nTabla de iteraciones")
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def plot_fixed_point(g, table, root):
    iterates = [table[0][1]] + [row[2] for row in table]
    span = max(iterates) - min(iterates)
    padding = max(span * 0.2, 0.5)
    lower, upper = min(iterates) - padding, max(iterates) + padding
    x_values = np.linspace(lower, upper, 1000)

    def evaluate_for_plot(value):
        try:
            result = g(value)
            return (
                result
                if not isinstance(result, complex) and math.isfinite(result)
                else np.nan
            )
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return np.nan

    g_values = np.array([evaluate_for_plot(value) for value in x_values])
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, g_values, label="g(x)", color="royalblue", linewidth=2)
    plt.plot(x_values, x_values, label="y = x", color="black", linestyle="--")
    for _, x, next_x, _, _ in table:
        plt.plot([x, x], [x, next_x], color="darkorange", linewidth=1.2)
        plt.plot([x, next_x], [next_x, next_x], color="darkorange", linewidth=1.2)
    plt.scatter(
        [root], [root], color="crimson", s=55, zorder=4, label=f"Raíz ≈ {root:.8f}"
    )
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Método de Punto Fijo")
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.savefig("punto_fijo_iteraciones.png")
    plt.show()
    plt.close()


def fixed_point(
    expression: str,
    g_expression: str,
    x0: float,
    tol: float = 1e-5,
    max_iter: int = 100,
):
    f = parse_function(expression)
    g = parse_function(g_expression)

    table = []
    x = x0

    for iteration in range(1, max_iter + 1):
        try:
            next_x = g(x)
            residual = f(next_x)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError) as exc:
            raise ValueError(
                f"Error evaluando g(x) o f(x) en x={x:.10g}: {exc}"
            ) from exc

        if isinstance(next_x, complex) or not math.isfinite(next_x):
            raise ValueError("Resultado complejo o divergente.")

        error = abs(next_x - x)
        table.append((iteration, x, next_x, residual, error))

        if error <= tol and abs(residual) <= tol:
            root = next_x
            break
        x = next_x
    else:
        raise RuntimeError(f"No hubo convergencia en {max_iter} iteraciones.")

    print_table(table)
    print(f"\nRaíz aproximada: {root:.10f}")
    print(f"Número de iteraciones: {len(table)}")
    plot_fixed_point(g, table, root)
    return root, table


if __name__ == "__main__":
    try:
        expression = input("Ingrese f(x)=0 [ej: e**-x - x o log(x) - 1]: ").strip()
        if not expression:
            expression = "e**-x - x"

        x0_text = input("Valor inicial x0 [0.5]: ").strip()
        x0 = float(x0_text) if x0_text else 0.5

        print("\n--- Generando y analizando opciones de g(x) ---")
        g_candidates = generate_candidate_g_functions(expression, x0)

        best_g = None
        min_deriv = float("inf")

        for idx, item in enumerate(g_candidates, start=1):
            status = "CONVERGE" if item["converges"] else "DIVERGE"
            deriv_val = item["derivative_val"]
            deriv_str = (
                f"{deriv_val:.4f}" if not math.isnan(deriv_val) else "N/A"
            )
            print(f"[{idx}] g(x) = {item['g_str']}")
            print(f"    |g'({x0})| = {deriv_str} -> Estado: {status}\n")

            # Buscar la opción con mejor tasa de convergencia (menor derivada en x0)
            if not math.isnan(deriv_val) and abs(deriv_val) < min_deriv:
                min_deriv = abs(deriv_val)
                best_g = str(item["g_sym"])

        if best_g is None and g_candidates:
            best_g = str(g_candidates[0]["g_sym"])

        print(f"Versión seleccionada automáticamente: g(x) = {best_g}")
        fixed_point(expression, best_g, x0, tol=1e-5)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
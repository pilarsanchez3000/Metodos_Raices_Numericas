# Laboratorio No. 4: Métodos Numéricos para el Cálculo de Raíces

**Repositorio:** `Metodos_Raices_Numericas`  
**Estudiante:** Pilar Sánchez  
**Institución:** Universidad Tecnológica de Panamá — Facultad de Ingeniería Mecánica  

---

## 3. Actividades con GitHub Copilot

En esta sección se documenta el proceso iterativo de colaboración con **GitHub Copilot** para el desarrollo, depuración, documentación y optimización de los scripts en Python correspondientes a los distintos métodos numéricos de búsqueda de raíces[cite: 1].

---

### 3.1 Registro de Prompts Utilizados y Evolución del Código

#### A. Método de Bisección (`lab4.py` / `lab4_prueba.py` / `biseccion.py`)
* **Prompt 1 (Inicio):** *"haz un código que use método de bisección que ingrese cualquier función y que evalúe hasta tolerancia error 10^-5 haz una tabla con las iteraciones y realiza una gráfica"*
  * **Evolución:** Copilot generó la estructura base para parsear funciones, calcular la bisección en $[a, b]$, estructurar la tabla de iteraciones y guardar la imagen con Matplotlib.
* **Prompt 2 (Conteo de Iteraciones):** *"mejora el código para que diga el número de iteraciones"*
  * **Evolución:** Se añadió un contador explícito e impresión en consola con el número total de pasos necesarios para alcanzar la tolerancia $10^{-5}$.
* **Prompt 3 (Entrada Explícita):** *"mejora el código para que me pida el intervalo a evaluar"*
  * **Evolución:** Se configuró el programa para solicitar directamente los límites $a$ y $b$, con verificación inmediata del cambio de signo $f(a) \cdot f(b) < 0$.
* **Prompt 4 (Manejo de Errores de Sintaxis):** *"Error: invalid decimal literal (<string>, line 1) corrige este error en el código"*
  * **Evolución:** Se incorporó la función `normalize_expression()` para procesar la entrada matemática del usuario convirtiendo notación informal (`2x`, `x^2`, `sen(x)`) a sintaxis válida de Python (`2*x`, `x**2`, `sin(x)`).

#### B. Método de Regula Falsi Modificada / Illinois (`falsa posición modificada.py` / `lab4_illinois.py`)
* **Prompt 1 (Visibilidad de Iteraciones):** *"mejora este codigo para que aprezca las iteraciones"*
  * **Evolución:** Copilot mejoró el código para mostrar cada paso en tiempo real en consola (con $a$, $b$, $c$, $f(c)$ y ancho del intervalo), imprimiendo la tabla final alineada y marcando en la gráfica los puntos numerados de cada iteración.
* **Prompt 2 (Falsa Posición Modificada / Illinois):** *"MEJORA ESTE CODIGO PARA QUE TRABAJE FALSA POSICION MODIFICADA O ILLINOIS"*
  * **Evolución:** Copilot detectó que la versión anterior utilizaba la lógica de Pegasus y la adaptó al método **Illinois**. Se ajustaron las ponderaciones para reducir el peso a la mitad (`wa *= 0.5` o `wb *= 0.5`) cuando un extremo permanece fijo en iteraciones consecutivas (`previous_side == "a"` o `"b"`), acelerando la convergencia y solucionando el estancamiento de bordes.
* **Prompt 3 (Búsqueda Automática de Intervalos):** *"agrega una función para buscar un intervalo con cambio de signo si los límites iniciales no cumplen"*
  * **Evolución:** Se implementó `find_bracket()` como mecanismo de seguridad para buscar un intervalo cercano con $f(a) \cdot f(b) < 0$ cuando los puntos ingresados por el usuario no encierran una raíz.

#### C. Método de Newton-Raphson y Raíz Cúbica (`lab4 parte3.py` / `lab4 .3 pruebas.py`)
* **Prompt 1 (Transición a Newton-Raphson):** *"mejora el código y usa newton raphson"*
  * **Evolución:** Se modificó el algoritmo para emplear la aproximación $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$, sustituyendo la búsqueda por intervalos por un valor inicial $x_0$.
* **Prompt 2 (Raíz Cúbica de $x$):** *"mejora este código para que realice la función raíz cúbica de $x$ por método de newton raphson y coloque sus debidas iteraciones y su debida gráfica"*
  * **Evolución:** Se especializó el solver para la ecuación $y^3 - x = 0$ con la fórmula simplificada $y_{n+1} = \frac{2y_n + x/y_n^2}{3}$, permitiendo evaluar valores positivos, negativos y cero, con centrado automático de la curva $y = \sqrt[3]{x}$ en la gráfica.

#### D. Método de Punto Fijo (`lab4part4puntofijo.py`)
* **Prompt 1 (Implementación de Punto Fijo):** *"modifica este código para que trabaje con punto fijo"*
  * **Evolución:** Se estructuró el flujo para iterar $x_{n+1} = g(x_n)$, solicitando la ecuación $f(x) = 0$ y su correspondiente despeje algebraico $x = g(x)$.
* **Prompt 2 (Eliminación de Parámetros Adicionales):** *"mejora el código para que haga la gráfica de punto fijo y elimina esa parte de lambda"*
  * **Evolución:** Se eliminó la estimación mediante factor de relajación $\lambda$ para trabajar de forma directa con el despeje $g(x)$ ingresado por el usuario, generando la gráfica de convergencia tipo *cobweb* (telaraña) intersectando $y = g(x)$ con $y = x$.

#### E. Método de la Secante (`lab4 parte5secante.py` / `secante.py`)
* **Prompt 1 (Método de la Secante):** *"mejora el código trabaja con método de secante haz que funcione al ingresar una función cualquiera"*
  * **Evolución:** Se adaptó el algoritmo para requerir dos puntos iniciales ($x_0, x_1$) e iterar según $x_{n+1} = x_n - f(x_n) \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$, prescindiendo del cálculo explícito de derivadas.

#### F. Refactorización, Docstrings Automáticos y Robustez de Código (`biseccion.py`, `secante.py`, `falsa posición modificada.py`)
* **Prompt:** *"mejora el codigo y agrega el docstrigns automatico"*
  * **Evolución:** Copilot realizó una refactorización integral en los scripts del proyecto. Se agregaron *docstrings* explicativos a todas las funciones públicas (`normalize_expression`, `parse_function`, `print_table`, `plot_*`, solvers y `main`). Se solucionó un fallo de expresiones regulares que transformaba `ln(x)` en `l*o*g*(x)` al procesar la multiplicación implícita, reemplazándolo por un parser controlado que maneja casos como `2x`, `ln(x)` y `x(2+3)`. Finalmente, se añadieron exportaciones explícitas con `__all__` y se validó el comportamiento mediante pruebas automáticas ejecutadas en la terminal.

---

### 3.2 Registro de Sugerencias de GitHub Copilot

| Sugerencia de Copilot | Estado | Razón de la Decisión |
| :--- | :---: | :--- |
| **Normalización de expresiones matemáticas:** Conversión automática de cadenas (`2x` $\rightarrow$ `2*x`, `x^2` $\rightarrow$ `x**2`, `sen` $\rightarrow$ `sin`, `ln` $\rightarrow$ `log`). | **Aceptada** | Evita errores de sintaxis (`invalid decimal literal` o `NameError`) al procesar entradas matemáticas en formato natural escritas por el usuario. |
| **Inclusión de Docstrings automáticos:** Generación de documentación estructurada en cada función pública con especificación de parámetros, tipos e historial de retornos. | **Aceptada** | Mejora significativamente la legibilidad, mantenibilidad y calidad técnica del código fuente del proyecto. |
| **Ajuste de peso Illinois (`wa *= 0.5` / `wb *= 0.5`):** Reducir a la mitad el valor ponderado del extremo estático cuando se repite el mismo lado en iteraciones consecutivas. | **Aceptada** | Resuelve el problema clásico de estancamiento de bordes del método de Regula Falsi tradicional, permitiendo que el intervalo se contraiga eficazmente. |
| **Parser controlado para multiplicación implícita:** Reemplazo de expresiones regulares genéricas por un procesador que evita corromper nombres de funciones matemáticas como `ln(x)` o `sin(x)`. | **Aceptada** | Soluciona errores en tiempo de ejecución al evaluar expresiones simbólicas avanzadas ingresadas por el usuario. |
| **Búsqueda automática de intervalo de respaldo (`find_bracket`):** Exploración automática para encontrar $f(a) \cdot f(b) < 0$ cuando los puntos ingresados no encierran una raíz. | **Aceptada** | Proporciona solidez al script evitando que el programa falle si el usuario ingresa un intervalo inicial incorrecto. |
| **Definición de exportación explícita (`__all__`):** Estructurar los módulos de Python delimitando exactamente las funciones públicas del script. | **Aceptada** | Proporciona una arquitectura modular más limpia y estándar para reutilizar los solvers en otros proyectos. |
| **Búsqueda automática de intervalos en Bisección:** Determinar automáticamente $[a, b]$ mediante escaneo en Bisección pura. | **Rechazada** | Se rechazó para mantener el control explícito del intervalo $[a, b]$ ingresado manualmente por el usuario, cumpliendo con la consigna académica de la práctica. |
| **Despeje de Punto Fijo mediante derivadas ($\lambda$ / Newton implícito):** Generar automáticamente $g(x) = x - \lambda f(x)$ o $g(x) = x - \frac{f(x)}{f'(x)}$. | **Rechazada** | Se rechazó para aplicar el método de **Punto Fijo clásico** puro ($x_{n+1} = g(x_n)$) basado en un despeje algebraico explícito, evitando confundir el método con Newton-Raphson. |
| **Tabla interactiva de iteraciones:** Formatear las columnas ($a$, $b$, $c$, $f(c)$, Error / $x_n$, $f(x_n)$) con alineación y precisión fija en consola. | **Aceptada** | Proporciona un reporte numérico claro, ordenado e ideal para la verificación de los resultados en los informes de laboratorio. |
| **Despliegue de ventana de gráfica (`plt.show()`):** Incluir visualización en pantalla además de guardar la imagen en disco (`.png`). | **Aceptada** | Permite inspeccionar visualmente la curva y la raíz calculada de manera inmediata al finalizar las iteraciones del programa. |

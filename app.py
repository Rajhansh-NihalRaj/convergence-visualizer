import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')

st.title("📊 Algorithm Convergence Visualizer")

# ---------------- EQUATIONS ----------------
equations = {
    "x^3 - x - 2": "x**3 - x - 2",
    "x^2 - 4": "x**2 - 4",
    "cos(x) - x": "cos(x) - x",
    "x^3 - 2x - 5": "x**3 - 2*x - 5",
    "e^(-x) - x": "exp(-x) - x",
}

choice = st.selectbox("Choose Equation", list(equations.keys()) + ["Custom"])

if choice == "Custom":
    expr = st.text_input("Enter equation (example: x**2 - 4)")
else:
    expr = equations[choice]

# ---------------- METHODS ----------------

def bisection(f, a, b):
    errors = []
    for _ in range(50):
        c = (a + b) / 2
        errors.append(abs(b - a))
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return errors

def newton(f, df, x0):
    errors = []
    for _ in range(50):
        try:
            x1 = x0 - f(x0)/df(x0)
        except:
            break
        errors.append(abs(x1 - x0))
        x0 = x1
    return errors

def secant(f, x0, x1):
    errors = []
    for _ in range(50):
        try:
            x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        except:
            break
        errors.append(abs(x2 - x1))
        x0, x1 = x1, x2
    return errors

def regula_falsi(f, a, b):
    errors = []
    for _ in range(50):
        try:
            c = (a*f(b)-b*f(a))/(f(b)-f(a))
        except:
            break
        errors.append(abs(f(c)))
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return errors

def fixed_point(f, x0):
    errors = []
    for _ in range(50):
        x1 = x0 - f(x0)
        errors.append(abs(x1 - x0))
        x0 = x1
    return errors

# ---------------- RUN ----------------

if expr:
    try:
        # Safe parse
        expr = expr.replace("^", "**")

        f_expr = sp.sympify(expr)
        f = sp.lambdify(x, f_expr, modules=["numpy"])

        df_expr = sp.diff(f_expr, x)
        df = sp.lambdify(x, df_expr, modules=["numpy"])

        if st.button("Run Visualization"):

            b = bisection(f, 1, 2)
            n = newton(f, df, 1.5)
            s = secant(f, 1, 2)
            r = regula_falsi(f, 1, 2)
            fp = fixed_point(f, 1.5)

            fig, ax = plt.subplots()

            ax.plot(b, label="Bisection")
            ax.plot(n, label="Newton")
            ax.plot(s, label="Secant")
            ax.plot(r, label="Regula Falsi")
            ax.plot(fp, label="Fixed Point")

            ax.legend()
            ax.set_title("Convergence Comparison")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Error")

            st.pyplot(fig)

            st.success("✅ Working!")

    except Exception as e:
        st.error(f"Error: {e}")
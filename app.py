import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Symbol
x = sp.symbols('x')

st.set_page_config(page_title="Convergence Visualizer", layout="centered")

st.title("📊 Algorithm Convergence Visualizer")

# ---------------- EQUATIONS ----------------
equations = {
    "x^3 - x - 2": "x^3 - x - 2",
    "x^2 - 4": "x^2 - 4",
    "cos(x) - x": "cos(x) - x",
    "x^3 - 2x - 5": "x^3 - 2*x - 5",
    "e^(-x) - x": "exp(-x) - x",
}

choice = st.selectbox("Choose Equation", list(equations.keys()) + ["Custom"])

if choice == "Custom":
    expr = st.text_input("Enter equation in x")
else:
    expr = equations[choice]

# ---------------- METHODS ----------------

def bisection(f, a, b, tol=1e-5):
    errors = []
    while abs(b - a) > tol:
        c = (a + b) / 2
        errors.append(abs(b - a))
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return errors

def newton(f, df, x0, tol=1e-5):
    errors = []
    x_val = x0
    while True:
        x_new = x_val - f(x_val)/df(x_val)
        errors.append(abs(x_new - x_val))
        if abs(x_new - x_val) < tol:
            break
        x_val = x_new
    return errors

def secant(f, x0, x1, tol=1e-5):
    errors = []
    while True:
        x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        errors.append(abs(x2 - x1))
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return errors

def regula_falsi(f, a, b, tol=1e-5):
    errors = []
    while True:
        c = (a*f(b) - b*f(a))/(f(b) - f(a))
        errors.append(abs(f(c)))
        if abs(f(c)) < tol:
            break
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return errors

def fixed_point(g, x0, tol=1e-5):
    errors = []
    while True:
        x1 = g(x0)
        errors.append(abs(x1 - x0))
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return errors

# ---------------- RUN ----------------

if expr:
    try:
        # CLEAN INPUT
        expr = expr.strip()
        expr = expr.replace("^", "**")

        # SAFE FUNCTION MAP
        allowed_functions = {
            "cos": sp.cos,
            "sin": sp.sin,
            "exp": sp.exp,
            "log": sp.log
        }

        # PARSE
        f_expr = sp.sympify(expr, locals=allowed_functions)

        # NUMERIC FUNCTIONS
        f = sp.lambdify(x, f_expr, "numpy")
        df_expr = sp.diff(f_expr, x)
        df = sp.lambdify(x, df_expr, "numpy")

        st.subheader("⚙️ Run Methods")

        if st.button("Run Visualization"):

            b = bisection(f, 1, 2)
            n = newton(f, df, 1.5)
            s = secant(f, 1, 2)
            r = regula_falsi(f, 1, 2)

            g = lambda val: val - f(val)
            fp = fixed_point(g, 1.5)

            # Plot
            fig, ax = plt.subplots()

            ax.plot(b, label="Bisection")
            ax.plot(n, label="Newton-Raphson")
            ax.plot(s, label="Secant")
            ax.plot(r, label="Regula Falsi")
            ax.plot(fp, label="Fixed Point")

            ax.set_xlabel("Iterations")
            ax.set_ylabel("Error")
            ax.set_title("Convergence Comparison")
            ax.legend()
            ax.grid()

            st.pyplot(fig)

            # Iteration count
            st.subheader("📈 Iteration Count Comparison")
            st.write({
                "Bisection": len(b),
                "Newton": len(n),
                "Secant": len(s),
                "Regula Falsi": len(r),
                "Fixed Point": len(fp)
            })

            st.success("✅ Visualization Generated Successfully!")

    except Exception as e:
        st.error("❌ Invalid equation. Please check syntax.")
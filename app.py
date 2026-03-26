import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

x = sp.symbols('x')

st.set_page_config(page_title="Convergence Visualizer", layout="wide")

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align: center;'>📊 Algorithm Convergence Visualizer</h1>",
    unsafe_allow_html=True
)

st.markdown("<p style='text-align: center; color: gray;'><i>Compare root-finding numerical methods in real-time by Rajhansh</i></p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

equations = {
    "x³ - x - 2": "x^3 - x - 2",
    "x² - 4": "x^2 - 4",
    "cos(x) - x": "cos(x) - x",
    "x³ - 2x - 5": "x^3 - 2*x - 5",
    "e^(-x) - x": "exp(-x) - x",
}

choice = st.sidebar.selectbox("Choose Equation", list(equations.keys()) + ["Custom"])

if choice == "Custom":
    expr = st.sidebar.text_input("Enter equation")
else:
    expr = equations[choice]

iterations = st.sidebar.slider("Iterations", 10, 100, 50)

selected_methods = st.sidebar.multiselect(
    "Select Methods",
    ["Bisection", "Newton", "Secant", "Regula Falsi", "Fixed Point"],
    default=["Bisection", "Newton", "Secant"]
)

# ---------------- SAFETY ----------------
def safe_append(errors, value):
    if abs(value) > 1e6:
        return False
    errors.append(abs(value))
    return True

# ---------------- METHODS ----------------

def bisection(f, a, b):
    errors = []
    for _ in range(iterations):
        c = (a + b) / 2
        if not safe_append(errors, b - a):
            break
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return errors

def newton(f, df, x0):
    errors = []
    for _ in range(iterations):
        try:
            x1 = x0 - f(x0)/df(x0)
            if not safe_append(errors, x1 - x0):
                break
            x0 = x1
        except:
            break
    return errors

def secant(f, x0, x1):
    errors = []
    for _ in range(iterations):
        try:
            x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
            if not safe_append(errors, x2 - x1):
                break
            x0, x1 = x1, x2
        except:
            break
    return errors

def regula_falsi(f, a, b):
    errors = []
    for _ in range(iterations):
        try:
            c = (a*f(b)-b*f(a))/(f(b)-f(a))
            if not safe_append(errors, f(c)):
                break
            if f(a)*f(c) < 0:
                b = c
            else:
                a = c
        except:
            break
    return errors

def fixed_point(f, x0):
    errors = []
    for _ in range(iterations):
        x1 = x0 - f(x0)
        if not safe_append(errors, x1 - x0):
            break
        x0 = x1
    return errors

# ---------------- MAIN ----------------

if expr:
    try:
        expr = expr.replace("^", "**")

        allowed = {
            "cos": sp.cos,
            "sin": sp.sin,
            "exp": sp.exp,
            "log": sp.log
        }

        f_expr = sp.sympify(expr, locals=allowed)
        f = sp.lambdify(x, f_expr, "numpy")

        df_expr = sp.diff(f_expr, x)
        df = sp.lambdify(x, df_expr, "numpy")

        col1, col2 = st.columns(2)

        # ---------------- GRAPH ----------------
        with col1:
            st.subheader("📈 Convergence Graph")

            fig, ax = plt.subplots()

            if "Bisection" in selected_methods:
                ax.plot(bisection(f, 1, 2), label="Bisection")

            if "Newton" in selected_methods:
                ax.plot(newton(f, df, 1.5), label="Newton")

            if "Secant" in selected_methods:
                ax.plot(secant(f, 1, 2), label="Secant")

            if "Regula Falsi" in selected_methods:
                ax.plot(regula_falsi(f, 1, 2), label="Regula Falsi")

            if "Fixed Point" in selected_methods:
                ax.plot(fixed_point(f, 1.5), label="Fixed Point")

            ax.set_yscale("log")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Error")
            ax.legend()
            ax.grid()

            st.pyplot(fig)

        # ---------------- 3D GRAPH ----------------
        with col2:
            st.subheader("🌐 3D Function Surface")

            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(111, projection='3d')

            X = np.linspace(-5, 5, 50)
            Y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(X, Y)

            Z = np.zeros_like(X)
            for i in range(iterations):
                Z += f(X) / (i + 1)

            ax3d.plot_surface(X, Y, Z, cmap='viridis')
            ax3d.set_title("Function Surface")

            st.pyplot(fig3d)

        # ---------------- INFO ----------------
        st.markdown("---")
        st.subheader("📊 Summary")

        st.info("💡 **Newton** usually converges quadratically (fastest), **Bisection** is guaranteed but slow, and **Secant** is a great middle-ground.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
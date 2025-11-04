import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Markov y Colas", layout="wide")

tab1, tab2 = st.tabs(["🔗 Cadenas de Markov", "📊 Teoría de Colas"])

# ----------------------------------------------------------
# 🔗 Pestaña 1: Cadenas de Markov
# ----------------------------------------------------------
with tab1:
    st.header("Simulación de Cadenas de Markov")

    st.write("Introduce la matriz de transición:")
    size = st.number_input("Número de estados", min_value=2, max_value=10, value=4)
    matrix = []

    for i in range(size):
        row = st.text_input(f"Fila {i+1} (separada por comas)", value="0.3,0.4,0.3,0.0")
        matrix.append([float(x) for x in row.split(",")])

    matrix = np.array(matrix)
    st.write("Matriz de transición:")
    st.write(matrix)

    # --- Estado inicial y simulación ---
    st.write("Estado inicial:")
    estado_inicial = st.text_input("Ejemplo: 1,0,0,0", value="1,0,0,0")
    estado = np.array([float(x) for x in estado_inicial.split(",")])

    pasos = st.slider("Número de pasos", 1, 50, 10)
    historia = [estado]

    for _ in range(pasos):
        estado = estado @ matrix
        historia.append(estado)

    historia = np.array(historia)
    st.line_chart(historia)

    # --- Cálculo de probabilidades de absorción ---
    st.subheader("📈 Probabilidades de absorción (si existen estados absorbentes)")

    absorbentes_input = st.text_input(
        "Estados absorbentes (separados por comas, ej: 1,2)", value="1,2"
    )
    try:
        absorbentes = [int(x.strip()) - 1 for x in absorbentes_input.split(",") if x.strip() != ""]
        transitorios = [i for i in range(size) if i not in absorbentes]

        if len(absorbentes) > 0 and len(transitorios) > 0:
            # Matrices R y Q
            Q = matrix[np.ix_(transitorios, transitorios)]
            R = matrix[np.ix_(transitorios, absorbentes)]

            # N = (I - Q)^-1
            I = np.eye(len(Q))
            N = np.linalg.inv(I - Q)

            # B = N * R
            B = N @ R

            st.write("Matriz Q (transitorio → transitorio):")
            st.write(Q)
            st.write("Matriz R (transitorio → absorbente):")
            st.write(R)
            st.write("Matriz fundamental N = (I - Q)⁻¹:")
            st.write(N)
            st.write("Matriz B = N · R (probabilidades de absorción):")
            st.write(B)

            # Mostrar interpretación
            st.markdown("**Interpretación:** Cada fila de B corresponde a un estado transitorio y muestra "
                        "la probabilidad de terminar en cada estado absorbente.")

            for i, estado_t in enumerate(transitorios):
                probs = ", ".join([f"{p:.4f}" for p in B[i]])
                st.write(f"Desde el estado {estado_t + 1} → {probs}")

        else:
            st.warning("Por favor define al menos un estado absorbente y uno transitorio.")
    except Exception as e:
        st.error(f"Error al calcular las probabilidades de absorción: {e}")


# ----------------------------------------------------------
# 📊 Pestaña 2: Teoría de Colas
# ----------------------------------------------------------
with tab2:
    st.header("Modelos de Teoría de Colas")

    modelo = st.selectbox("Selecciona el modelo", ["M/M/1", "M/M/c", "M/M/1/K", "M/G/1"])

    lambda_ = st.number_input("Tasa de llegada (λ)", min_value=0.1, value=1.0)
    mu = st.number_input("Tasa de servicio (μ)", min_value=0.1, value=1.5)

    if modelo == "M/M/1":
        rho = lambda_ / mu
        L = rho / (1 - rho)
        W = 1 / (mu - lambda_)
        Lq = rho**2 / (1 - rho)
        Wq = Lq / lambda_

        st.write(f"ρ (Utilización): {rho:.2f}")
        st.write(f"L (Clientes en sistema): {L:.2f}")
        st.write(f"Lq (Clientes en cola): {Lq:.2f}")
        st.write(f"W (Tiempo en sistema): {W:.2f}")
        st.write(f"Wq (Tiempo en cola): {Wq:.2f}")

    elif modelo == "M/M/c":
        c = st.number_input("Número de servidores (c)", min_value=1, value=2)
        rho = lambda_ / (c * mu)

        def P0():
            sum_terms = sum([(lambda_/mu)**n / math.factorial(n) for n in range(c)])
            last_term = ((lambda_/mu)**c / math.factorial(c)) * (1 / (1 - rho))
            return 1 / (sum_terms + last_term)

        p0 = P0()
        Lq = (p0 * ((lambda_/mu)**c) * rho) / (math.factorial(c) * (1 - rho)**2)
        L = Lq + lambda_/mu
        W = L / lambda_
        Wq = Lq / lambda_

        st.write(f"ρ (Utilización): {rho:.2f}")
        st.write(f"P₀ (Probabilidad de 0 clientes): {p0:.4f}")
        st.write(f"L (Clientes en sistema): {L:.2f}")
        st.write(f"Lq (Clientes en cola): {Lq:.2f}")
        st.write(f"W (Tiempo en sistema): {W:.2f}")
        st.write(f"Wq (Tiempo en cola): {Wq:.2f}")

    elif modelo == "M/M/1/K":
        K = st.number_input("Capacidad máxima (K)", min_value=1, value=5)
        rho = lambda_ / mu

        if rho == 1:
            P0 = 1 / (K + 1)
        else:
            P0 = (1 - rho) / (1 - rho**(K + 1))

        Pn = [P0 * rho**n for n in range(K + 1)]
        L = sum(n * Pn[n] for n in range(K + 1))
        lambda_eff = lambda_ * (1 - Pn[K])
        W = L / lambda_eff

        st.write(f"ρ (Utilización): {rho:.2f}")
        st.write(f"P₀ (Probabilidad de 0 clientes): {P0:.4f}")
        st.write(f"L (Clientes en sistema): {L:.2f}")
        st.write(f"λₑ (Tasa efectiva de llegada): {lambda_eff:.2f}")
        st.write(f"W (Tiempo promedio en sistema): {W:.2f}")

        fig, ax = plt.subplots()
        ax.bar(range(K + 1), Pn, color='lightgreen')
        ax.set_title("Distribución de estados (Pn)")
        ax.set_xlabel("Número de clientes")
        ax.set_ylabel("Probabilidad")
        st.pyplot(fig)

    elif modelo == "M/G/1":
        E_s = st.number_input("Tiempo medio de servicio (E[S])", min_value=0.1, value=1.0)
        Var_s = st.number_input("Varianza del servicio (Var[S])", min_value=0.0, value=0.5)
        rho = lambda_ * E_s
        Lq = (lambda_**2 * Var_s + rho**2) / (2 * (1 - rho))
        L = Lq + rho
        W = L / lambda_
        Wq = Lq / lambda_

        st.write(f"ρ (Utilización): {rho:.2f}")
        st.write(f"L (Clientes en sistema): {L:.2f}")
        st.write(f"Lq (Clientes en cola): {Lq:.2f}")
        st.write(f"W (Tiempo en sistema): {W:.2f}")
        st.write(f"Wq (Tiempo en cola): {Wq:.2f}")
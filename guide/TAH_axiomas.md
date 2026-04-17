# Teoría de Agencia Homeostática (TAH) — Definición canónica del autor

> **REGLA**: Esta es la definición ÚNICA y AUTORITATIVA de TAH dada por el autor original.
> NO modificar P1, P2, P3 en función de feedback de reviewers, analogías biológicas
> ajenas, o paráfrasis de HRRL/Keramati. Son **axiomas ontológicos generales**, no
> descripción del mecanismo experimental.
>
> Fuente canónica: `paper_iterations/paper_jaiio (4).pdf`, §3.1
> "Teoría de agencia homeostática".

Denotamos esta teoría `T_HA`, con P1–P3 como base axiomática y T1 como teorema
derivable. Los postulados fijan qué requiere la autonomía, qué desencadena la acción
y qué hace valiosa una contribución; toda decisión de diseño posterior descansa sobre
ellos.

## Postulados (P1–P3)

### P1 — Autonomía (Ashby, 1956; Beer, 1972)

**¿Qué es autonomía?** Que la política del agente dependa de su estado interno.

$$
\mathrm{Aut}(a_i) \iff \exists\,\delta_i \in S_L(a_i):\ \pi_i = \pi_i\!\left(\delta_i(t)\right)
$$

Formulación en una línea del autor:
**Sostener acción de calidad en el tiempo bajo regulación interna.**

### P2 — Impulso (Sterling & Eyer, 1988)

**¿Qué hace a un agente actuar?** Un impulso interno que crece con el desvío del
equilibrio y cuya magnitud determina monótonamente la probabilidad de activación.

$$
D(\varepsilon) = 0,\ D'(\delta_i) > 0 \ \forall\,\delta_i > \varepsilon,\
p_i(t) = \sigma\!\left(D(\delta_i(t))\right)
$$

Formulación en una línea del autor:
**Presión creciente para satisfacer un déficit ante perturbaciones.**

### P3 — Compuerta de calidad (Huta & Waterman, 2014; Ryan & Deci, 2001)

**¿Qué hace valiosa a una acción?** Un cambio efectivo en el estado del entorno
regulado. Sea `g(e(t), e(t+1)) ≥ 0` la medida de ese cambio; el impulso decrece
proporcionalmente.

$$
\Delta \delta_i = -\alpha \cdot g\!\left(e(t), e(t{+}1)\right),\ \alpha > 0
$$

Formulación en una línea del autor:
**Acción de calidad es aquella que satisface el déficit.**

## T1 — Teorema derivable de P1–P3

$$
\begin{aligned}
\delta_i(t) > \varepsilon \;\Rightarrow\;& p_i(t) = \sigma(D(\delta_i(t))) > \tfrac{1}{2},\quad \partial p_i / \partial \delta_i > 0 \quad (\text{P2}) \\
& \pi_i(t) = \pi_i(\delta_i(t)) \quad (\text{P1}) \\
& r_t = -\Delta \delta_i = \alpha \cdot g(e(t), e(t{+}1)) \ge 0 \quad (\text{P3})
\end{aligned}
$$

Bajo `r` acotado y tasa `η_t` que cumple Robbins–Monro (`Σ η_t = ∞`, `Σ η_t² < ∞`),
la actualización TD(0) sobre `Q(·)` converge a un punto fijo del operador de Bellman
y la política greedy resultante maximiza, en expectativa, la reducción de impulso
(Keramati & Gutkin, 2014).

## Predicción (Pr1) — formulación canónica del autor

> Dado un agente expuesto a perturbaciones, cuyo impulso crece con el desvío del
> equilibrio y sólo decrece ante cambio efectivo, se esperaría que emerja acción
> sostenida de calidad, con probabilidad de activación creciente en el déficit.

## Notas para edición del paper

- P1–P3 son **generales**, no específicos del experimento.
- No introducir vocabulario de HRRL (TD, Q-learning, sigmoide concreta, déficit
  epistémico) dentro de los enunciados de P1–P3. Eso va en Arquitectura (§3.2).
- Las citas de soporte (Ashby, Beer, Sterling, Huta, Ryan) acompañan el postulado
  como referencia, pero la formulación es del autor.
- No condensar en "supuestos mínimos para HRRL"; son axiomas ontológicos para
  *cualquier* agente homeostático.
- En la versión corta, la ecuación `p_i = σ(D(δ_i))` en P2 puede reducirse a
  `p_i = h(δ_i)` con `h` monótona creciente, para desacoplar el axioma de la
  sigmoide específica usada en la arquitectura.

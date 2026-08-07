# Papers de interés — Homeostasis, RL y Agentes Autónomos

---

## Fundacionales

### Keramati & Gutkin (2014)
**Homeostatic reinforcement learning for integrating reward collection and physiological stability**
*eLife, 3:e04811*

Define HRRL: drive D(H_t) como distancia al setpoint en espacio homeostático N-dimensional. Reward = reducción de drive. Teorema de equivalencia: maximizar reward ≡ minimizar desviación del setpoint. RL integrado para aprender políticas. NO tiene drift basal ni resorte elástico — el estado solo cambia por acciones. NO tiene gate sigmoide — usa política RL (softmax sobre Q).

→ Diferencia clave con Pro-Action: Γ reemplaza RL con gate sigmoide + agrega drift λ y resorte α·g.

---

### Hulme, Morville & Gutkin (2019)
**Neurocomputational theories of homeostatic control**
*Physics of Life Reviews*

Review de HRRL como teoría de control. El lazo homeostático como controlador feedback. Drive como "energía de motivación". Interpretación del drive como sorpresa (negative log-probability).

→ Relevante para enmarcar Γ en lenguaje de teoría de control: setpoint, ganancia, estabilidad, oscilaciones.

---

## Extensiones de HRRL

### Laurençon, Ségerie, Lussange & Gutkin (2021)
**Continuous Homeostatic Reinforcement Learning for Self-Regulated Autonomous Agents**
*arXiv:2109.06580*

Extiende HRRL a tiempo continuo. Introduce "self-regulated agents": estado interno cambia continuamente incluso sin acciones (≈ drift λ). Drive como norma euclidiana: D(δ) = √(ε + δᵀδ). Agente no conoce su propio cuerpo (f, g, S desconocidos) — debe aprender cómo su estado responde a sus acciones. Cross-need interactions: reward por saciar una necesidad depende del estado de las otras. Equivalencia probada en tiempo continuo.

→ Clave para Γ(n>1): norma euclidiana vs producto de sigmoides como formas alternativas de componer drives. Cross-coupling formalizado.

---

### Yoshida, Sprekeler & Gutkin (2025)
**Linking homeostasis to reinforcement learning: internal state control of motivated behavior**
*Current Opinion in Behavioral Sciences, 101611*

Posiciona HRRL como framework para AI y robótica. Risk aversion emerge naturalmente (drive cuadrático penaliza desviaciones grandes). Anticipatory regulation (allostasis): agentes aprenden a actuar antes de que el déficit sea severo. Deep RL + HRRL para exploración autónoma y comportamiento jerárquico. Aplicaciones a desórdenes psiquiátricos (drive anormal → comportamiento patológico).

→ Relevante para: allostasis en Γ, risk aversion como propiedad emergente, patologías de drive en multi-agente.

---

## Homeostasis + LLMs

### Li (2026)
**Metabolic Self-Organization: Emergence of Autonomous Agency in a Metabolically Constrained LLMs**
*bioRxiv 2026.05.13.724883*

Aplica un modelo metabólico a Qwen2.5-1.5B: cada token generado consume un presupuesto energético finito, cuantificado vía variational free energy (VFE). Feedback interoceptivo por el input stream. 7 experimentos.

Hallazgos: (i) feedback extiende supervivencia de ~20 a >31 pasos; ablación causa colapso en 13 pasos. (ii) estructura temporal importa más que magnitud de perturbación (OU noise 20.5 vs white noise 8.6 pasos, p≈10⁻¹¹). (iii) piso de compresión en ~3.2 nats. (iv) feedback desacopla VFE de energía (pendiente 0.0004 vs 0.0043), forzando frugalidad constante.

→ **Conexión directa con Pro-Action:** "existential vulnerability can catalyse synthetic agency". El drift λ es análogo al consumo metabólico — el agente gasta recursos solo por existir. La saciación α·g es análoga a la recarga de energía. La diferencia: Li usa VFE como señal, Pro-Action usa g estructural. ¿Se puede reformular g como costo termodinámico?

---

### Chase (2025, rev. 2026)
**Homeostatic Drive as Policy Precision: Understanding Biological Motivation Through Large Language Model Inference Architecture**
*ResearchGate*

Traduce motivación biológica al vocabulario de inferencia de LLMs:
- **Temperatura = precision**: el drive homeostático no "ordena" actuar, ajusta la precisión de las predicciones del agente, estrechando el espacio de acciones probables.
- **System prompt = contexto interoceptivo**: la sed no es una instrucción, es un contexto que sesga toda la inferencia.
- **Reward ≠ señal**: la recompensa es señal de entrenamiento (actualiza pesos), no parte de la inferencia en tiempo real.

→ **Conexión con Pro-Action:** el gate sigmoide σ(k(δ−θ)) es precisamente esto — δ modula la precisión de la decisión de actuar. No "ordena" hablar, ajusta la probabilidad. La temperatura del sampling del LLM podría acoplarse a δ: a mayor déficit, menor temperatura (más determinista, más urgente).

---

## Críticas y frameworks alternativos

### Rule-Relocation Problem (varios autores, 2025-2026)
**The Rule-Relocation Problem** — arXiv:2512.21000

Crítica a arquitecturas homeostáticas: el setpoint s* es una norma impuesta por el diseñador, no emerge del agente. "The Whistle has moved from the arbitrary function R to the fixed vector s*". La normatividad no se elimina, se reubica. Un sistema que no puede modificar su propio s* no puede adaptarse a condiciones cambiantes.

→ **Conexión con Pro-Action:** el Problema de la Señal de Saciedad es un caso especial de Rule-Relocation. g es la "norma reubicada". La pregunta no es solo cómo definir g, sino si el agente puede aprender a redefinir su propio g.

---

## Self-Regulated Learning (SRL)

### Tinajero, Mayo, Villar & Martínez-López (2024)
**Classic and modern models of self-regulated learning: integrative and componential analysis**
*Frontiers in Psychology, 15:1307574 · PMC10958659*

Revisión crítica de modelos clásicos y modernos de SRL. Analiza y compara los componentes y fases de Zimmerman, Boekaerts, Winne & Hadwin, y Pintrich.

Hallazgos clave:
- Evolución de **cold → hot self-regulation**: los modelos modernos integran lo afectivo/motivacional, no solo lo cognitivo.
- De actividad **consciente → implícita**: la autorregulación puede operar automáticamente, no solo deliberativamente.
- De funcionamiento **individual → interindividual**: la regulación ocurre entre agentes, no solo dentro de uno.
- **Ciclo prototípico**: forethought → performance → self-reflection (Zimmerman), con loops de avance y retroceso.
- Componentes transversales: conocimiento metacognitivo, habilidades metacognitivas, experiencias metacognitivas. El monitoring atraviesa todas las fases, no está confinado a una sola.

→ **Conexión con Pro-Action:** El ciclo SRL es isomorfo al lazo homeostático: forethought = acumulación de déficit (drift λ), performance = activación (gate σ) + acción, self-reflection = saciación (α·g). La transición cold→hot que documentan es exactamente lo que Γ operacionaliza: el drive δ es el componente "caliente" (afectivo/motivacional) que dispara la acción. La dimensión interindividual anticipa sistemas multi-agente donde los agentes se co-regulan. El hallazgo de que el monitoring es transversal (no confinado a una fase) sugiere que g podría computarse continuamente, no solo post-acción.

---

### Fleur, Bredeweg & van den Bos (2021)
**Metacognition: ideas and insights from neuro- and educational sciences**
*npj Science of Learning, 6:13 · PMC8187395*

Revisión interdisciplinaria de la metacognición desde las neurociencias cognitivas y las ciencias de la educación. Define metacognición como dos componentes: **metacognitive knowledge** (ser consciente de los propios procesos cognitivos) y **metacognitive control** (regularlos).

Distinciones clave:
- **Online vs. offline metacognition:**
  - *Online:* en el momento, mayormente automática, sin meta-representaciones explícitas. Asociada a executive function (EF).
  - *Offline:* más lenta, reflexiva, requiere meta-representaciones. Asociada a metacognitive judgements (ej. confidence ratings en tareas 2-AFC).
- **Neurociencia cognitiva:** se enfoca en juicios metacognitivos offline (meta-d', confidence) y EF online. Protocolos de laboratorio estrechos (2-AFC) — validez ecológica cuestionable.
- **Ciencias de la educación:** mide metacognición en contexto de aprendizaje real (cuestionarios como el MAI, learning journals, prompts metacognitivos). Énfasis en offline meta-control (planificación, regulación estratégica).
- **Gap identificado:** el offline meta-control está subestudiado en neurociencia, mientras que es central en educación.
- **Entrenamiento:** intervenciones que fomentan self-reflection y conocimiento del propio aprendizaje (offline meta-knowledge) son las que más benefician el rendimiento académico.

---

## Cibernética y agencia

### Pickering (2024)
**What Is Agency? A View from Science Studies and Cybernetics**
*Biological Theory, 19*

Agencia como "danza de agencia" universal — no restringida a organismos, sino inmanente en la materia. Islas de estabilidad: agentes como constelaciones transitorias que emergen del flujo. Agencia mínima (Okasha) extendida a sistemas no orgánicos. El termostato como ejemplo canónico: un ensamblaje de partes acopladas que cierra un lazo y constituye agencia propositiva de nivel superior. Reentry/recursión como transición de agencia sin propósito a agencia con propósito. Sistemas que aprenden con caminos inescrutables como ejemplos de normatividad emergente.

→ **Conexión con Pro-Action:** Γ es exactamente un mecanismo de "cierre de lazo" en el sentido de Pickering. El drift λ + gate σ + saciación α·g constituyen una isla de estabilidad homeostática. La agencia no está en el LLM ni en el prompt — emerge del acoplamiento entre partes (déficit, gate, acción, saciación). La pregunta de Pickering sobre normatividad emergente es la misma que el Problema de la Señal de Saciedad: ¿de dónde viene g?

---

### Wang, Yang, Zhao, Lin & Hu (2026)
**The Agent Use of Agent Beings: Agent Cybernetics Is the Missing Science of Foundation Agents**
*arXiv:2605.10754*

La cibernética como scaffold teórico para foundation agents. Mapean 6 leyes canónicas de cibernética clásica a 6 principios de diseño de agentes, sintetizados en 3 desiderata: **reliability, lifelong running, self-improvement**.

**Las 6 leyes → principios:**

| Ley cibernética | Principio de diseño |
|---|---|
| **Feedback Principle** (Wiener): u(t) = K(r(t) − y(t)), lazo cerrado corrige desviaciones | Closed-loop agent architecture: el output del agente realimenta su estado interno |
| **Law of Requisite Variety** (Ashby): V(O) ≥ V(D) − V(R), solo variedad destruye variedad | El agente debe tener al menos tanta capacidad de respuesta como variedad de perturbaciones del entorno |
| **Good Regulator Theorem** (Conant & Ashby): todo buen regulador contiene un modelo del sistema | El agente necesita un modelo interno del entorno (world model) |
| **Shannon's Channel Capacity**: comunicación confiable sobre canales ruidosos | El agente debe codificar información robusta frente a ruido (contexto largo, hallucination) |
| **Second-Order Cybernetics** (von Foerster): el observador en la descripción del sistema | El agente debe modelar su propio proceso de razonamiento (metacognición, self-reflection) |
| **Engineering Cybernetics** (Qian Xuesen): sistemas confiables desde componentes no confiables | El agente debe operar confiablemente a pesar de un LLM subyacente no determinista |



---

## Needs-Based AI — El linaje de The Sims

Inspiración fundacional del operador Pro-Action. The Sims (Will Wright, Maxis, 2000) implementó un sistema donde cada Sim tiene necesidades (hambre, energía, social, diversión, vejiga, higiene) que decaen con el tiempo y se satisfacen interactuando con objetos del entorno. La arquitectura clave: **objetos inteligentes, agentes "tontos"** — los objetos anuncian affordances (acciones + recompensa de necesidad), el Sim puntúa según sus necesidades actuales y elige la mejor.

### Zubek — Needs-Based AI (draft)
**Needs-Based AI**
*Robert Zubek, Northwestern University / EA/Maxis*

El overview técnico definitivo. Zubek trabajó en The Sims y reimplementó la arquitectura en otros juegos. Formaliza el algoritmo: (1) examinar objetos cercanos y sus affordances, (2) puntuar cada affordance según necesidades actuales, (3) elegir la mejor, (4) ejecutar la secuencia de acciones. Las necesidades decaen con el tiempo (`needs get worse and more urgent`). Emparenta el enfoque con behavior-with-activation-level action selection de robótica autónoma (Arkin 1998).

→ **Conexión con Pro-Action:** El decay de necesidades en The Sims es el drift λ. La puntuación de affordances es análoga a g (qué tan bien una acción satisface una necesidad). La diferencia: The Sims usa argmax sobre scores, Pro-Action usa gate sigmoide estocástico. The Sims tiene n necesidades independientes, Γ las acopla vía producto de sigmoides.

---

### Forbus & Wright — The Sims AI course notes
**Course notes on The Sims AI**
*Ken Forbus & Will Wright, Northwestern University*

Material original del curso donde Wright enseñó la arquitectura de The Sims. Referenciado por Zubek como una de las pocas fuentes sobrevivientes del diseño original. Cubre: Maslow's hierarchy como base de necesidades, smart objects con affordance broadcasting, distributed intelligence (herencia de SimAnt y sus rastros de feromonas).

---

### Wright — Diseño de sistemas emergentes
**Will Wright's design philosophy**

Tres influencias clave:
- **Jay Forrester** (systems dynamics, MIT): inspiración para SimCity. Sistemas complejos desde reglas locales simples.
- **Christopher Alexander** (A Pattern Language): gramática de diseño espacial.
- **SimAnt**: inteligencia distribuida — feromonas virtuales en el entorno, no en los agentes. "Could we build a more robust simulation of human behavior if we adopted this ant model, where we distribute the intelligence not through the agents, but through the environment?"

→ **Conexión con Pro-Action:** Γ invierte esto. La inteligencia SÍ está en el agente (drive, gate, saciación). El entorno (grafo de argumentos) es pasivo. The Sims distribuye la inteligencia en objetos; Pro-Action la concentra en el lazo homeostático del agente. Son arquitecturas opuestas.

---

### Arkin (1998) — Behavior-Based Robotics
**Behavior-Based Robotics**
*Ronald Arkin, MIT Press*

El linaje académico de needs-based AI. Acción seleccionada por nivel de activación de comportamientos competitivos. Cada comportamiento tiene un nivel de activación basado en necesidades internas y estímulos externos. La acción ganadora es la de mayor activación (o muestreo estocástico proporcional).

→ **Conexión con Pro-Action:** El gate sigmoide σ(k(δ−θ)) es un nivel de activación. La diferencia: Arkin usa competición entre comportamientos (gana el de mayor activación), Γ usa producto de gates (todos deben estar abiertos). Son dos formas de componer drives múltiples.

---

## Video

https://www.youtube.com/watch?v=-I0WYFjIuSU
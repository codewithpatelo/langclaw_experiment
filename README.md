# Activación Endógena vía Homeostasis Epistémica en Sistemas Multi-Agente basados en LLM

> Repositorio del trabajo experimental presentado como póster en la
> **Escuela Sudamericana de NLP 2026**. Contiene el código del
> benchmark, los resultados de 20 semillas pareadas, los scripts de
> análisis, el mock paper y el póster entregado.
>
> **Pregunta de investigación:** ¿la activación endógena (regulación
> homeostática) es más resiliente al colapso de contexto que el
> ruteo exógeno (LangGraph) en debate adversarial?

---

## Tabla de contenidos

1. [Resumen](#1-resumen)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Instalación](#3-instalación)
4. [Ejecución del benchmark](#4-ejecución-del-benchmark)
5. [Evaluación con jueces y análisis](#5-evaluación-con-jueces-y-análisis)
6. [Semillas y calibración](#6-semillas-y-calibración)
7. [Resultados](#7-resultados)
8. [Compilación del paper](#8-compilación-del-paper)
9. [Problemas frecuentes](#9-problemas-frecuentes)
10. [Uso de LLMs](#10-uso-de-llms)
11. [Citación](#11-citación)
12. [Licencia](#12-licencia)

---

## 1. Resumen

Los sistemas multi-agente basados en LLM tienden a perder coherencia
a medida que la interacción se extiende — un fenómeno conocido como
*colapso de contexto* que, en escenarios de debate competitivo,
invalida la deliberación. Los frameworks contemporáneos orquestan
los turnos exógenamente (ruteadores, colas, round-robin), sin un
mecanismo interno que regule cuándo un agente *debería* hablar.

Este trabajo propone un mecanismo de **activación endógena** basado
en homeostasis epistémica: cada agente mantiene un *déficit
epistémico* `δ_i` que crece durante la inactividad y se reduce tras
producir un aporte de calidad. Un gate sigmoide sobre `δ_i` determina
la probabilidad de activación. El mecanismo se formaliza mediante los
**Axiomas de Autonomía Homeostática (AAH-1–3)** y se implementa como
la **Ecuación Pro-Acción Reducida (EPR, Γ(n=1))**, sin aprendizaje
por refuerzo.

Se evalúa frente a un ruteador exógeno LangGraph en debates
adversariales de texto con **20 semillas pareadas**, **cuatro
condiciones** (EPR, EPR_Q, EPR_SHAM, LangGraph) y **jueces LLM duales
ciegos al modo de orquestación** (DeepSeek V4-Pro y GLM-5.2).

### Condiciones experimentales

- **EPR** (Γ(n=1)) — condición principal. Activación endógena vía
  gate sigmoide sobre `δ_i`. Sin Q-learning.
- **EPR_Q** — EPR + Q-learning (ablación: ¿aporta el Q-learner?).
- **EPR_SHAM** — sigmoide con `δ` aleatorio (ablación: ¿importa el
  loop homeostático o basta la forma del gate?).
- **LangGraph** — ruteo exógeno: un LLM decide qué agente habla en
  cada heartbeat.

### Organización de los agentes

10 agentes LLM en dos facciones opuestas (GOV/OPP) de 5 agentes cada
una. Cada agente mapea a un subsistema del Viable System Model
(S1–S5: operaciones, coordinación, control, inteligencia, estrategia).
Los agentes ejecutan un loop cognitivo con acciones `DEBATE`, `SEARCH`,
`READ`, `MESSAGE` y `PASS`, más mensajería dirigida tipo FIPA.

![Arquitectura del agente](langclaw/figs/illustration_1.png)

![Organización por facciones (VSM S1–S5)](langclaw/figs/illustration_2.png)

---

## 2. Estructura del repositorio

```text
├── langclaw/                       # Implementación del sistema
│   ├── homeostasis.py              # Drive epistémico: decaimiento, gate sigmoide, saciedad
│   ├── q_learner.py                # TD(0) lineal con normalización y clipping
│   ├── delp_graph.py               # Grafo de argumentos (AAF) + proxy Δφ*
│   ├── graph_query.py              # Consultas estructurales para acción SEARCH
│   ├── agent.py                    # Loop del agente, prompts, mensajería FIPA
│   ├── simulation.py               # Entorno, ambos modos de orquestación
│   ├── langgraph_flow.py           # Router LangGraph (baseline exógeno)
│   ├── router.py                   # Router LLM neutral
│   ├── router_informed.py          # Router informado con features estructurales
│   ├── memory.py                   # Memoria de tres capas (episódica/semántica/working)
│   ├── budget.py                   # Rate limits de API
│   ├── actions.py                  # Utilidades de acciones, StimulusEvaluator
│   ├── core_metric.py              # Métrica de coherencia temporal CORE
│   ├── metrics.py                  # PRR_G, IR, aceptación AAF, pendientes
│   ├── events.py                   # Eventos de tick/argumento/shutdown
│   ├── schemas.py                  # Schemas de logging (Pydantic)
│   ├── seeds.py                    # Fábrica determinista de semillas
│   └── figs/                       # Ilustraciones de arquitectura
├── benchmark.py                    # Benchmark multi-semilla (EPR vs LangGraph)
├── run_parallel.py                 # Runner paralelo del benchmark
├── run_collapse_judge.py           # Juez de colapso (LLM, ciego al modo)
├── run_structural_judge.py         # Juez estructural (LLM, ciego al modo)
├── analyze_collapse.py             # Análisis de colapso y estadística
├── analyze_judge_agreement.py      # Acuerdo entre jueces (κ de Cohen)
├── analyze_judge_collapse.py       # Análisis de colapso del juez
├── analyze_g_validity.py           # Análisis de validez de Δφ*
├── prepare_human_validation.py     # Preparación de muestras para validación humana
├── merge_results.py                # Consolidación de resultados multi-semilla
├── benchmark_results_v7/           # Resultados (20 semillas, 4 condiciones)
│   ├── benchmark_report.json       # Reporte consolidado
│   └── logs_{mode}_seed{N}.json    # Logs por corrida
├── collapse_judge_results.json     # Resultados del juez de colapso
├── collapse_judge_checkpoint.json  # Checkpoint del juez de colapso
├── recalibration_results.json      # Hiperparámetros calibrados
├── mock_paper.tex                  # Mock paper (formato LNCS)
├── mock_paper.pdf                  # PDF compilado
├── references.bib                  # Bibliografía
├── nlpschool_poster_gerpe.pdf      # Póster entregado en Escuela NLP 2026
├── requirements.txt                # Dependencias de Python
├── .env.example                    # Template de claves (DeepSeek + Z.AI)
├── LICENSE
└── README.md
```

---

## 3. Instalación

Probado en Python 3.11 sobre Windows 10/11 (PowerShell) y Linux.

```powershell
# 1) Crear y activar un entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell
# source .venv/bin/activate      # bash/zsh

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Configurar credenciales
copy .env.example .env           # cp .env.example .env en Linux
# Editar .env y completar las claves de DeepSeek (agentes) y Z.AI (juez GLM-5.2)

# 4) Verificación rápida (~2-3 minutos)
python benchmark.py --preflight --preflight-ticks 4 --seeds 7

# 5) Benchmark para una semilla (EPR + LangGraph)
python benchmark.py --iterations 80 --seeds 7 --modes hrrl langgraph
```

> El flag `--modes hrrl` activa la condición EPR (activación
> endógena). El nombre interno se mantiene por compatibilidad con
> los logs existentes.

Los resultados se guardan en `benchmark_results_v7/` (o el directorio
indicado con `--output-dir`).

---

## 4. Ejecución del benchmark

El benchmark ejecuta 10 agentes LLM (dos facciones de 5) en debate
adversarial durante un número fijo de heartbeats. Cuatro condiciones:

| modo (`--modes`) | fuente de activación              | descripción                |
|------------------|-----------------------------------|----------------------------|
| `hrrl`           | endógena (sigmoide sobre `δ_i`)   | EPR — condición principal  |
| `hrrl_q`         | endógena + Q-learning             | EPR_Q — ablación           |
| `hrrl_sham`      | sigmoide con `δ` aleatorio        | EPR_SHAM — ablación        |
| `langgraph`      | router LLM exógeno                | LangGraph — baseline       |

### Una semilla

```bash
python benchmark.py \
  --iterations 80 \
  --seeds 7 \
  --modes hrrl langgraph \
  --config recalibration_results.json \
  --api-hard-limit 500 \
  --output-dir benchmark_results_v7
```

### Evaluación completa (20 semillas, 4 condiciones)

Las semillas se corren en 4 lotes de 5: `{7, 17, 99, 123, 256}`,
`{1001, 2002, 3003, 4004, 5005}`, `{6006, 7007, 8008, 9009, 10010}`,
`{11011, 12012, 13013, 14014, 15015}`.

```bash
python run_parallel.py \
  --iterations 80 \
  --seeds 7 17 99 123 256 1001 2002 3003 4004 5005 \
            6006 7007 8008 9009 10010 11011 12012 13013 14014 15015 \
  --modes hrrl hrrl_q hrrl_sham langgraph \
  --config recalibration_results.json \
  --output-dir benchmark_results_v7
```

### Consolidación de resultados

```bash
python merge_results.py --results-dir benchmark_results_v7
```

Genera `benchmark_report.json` con métricas consolidadas.

### Análisis estadístico

Se aplican tests de Wilcoxon signed-rank con corrección de Bonferroni
(α = 0.0056, 9 tests) sobre las métricas consolidadas. Los intervalos
de confianza al 95% se calculan por bootstrap (10.000 resamples).

---

## 5. Evaluación con jueces y análisis

Dos jueces LLM (DeepSeek V4-Pro y GLM-5.2) evalúan la calidad de los
debates **ciegos al modo de orquestación**. Cada juez puntúa calidad
argumentativa, engagement y novedad (escala 1–5).

### Juez de colapso

```bash
python run_collapse_judge.py \
  --results-dir benchmark_results_v7 \
  --output collapse_judge_results.json
```

### Juez estructural

```bash
python run_structural_judge.py \
  --results-dir benchmark_results_v7
```

### Scripts de análisis

```bash
# Análisis de colapso (tasas, risk ratios, chi-cuadrado, tests de tendencia)
python analyze_collapse.py --results-dir benchmark_results_v7

# Acuerdo entre jueces (κ de Cohen)
python analyze_judge_agreement.py \
  --judge1 collapse_judge_results.json

# Análisis de validez de Δφ*
python analyze_g_validity.py --results-dir benchmark_results_v7

# Preparación de muestras para validación humana
python prepare_human_validation.py --results-dir benchmark_results_v7
```

---

## 6. Semillas y calibración

### Semillas

Cada fuente de aleatoriedad — `random` de Python, `numpy`, salt de
muestreo del LLM, asignación de IDs — se deriva deterministamente
desde una semilla maestra vía `langclaw/seeds.py`. La semilla `42` se
mantiene separada, exclusivamente para calibración de hiperparámetros,
y está **excluida de la evaluación** para evitar leakage.

### Calibración

El archivo `recalibration_results.json` contiene los hiperparámetros
utilizados en todos los experimentos reportados. Para recalibrar:

```bash
python benchmark.py --calibrate \
  --calibration-ticks 10 \
  --calibration-seed 42 \
  --api-hard-limit 200
```

Dos criterios *a priori* fijan la selección — densidad de debate en
el rango operativo y estabilidad del déficit alrededor del set-point —
y los valores se congelan para todas las semillas de evaluación.

### Framework estadístico

Todas las comparaciones entre condiciones usan el **Wilcoxon signed-rank
test pareado por semilla** (n = 20 pares). El pareo es válido porque las
cuatro condiciones (EPR, EPR\_Q, EPR\_SHAM, LangGraph) comparten el mismo
conjunto de semillas, controlando la variabilidad inter-semilla.

Con 3 comparaciones contra EPR (EPR\_Q, EPR\_SHAM, LangGraph) se aplica
**corrección de Bonferroni**: α = 0.05 / 3 = **0.0167**. Una diferencia
es significativa solo si p < 0.0167.

> **Nota metodológica:** los puntajes del juez LLM se promedian por
> semilla antes del test. Pooling de argumentos individuales (n ≈ 800–1200
> por condición) viola la independencia — los argumentos dentro de una
> semilla comparten contexto, agente y trayectoria — e infla
> artificialmente la significancia. La unidad de análisis para la
> inferencia es la semilla, no el argumento individual.

---

## 7. Resultados

Resultados consolidados (20/20 semillas, Wilcoxon signed-rank con
Bonferroni α = 0.0056) en `benchmark_results_v7/benchmark_report.json`:

| Métrica                    | EPR    | EPR_Q  | EPR_SHAM | LangGraph |
|----------------------------|--------|--------|----------|-----------|
| Debates totales            | 43.5   | 44.9   | 40.4     | 60.6      |
| Share máx. por agente      | 0.119  | 0.115  | 0.209*** | 0.276***  |
| Δφ* (media)                | 0.463  | 0.459  | 0.449    | 0.442***  |
| Déficit máx. (agente)      | 1.02   | 0.95   | 3.74***  | 2.54***   |
| Initiative ratio (IR)      | 1.00   | 1.00   | 1.00     | 0.00      |

> `***p < 0.001` (significativo tras Bonferroni). Detalles
> estadísticos completos en `mock_paper.tex`, Tabla 1.

**Hallazgos principales:**

- EPR distribuye los turnos **2.3×** más equitativamente que LangGraph
  (share máx. 0.119 vs 0.276, p < 0.001).
- EPR controla el déficit epistémico (1.02 vs 2.54, p < 0.001).
- La ablación sham confirma que el loop homeostático — no sólo la
  forma del gate — es lo que controla el déficit (3.74 vs 1.02,
  p < 0.001).
- El Q-learning no aporta beneficio claro y podría amplificar el
  colapso de contexto.
- Las pendientes temporales no alcanzan significancia, sugiriendo
  que el colapso de contexto es parcialmente endógeno al LLM.

---

## 8. Compilación del paper

El mock paper (`mock_paper.tex`) usa la clase LNCS con BibLaTeX y
Biber.

```bash
pdflatex mock_paper.tex
biber mock_paper
pdflatex mock_paper.tex
pdflatex mock_paper.tex
```

Produce `mock_paper.pdf`. La bibliografía está en `references.bib`.

---

## 9. Problemas frecuentes

| Síntoma                          | Causa probable / solución                                                                  |
|----------------------------------|--------------------------------------------------------------------------------------------|
| `AuthenticationError`            | Falta `.env` o `OPEN_AI_API_KEY` vacía; copiar de `.env.example` y setear la clave de DeepSeek. |
| HTTP 429 (rate limit)            | Esperar recuperación de cuota y re-ejecutar; el benchmark resume desde el último checkpoint. |
| `UnicodeEncodeError` en Windows  | Setear `PYTHONIOENCODING=utf-8` y `PYTHONUTF8=1` en el entorno.                            |
| Checkpoint stale tras cambio     | Pasar `--clean` a `benchmark.py` para descartar el checkpoint y empezar de cero.           |

---

## 10. Uso de LLMs

Todos los modelos de IA se utilizaron bajo supervisión humana
continua. El autor retiene responsabilidad total sobre el contenido.

### Modelos experimentales (parte del artefacto)

| Modelo                          | Rol                                                              |
|---------------------------------|------------------------------------------------------------------|
| **DeepSeek V4-Flash**           | Backbone de todos los agentes deliberativos, router LangGraph y módulos auxiliares |
| **DeepSeek V4-Pro**             | Juez LLM offline (ciego al modo de orquestación)                 |
| **GLM-5.2** (Z.AI)              | Juez LLM offline (ciego al modo de orquestación)                 |

### Herramientas de asistencia (no parte del experimento)

| Herramienta                    | Rol                                                              |
|--------------------------------|------------------------------------------------------------------|
| **GLM-5.2** (Z.AI) vía Devin (Windsurf) | Redacción de código, refactorización, debugging y asistencia en redacción del manuscrito |

---

## 11. Citación

```bibtex
@unpublished{gerpe2026epr,
  author       = {Patricio Gerpe},
  title        = {Endogenous Activation via Epistemic Homeostasis
                  in LLM-Based Multi-Agent Systems},
  year         = {2026},
  note         = {Mock paper y póster presentados en la Escuela
                  Sudamericana de NLP 2026, Argentina.}
}
```

---

## 12. Licencia

El código fuente se distribuye bajo los términos de [`LICENSE`](LICENSE).

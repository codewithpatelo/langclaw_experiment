# Code-Paper Audit (Fase 0A)

Auditoría de correspondencia estricta entre `paper_jaiio.tex` y la implementación
(`langclaw/*`, `calibrate_hyperparams.py`, `benchmark.py`). Cada fila reporta:
qué dice el paper, qué implementa el código, y veredicto.

## 1. Drive function y reward

| Concepto | Paper | Código | Veredicto |
|---|---|---|---|
| `D(δ) = (δ-ε)^m`, m=2 | Eq. 1 | `EpistemicDrive.drive_value`, `compute_reward` con `epsilon=0.1, m=2` | OK |
| `r_t = D(δ_t) - D(δ_{t+1})` | Eq. 6 | `compute_reward(d_before, d_after) = (d_before-ε)^m - (d_after-ε)^m` | OK |
| `δ_{t+1} = δ_t + λ` (decay) | Eq. 2 | `decay(lambda_rate=0.05)` | OK |
| `δ_{t+1} = δ_t + γ·r` (estímulo) | Eq. 3 | `stimulate(relevance, gamma=0.1)` | OK |
| `δ_{t+1} = max(ε, δ_t - α·Δφ*)` (saciedad) | Eq. 4 | `satiate(delta_phi, alpha)` con `BASELINE=0.1` | OK |
| `p = 1/(1+e^{-k(δ-θ)})`, k=10, θ=0.7 | Eq. 5 | `get_activation_probability(k=10.0, theta=0.7)` | OK |

## 2. Hiperparámetros (calibración seleccionada)

| Param | Paper | `calibration_results.json` | Código (default) | Inyección |
|---|---|---|---|---|
| α (DEBATE) | 3.0 | 3.0 (best) | 2.0 | `SotopiaEnvironment(debate_alpha=3.0)` |
| Pesos StimulusEvaluator | (0.40, 0.15, 0.15, 0.15, 0.15) faction-dominant | faction_dominant | 0.20 cada uno | `SotopiaEnvironment(stimulus_weights=...)` |
| Calibración: ticks | 10 | 10 | 15 (CLI default) | CLI `--ticks 10` |
| Calibración: seed | 42 | 42 | — | CLI `--seed 42` |

**Hallazgo (Reviewer-flagged data leakage):** la calibración usa seed=42 que también
está en el conjunto de evaluación. Confirmado en `calibration_results.json` y
`micro_sim_params.seed = 42`. Debe declararse honestamente en limitaciones.

## 3. Espacio de acciones (DISCREPANCIA ALTA)

| Aspecto | Paper | Código |
|---|---|---|
| Acciones declaradas | `{DEBATE, SEARCH, READ, MESSAGE, PASS}` (5) | LLM acepta esas mismas 5 (`AgentAction.action`) ✓ |
| Acciones del Q-learner | (no especifica) | `("DEBATE_STIMULUS", "DEBATE_PROACTIVE", "SEARCH", "READ", "MESSAGE")` — 5, **NO incluye PASS** |
| Mecanismo PASS en HRRL | implícito vía sigmoide | `if rng.random() >= activation_prob: return PASS` (línea 348 `agent.py`); el Q-learner NUNCA elige PASS |
| Sub-acciones DEBATE | DEBATE único | `DEBATE_STIMULUS` (responde estímulo) vs `DEBATE_PROACTIVE` (sin estímulo) |

**Acción correctiva paper:** aclarar que (a) PASS no es acción del Q-learner sino
estado de no-activación de la sigmoide, y (b) DEBATE se especializa en dos
modos según presencia de estímulo en buffer.

## 4. Estado φ(s) del Q-learner (DISCREPANCIA MEDIA)

| Aspecto | Paper | Código |
|---|---|---|
| Dimensión | 4: `[δ, δ², ρ_G, n_stim]` | 5: `[δ̂, δ̂², ρ̂_G, n̂_stim, n̂_msg]` |
| Normalización | no menciona | `δ̂ = clip((δ-0.1)/2.0, 0, 1)`, `n̂_stim = clip(n/5, 0,1)`, `n̂_msg = clip(n/5, 0,1)` |
| Quinta feature | ausente | `n_messages_received / 5` (presión de mensajes directos) |

**Acción correctiva paper:** actualizar Eq./descripción de features a 5 componentes
normalizadas; justificar la inclusión de `n̂_msg` como sensor de presión inter-agente.

## 5. Q-learning (DISCREPANCIA MEDIA)

| Aspecto | Paper | Código |
|---|---|---|
| Update TD(0) | `w_a += η[r + γ max Q' - Q]·φ` | OK estructura |
| η, γ, ε | η=0.01, γ=0.95, ε=0.1 | `eta=0.01, gamma=0.95, epsilon=0.1` ✓ |
| Inicialización | "ceros" | `np.zeros(N_FEATURES)` ✓ |
| Clipping TD-error | **no menciona** | `np.clip(td_error, -1.0, 1.0)` |
| Clipping pesos | **no menciona** | `np.clip(w, -5.0, 5.0)` |
| L2 reg | **no menciona** | `w *= (1 - η·λ_2)`, `λ_2 = 1e-4` |

**Acción correctiva paper:** agregar nota breve sobre estabilización numérica
(clipping TD ±1, clipping pesos ±5, L2 1e-4) en Sec. 3 o nota al pie. Reviewers
lo van a pedir y omitirlo es ocultar la adaptación.

## 6. Δφ* (proxy de calidad estructural)

| Aspecto | Paper | Código (`calculate_phi_star_proxy`) |
|---|---|---|
| Componentes y pesos | `(1/3) C_B(target) + (1/3) 1_cycle + (1/3) Div` | OK, 1/3 cada uno |
| C_B | betweenness centrality del nodo atacado | `nx.betweenness_centrality(g)[target]` ✓ |
| 1_cycle | bonus por ciclo dialéctico | `nx.shortest_path(target, node_id)` existe → 1.0 |
| Div | diversidad de agentes vecinos | `len(unique_agents) / total_agents_in_graph` |
| Range | [0,1] | `min(1, max(0, phi))` ✓ |

OK estructural. Reviewer-flag: "max-entropy weighting" — el código sí usa pesos
uniformes 1/3 (Jaynes 1957 es defensa razonable; el paper debe llamar a esto
"asignación uniforme por ausencia de prior" más que "máxima entropía").

## 7. AAF metrics (Dung 1995)

| Métrica | Paper | Código |
|---|---|---|
| Grounded extension | iteración de punto fijo | `_grounded_extension()` OK |
| `acceptance_ratio` | `α = |GE|/|A|` | OK |
| `defeat_cycle_count` | `|SCC_{>1}|` (Tarjan) | `nx.strongly_connected_components` OK |
| `dialectical_completeness` | `δ = |addressed|/|A|` | OK |

## 8. PRR_G (DEFINICIÓN OPERACIONAL FALTANTE EN PAPER)

| Aspecto | Paper | Código (`peer_reference_rate_graph`) |
|---|---|---|
| Cita | `\cite{anonymous2026madprotocols}` (anónima) | docstring cita "Marandi 2026, arXiv:2603.28813" en `peer_reference_rate` (text variant) |
| Definición | "rate at which agents reference peer claims" | `count(target_node_id != None) / count(action == DEBATE)` |
| Variantes | una sola | dos: `peer_reference_rate` (texto) y `peer_reference_rate_graph` (estructural) |

**Acción correctiva paper:** (a) eliminar referencia anónima, (b) dar la definición
operacional explícita "PRR_G = #DEBATEs con target ≠ ∅ / #DEBATEs", (c) si se
mantiene cita externa, citar la versión correcta (no "Marandi 2026" inventada).

## 9. Initiative Ratio (IR — tautológica)

| Aspecto | Paper | Código |
|---|---|---|
| Definición | "fraction of self-initiated turns" | `count(trigger=HOMEOSTATIC) / count(action != PASS)` |
| Valor esperado HRRL | ~1.0 | confirmado por construcción |
| Valor esperado LangGraph | ~0.0 | confirmado por construcción |

**Hallazgo (Reviewer-flagged):** IR es tautológica por diseño. El docstring del
código ya lo reconoce: "structural property of the orchestration design, not a
discourse quality outcome". El paper debe rebajar IR a "validity check" y NO
presentarlo como métrica de comparación.

## 10. LangGraph baseline (asimetría de información)

| Aspecto | Paper | Código (`router.py`, `_langgraph_single_tick`) |
|---|---|---|
| Prompt enrutador | "neutral, no menciona métricas" | confirmado: `_ROUTER_PROMPT` solo recibe `{context}` (texto últimos 6 args) y `{agent_ids}` |
| Política | no documentada | LLM elige speaker; fallback round-robin |
| Información que recibe el router | (no aclarado) | **SOLO TEXTO**: discourse_context (últimos 6 args como string) + lista de agent_ids. NO recibe centralidad, faccional, novedad, presión |
| Capacidades del agente seleccionado | "idénticas a HRRL" | confirmado: mismo `agent.step()`, mismo memoria, mismo StimulusEvaluator (sí calcula relevancia local), mismo cognitive graph |
| Asimetría real | ausente | la sigmoide HRRL recibe `δ` modulado por StimulusEvaluator (5 criterios estructurales); el router LangGraph recibe solo texto |

**Acción correctiva paper:** documentar literal el `_ROUTER_PROMPT` (en apéndice
o nota), reconocer que el router NO ve métricas estructuradas mientras HRRL
sí las internaliza vía StimulusEvaluator → deficit → sigmoide. Esta asimetría
es de diseño (corresponde al contraste exógeno vs endógeno), no un defecto,
pero debe explicitarse.

## 11. Ciclo cognitivo (THINK→PLAN→EXECUTE→OBSERVE)

| Fase | Paper | Código (`_cognitive_loop` agent.py) |
|---|---|---|
| TRIAGE | drenaje de buffer + StimulusEvaluator | OK líneas 290-300 |
| THINK | "evalúa presión, p<θ → PASS" | OK líneas 343-350; PASS si `rng.random() >= activation_prob` o presupuesto agotado |
| PLAN | Q-learner argmax ε-greedy | OK línea 365 (loop hasta MAX_COGNITIVE_LOOPS=3) |
| EXECUTE | `_execute_debate/_execute_message/_do_search/_do_read` | OK líneas 367-398 |
| OBSERVE | reward + TD update | OK líneas 401-416 |
| Disparador | "evento o tick" | el ciclo se ejecuta UNA vez por TickElapsedEvent recibido; NewArgumentEvent y DirectMessageEvent se BUFFEREAN sin disparar el ciclo (se procesan en el siguiente tick) |

**Hallazgo:** el paper afirma "ciclo cognitivo dirigido por evento", pero en el
código el ciclo se dispara por `TickElapsedEvent` (heartbeat), y los eventos
NewArgument/DirectMessage solo modulan el buffer. La activación es "informada
por evento" (vía StimulusEvaluator que actualiza el deficit) pero "disparada
por tick". El texto actual "se procesa por evento, no por pulso" es preciso
para la SELECCIÓN del estímulo, pero el ciclo en sí sigue siendo un heartbeat.
Conviene refinar el lenguaje a: "el heartbeat dispara el ciclo; cuando el ciclo
corre, el agente selecciona el evento más relevante de su buffer."

## 12. Configuración LLM (FALTANTE EN PAPER)

| Item | Paper | Código |
|---|---|---|
| Modelo | "GPT-5-nano" | `gpt-5-nano` ✓ |
| Temperatura | no menciona | no configurable (gpt-5-nano reasoning model, default=1) |
| `max_completion_tokens` | no menciona | 50000 (debate y análisis) |
| Timeout | no menciona | 60s |
| Retries | no menciona | MAX_RETRIES=4 → hasta 5 intentos |
| Backoff | no menciona | exponencial (5·2^a si rate-limit, 2·2^a si otro) |
| Loops cognitivos | no menciona | MAX_COGNITIVE_LOOPS=3 |
| Seed LLM | "seed por agente" | `llm_seed` por agente vía SeedFactory ✓ |
| Concurrencia | no menciona | `asyncio.Semaphore(2)` para llamadas LLM |

**Acción correctiva paper:** agregar tabla compacta o nota al pie con parámetros
de configuración LLM (modelo, max tokens, timeout, retries, backoff, semaphore).
Sin esto la reproducibilidad es imposible.

## 13. Seeds y reproducibilidad

| Item | Paper | Código |
|---|---|---|
| Seeds evaluación | {7, 17, 42, 123, 256} | confirmados en `experiment_status.json` y nombres de archivo |
| Seed master → primos | SeedFactory deriva primos por componente | OK `langclaw/seeds.py` (no auditado en detalle pero presente) |
| Componentes seedados | simulation, agent_X_rng, agent_X_llm, router_llm | confirmado en `simulation.py` líneas 234-272 |

## 14. VSM (10 agentes = 2 facciones × 5 subsistemas)

| Item | Paper | Código (`AGENT_ROLES`) |
|---|---|---|
| 10 agentes | sí | 10 entradas (5 GOV + 5 OPP) ✓ |
| 5 subsistemas S1-S5 | Operations, Coordination, Control, Intelligence, Strategy | OK ✓ |
| Prompts en español | sí | OK ✓ |
| Justificación tamaño | "unidad mínima viable" | el reviewer pide reificación: defender vía argumento de operacionalización (ver Fase 6) |

## 15. Decaimiento por tick — ¿qué agentes decaen?

| Modo | Paper | Código |
|---|---|---|
| HRRL | "todos decaen cada tick" | `decay()` se llama dentro del cognitive loop de CADA agente que recibe TickElapsedEvent (todos, líneas 287 agent.py) ✓ |
| LangGraph | "todos decaen cada tick" | `agent.drive.decay()` para CADA agente al inicio del tick (línea 551 simulation.py), antes de elegir speaker ✓ |
| Round-robin / random | (no comentado) | sólo decae el agente NO seleccionado (línea 828 simulation.py) — diferente comportamiento, pero esos modos no se usan en el experimento principal |

OK para los dos modos comparados (HRRL vs LangGraph).

## 16. Conteo de debates (volume confound)

| Item | Paper | Código |
|---|---|---|
| Reporte | "HRRL ~5.5× más debates que LangGraph" | a verificar con análisis Fase 0B |
| Causa | en HRRL muchos agentes pueden activarse el mismo tick; en LangGraph solo 1 router-selected | confirmado: `_run_hrrl` permite N agentes activos/tick; `_langgraph_single_tick` activa SOLO el speaker seleccionado |
| Consecuencia | más debates → más densidad → posible sesgo en AAF/PRR_G | a cuantificar Fase 0B (volume-matched) |

## Resumen de discrepancias accionables

**Críticas (deben arreglarse en paper):**
1. Espacio de acciones del Q-learner (5 sin PASS, dos sub-debates).
2. Estado φ(s) tiene 5 features normalizadas, no 4.
3. PRR_G: definir operacionalmente y eliminar referencia inventada.
4. Asimetría LangGraph baseline: el router solo recibe texto.
5. Estabilización numérica del Q-learner (clipping + L2): mencionar.

**Importantes (mejoran rigor):**
6. Configuración LLM: tabla compacta de parámetros operativos.
7. IR como validity check, no comparación.
8. Pesos Δφ*: "asignación uniforme por ausencia de prior", no "máxima entropía".
9. Refinar lenguaje del ciclo cognitivo: heartbeat dispara, evento prioriza.

**Reconocer en limitaciones:**
10. Data leakage: seed=42 en calibración y evaluación.
11. Volume confound: complementar con análisis volume-matched (Fase 0B).

# Mathematical Notes — Factor-Gated Routing (FGR)

## Motivation

Factor-Gated Routing (FGR) decomposes a diffusion model into $K$ per-factor streams, each processing a dedicated factor embedding $e_i$. Cross-stream edges form a directed acyclic graph (DAG) $G = (V, E)$ where $V = \{1, \ldots, K\}$ and $E \subseteq V \times V$. Each stream $i$ produces a contribution $h_i$ that is modulated by a scalar gate $g_i \in [0, 1]$ before aggregation. Inference-time gating sets $g_i = 0$ to suppress factor $i$, analogous to a do-operator. These propositions provide architectural motivation; they are not formal theorems.

---

## Proposition 1 (Gate Isolation)

Let $\epsilon_\theta$ denote an FGR denoiser with $K$ streams. For any stream $i$, the factor embedding $e_i$ enters the model **only** through stream $i$'s initial transformation and any cross-stream edges where stream $i$ appears as a parent (i.e., $i \to j$ for some $j$). If $g_i = 0$ and no edge $i \to j$ exists in the DAG (i.e., stream $i$ has no children), then the gradient of the denoiser output with respect to $e_i$ is zero everywhere:

$$\frac{\partial \epsilon_\theta(x_t, t, \{e_k\}_{k=1}^K)}{\partial e_i} = 0.$$

When $g_i = 0$ and cross-stream edges $i \to j$ do exist, stream $i$'s contribution propagates only through the attention weights of those child streams, but the multiplicative gate zeroes the stream output $g_i \cdot h_i$ before it reaches the final aggregation. Provided the aggregation is linear in each $h_i$, factor $i$ contributes exactly zero to the predicted noise $\epsilon_\theta$.

More generally, for any downstream quantity $Q$ that depends on the generated sample, the conditional expectation conditioned on factor $i$ under gate $g_i = 0$ is identical to the expectation under a do-intervention $\text{do}(f_i = \text{dropped})$:

$$\mathbb{E}[Q \mid g_i = 0] = \mathbb{E}[Q \mid \text{do}(f_i)].$$

**Interpretation.** Gate isolation is what distinguishes FGR gating from post-hoc manipulations such as classifier-free guidance (CFG): the gate directly zeroes the sole information pathway for factor $i$, rather than subtracting a scaled conditional from an unconditional distribution.

---

## Proposition 2 (Structural Disentanglement Bias)

Consider two architectures for $K$-factor conditional generation:

1. **FGR**: $K$ independent streams with DAG-structured cross-attention and per-stream gates $\{g_i\}$.
2. **Single-stream DiT**: A monolithic transformer that receives concatenated factor embeddings $[e_1 \| \cdots \| e_K]$ and is trained with CFG.

In the single-stream model, factor information is entangled across all attention heads from the first layer onward. To suppress factor $i$ at inference time, the model must rely on CFG:

$$\epsilon_\theta^{\text{CFG}}(x_t, t, e) = \epsilon_\theta(x_t, t, \varnothing) + w\big(\epsilon_\theta(x_t, t, e) - \epsilon_\theta(x_t, t, \varnothing_i)\big),$$

where $\varnothing_i$ denotes factor $i$ dropped and the remaining factors retained. This subtraction **confounds** factor $i$ with higher-order interactions stored in the joint embedding space — the guidance direction $\epsilon_\theta(x_t, t, e) - \epsilon_\theta(x_t, t, \varnothing_i)$ is not a pure factor-$i$ signal.

FGR, by contrast, allocates a dedicated computational pathway per factor. The structural prior that factor $i$ flows through stream $i$ (and only through stream $i$, absent cross-stream edges) creates a parameterization where learning factor-specific representations requires less data and fewer optimization steps than recovering the same structure from a fully entangled representation. Informally, the FGR parameterization reduces the effective sample complexity for learning factor-disentangled representations relative to a single-stream model of comparable capacity.

---

## Proposition 3 (DAG Routing Correctness)

Let the true generative process admit a partial causal order $\prec$ over factors $f_1, \ldots, f_K$ such that factor $f_j$ can depend causally on $f_i$ only if $i \prec j$. Construct a DAG $G = (V, E)$ where $E = \{(i, j) \mid i \prec j \text{ and } \text{Pa}(j)_{\text{true}} \text{ includes } i\}$.

In FGR with cross-stream attention following $G$, the information reaching stream $j$ is a function of $\{e_i \mid \text{there exists a directed path } i \rightsquigarrow j \text{ in } G\}$. Because $G$ is a DAG respecting $\prec$, no directed cycle exists: a stream never receives information about a downstream factor, preventing the model from "peeking" at consequences before inferring causes.

In a single-stream model, all-to-all self-attention allows any token pair to attend regardless of causal ordering. While the model *can* in principle learn to mask future-looking attention patterns, this must be learned from data rather than being enforced by architecture. The FGR DAG removes this degree of freedom, encoding the known causal ordering as a hard structural constraint.

**Corollary.** If the true DAG is unknown, FGR can be configured with a supergraph of the true DAG and the model will learn to attend only along necessary edges, but never along edges that violate a correct DAG that is known a priori.

---

## Proposition 4 (Gating Monotonicity)

Let $g_i$ be the scalar gate modulating stream $i$'s output: the stream contribution to the denoiser is $g_i \cdot h_i(x_t, t)$, with $g_i \in [0, 1]$. Define the **factor effect** as the expected change in the generated sample when varying $g_i$ from 0 to 1, holding all other gates fixed:

$$\Delta_i(g_i) = \mathbb{E}\big[\,\| x_0(g_i) - x_0(0) \|^2 \,\big],$$

where $x_0(g_i)$ denotes the ODE/SDE-integrated sample with gate $g_i$.

If the learned stream $h_i$ captures factor $i$'s variation non-degenerately (i.e., $\mathbb{E}[\|h_i\|^2] > 0$), and the integration scheme is Lipschitz-continuous in the denoiser, then $\Delta_i(g_i)$ is monotonic in $g_i$:

$$0 = \Delta_i(0) \leq \Delta_i(g_i) \leq \Delta_i(g_i') \leq \Delta_i(1) \quad \text{for } 0 \leq g_i \leq g_i' \leq 1.$$

**Testable consequence.** For an oracle that measures the semantic change attributable to factor $i$, the observed change magnitude should increase monotonically with $g_i$. Deviations from monotonicity indicate either (1) cross-stream attention leaking factor $i$ through child streams, (2) the model failing to learn a disentangled stream representation, or (3) nonlinear interactions between factor $i$ and the learned manifold geometry. This makes gating monotonicity a useful diagnostic.

---

> **Note.** These are architectural design justifications, not formal theorems. Formal proofs would require additional assumptions about the data distribution and optimization landscape.

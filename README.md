# Adaptive Unified Retrieval & Reasoning for RAG

> A production-oriented Agentic RAG system built with LangGraph that dynamically routes queries between local knowledge retrieval and web search, validates retrieval and generated answers, detects hallucinations, applies security controls, and provides evaluation and observability.

---

## Abstract

Conventional Retrieval-Augmented Generation (RAG) follows a fixed *retrieve → read → answer* pipeline. That design has three well-known weaknesses: (1) it retrieves even when retrieval is unnecessary or when the knowledge base cannot contain the answer, (2) it passes whatever was retrieved to the generator without checking whether it is relevant, and (3) it returns the first generated answer without checking whether the answer is grounded in the evidence or actually responsive to the question.

**Adaptive Unified Retrieval & Reasoning for RAG** replaces the fixed pipeline with a stateful control loop implemented as a LangGraph state machine. The system (a) *routes* each question to the most appropriate knowledge source, (b) *grades* retrieved evidence before using it, (c) *falls back* to web search (optionally gated by human approval) when local evidence is insufficient, (d) *verifies* the generated answer along two independent axes, **groundedness** and **answer relevance**, and (e) *retries* generation when either gate fails. A dedicated security layer treats all documents and web content as untrusted data, and an evaluation and observability layer records how each answer was produced.

This document describes the system at two levels: a practical level (installation, deployment, API) and a design level (formal workflow, state model, control-flow guarantees, threat model, evaluation methodology, limitations).

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation and Problem Statement](#motivation-and-problem-statement)
3. [Background and Related Work](#background-and-related-work)
4. [Key Features](#key-features)
5. [System Architecture](#system-architecture)
6. [Formal Workflow Specification](#formal-workflow-specification)
7. [Core Workflow Components](#core-workflow-components)
8. [Retrieval Layer](#retrieval-layer)
9. [Quality Gates: LLM-as-Judge Design](#quality-gates-llm-as-judge-design)
10. [Human-in-the-Loop](#human-in-the-loop)
11. [Security](#security)
12. [Evaluation Methodology](#evaluation-methodology)
13. [Observability](#observability)
14. [Cost and Latency Analysis](#cost-and-latency-analysis)
15. [Failure Modes and Limitations](#failure-modes-and-limitations)
16. [Technology Stack](#technology-stack)
17. [Project Structure](#project-structure)
18. [Installation](#installation)
19. [Running Locally](#running-locally)
20. [Running with Docker](#running-with-docker)
21. [API](#api)
22. [Testing](#testing)
23. [Example Workflow](#example-workflow)
24. [Why This Project Is Agentic](#why-this-project-is-agentic)
25. [Engineering Highlights](#engineering-highlights)
26. [Future Work](#future-work)
27. [Project Status](#project-status)

---

## Overview

**Adaptive Unified Retrieval & Reasoning for RAG** is an end-to-end Retrieval-Augmented Generation system designed around an agentic workflow rather than a fixed retrieval pipeline.

Instead of always performing vector search, the system evaluates the user's question and dynamically determines the appropriate path:

- Retrieve information from uploaded documents
- Fall back to web search when local knowledge is insufficient
- Generate an answer from trusted context
- Evaluate whether retrieved documents are relevant
- Check whether the generated answer is grounded
- Verify whether the answer actually addresses the question
- Retry generation when quality gates fail
- Record evaluation and observability information for each run
- Apply security controls against prompt injection and sensitive information leakage
- Support human approval before web search when required

The project includes a **FastAPI backend**, **Next.js frontend**, **LangGraph agent workflow**, vector-based retrieval, LLM-based grading, security controls, evaluation tracking, observability, and Docker deployment.

---

## Motivation and Problem Statement

A naive RAG pipeline can be written as a composition of three functions:

```text
answer = G( q, R(q) )
```

where `q` is the user question, `R` is a retriever returning the top-*k* chunks from a vector index, and `G` is a generator conditioned on the question and retrieved chunks. This composition has several structural failure modes:

| # | Failure mode | Description | Consequence |
|---|---|---|---|
| F1 | **Unnecessary retrieval** | Retrieval is executed for questions that need no external evidence (or need a different source). | Wasted latency and cost; irrelevant context can degrade answers. |
| F2 | **Retrieval miss** | The knowledge base does not contain the answer, but top-*k* search always returns *something*. | The generator is handed plausible-looking but irrelevant context. |
| F3 | **Context contamination** | Some retrieved chunks are relevant and others are noise. | Noise dilutes or misleads generation. |
| F4 | **Ungrounded generation** | The model asserts facts that are not supported by the provided context. | Hallucination presented as sourced fact. |
| F5 | **Non-responsive generation** | The answer is fluent and grounded but does not address the question asked. | User receives a correct-looking but useless response. |
| F6 | **Prompt injection via content** | Documents or web pages contain instructions aimed at the model. | Agent hijacking, data exfiltration, policy bypass. |
| F7 | **Opacity** | No record of which path produced the answer. | Debugging and quality analysis are difficult. |

This project addresses each failure mode with an explicit mechanism:

| Failure mode | Mitigation in this system |
|---|---|
| F1 Unnecessary retrieval | Query router selects the knowledge source per question. |
| F2 Retrieval miss | Retrieval grader + web-search fallback (optionally human-approved). |
| F3 Context contamination | Per-document relevance grading filters irrelevant chunks before generation. |
| F4 Ungrounded generation | Hallucination (groundedness) grader with retry. |
| F5 Non-responsive generation | Answer grader with retry. |
| F6 Prompt injection | Prompt/document guards, untrusted-content wrapping, output guard. |
| F7 Opacity | Observability records for every run; persisted evaluation history. |

---

## Background and Related Work

The design draws on a body of work on retrieval-augmented and self-reflective language-model systems. The descriptions below summarize the *ideas* this project builds on; see [References](#references) for the full citations.

- **Retrieval-Augmented Generation (Lewis et al., 2020).** Introduced the formulation of conditioning a generator on passages retrieved from an external non-parametric memory, separating *what the model knows* from *what it can look up*. This project uses the same core principle with a dense-vector retriever and a prompted LLM generator.
- **Self-RAG (Asai et al., 2023).** Proposed training a model to decide *when* to retrieve and to critique its own retrieved passages and generations. This project adopts the *behavioral* idea (adaptive retrieval and self-critique) but implements it with **external LLM graders orchestrated by a graph**, so no fine-tuning is required and each judgment is independently inspectable.
- **Corrective RAG (CRAG; Yan et al., 2024).** Introduced a retrieval evaluator that assesses the quality of retrieved documents and triggers corrective actions, including web search, when retrieval is judged unreliable. The retrieval-grader → web-search fallback path here follows this pattern.
- **Adaptive-RAG (Jeong et al., 2024).** Argued for selecting a retrieval strategy based on query complexity instead of applying one strategy to all queries. The query router in this project plays the analogous role, choosing between local retrieval and web search.
- **ReAct (Yao et al., 2022).** Demonstrated interleaving reasoning with tool use. The graph-based control loop here is a constrained, explicitly specified variant: the set of allowed transitions is fixed at design time rather than left to free-form model decisions, which makes behavior testable.
- **RAGAS (Es et al., 2023).** Provided reference-free metrics for RAG evaluation (faithfulness, answer relevancy, context precision/recall). The `evaluation/ragas/` component of this project is organized around these metric families.
- **Indirect Prompt Injection (Greshake et al., 2023) and prompt-injection research (Perez & Ribeiro, 2022).** Showed that instructions embedded in retrieved content can hijack LLM-integrated applications. This motivates treating all retrieved and uploaded text as untrusted *data*.
- **Lost in the Middle (Liu et al., 2023).** Showed that models use information placed in the middle of long contexts less reliably. This motivates filtering context to relevant chunks rather than concatenating everything retrieved.
- **OWASP Top 10 for LLM Applications.** Used as a checklist for the security design (prompt injection, sensitive information disclosure, improper output handling, excessive agency, unbounded consumption).

**Positioning.** The contribution of this project is primarily *systems engineering*: combining routing, corrective retrieval, dual-axis answer verification, bounded retry, human approval, a security layer, evaluation, and observability in one tested, deployable application. It does not claim a new retrieval or training algorithm.

---

## Key Features

### Agentic Adaptive Retrieval

The system does not blindly retrieve documents for every query.

It dynamically routes requests based on the question:

```text
                    User Question
                         │
                         ▼
                  Security Checks
                         │
                         ▼
                   Query Router
                    /         \
                   /           \
                  ▼             ▼
          Local Knowledge     Web Search
             Retrieval        (when needed)
                  │             │
                  ▼             ▼
           Retrieval Grader     │
                  │             │
          ┌───────┴───────┐     │
          │               │     │
       Relevant        Not Relevant
          │               │     │
          │               └─────┘
          │                   │
          └─────────┬─────────┘
                    ▼
                Generation
                    │
                    ▼
          Hallucination Grader
                    │
                    ▼
              Answer Grader
                    │
              ┌─────┴─────┐
              │           │
             Pass        Fail
              │           │
              ▼           ▼
           Response     Retry
                         │
                         └──────► Generation
```

### Feature Summary

| Capability | What it does | Failure mode addressed |
|---|---|---|
| Adaptive routing | Chooses local retrieval vs. web search per question | F1 |
| Retrieval grading | Scores retrieved chunks for relevance before generation | F2, F3 |
| Web-search fallback | Obtains external evidence when local evidence is insufficient | F2 |
| Hallucination grading | Checks the answer is supported by the context | F4 |
| Answer grading | Checks the answer addresses the question | F5 |
| Bounded retry | Regenerates on gate failure, with a retry counter | F4, F5 |
| Human approval | Gates web-search execution | Excessive agency |
| Security layer | Input, content, and output controls | F6 |
| Evaluation + observability | Persisted metrics and per-run traces | F7 |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                         Next.js UI                           │
│                                                              │
│  Chat • Document Upload • Evaluation • Observability         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                         │
│                                                              │
│  API • Uploads • Chat • Evaluation • Observability           │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       Security Layer                          │
│                                                              │
│  Prompt Injection • Jailbreak Detection • Secret Redaction  │
│  Document Injection Detection • Output Protection           │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        LangGraph                              │
│                                                              │
│  Router → Retrieve → Grade → Generate → Validate → Retry     │
│                 │                                            │
│                 └──────────────► Web Search                   │
└───────────────┬───────────────────────────────┬──────────────┘
                │                               │
                ▼                               ▼
       Vector Store / RAG                  Web Search
                │                               │
                └───────────────┬───────────────┘
                                ▼
                           LLM / Groq
                                │
                                ▼
                    Evaluation & Observability
```

### Layer Responsibilities

| Layer | Responsibility | Main modules |
|---|---|---|
| Presentation | Chat, upload, evaluation and observability views | `frontend/` |
| API | HTTP interface, validation, upload handling, orchestration entry point | `backend/api.py`, `backend/main.py` |
| Security | Input screening, untrusted-content handling, output redaction | `backend/security/` |
| Orchestration | Stateful graph: routing, grading, generation, retry | `backend/graph/` |
| Retrieval | Ingestion, chunking, embedding, vector search | `backend/ingestion.py`, `backend/retrieval.py` |
| Generation / judging | Prompted LLM chains (generator and graders) | `backend/graph/chains/`, `backend/model.py` |
| Response assembly | Final response, source attribution | `backend/response.py`, `backend/sources.py` |
| Measurement | Evaluation and run-level telemetry | `backend/evaluation.py`, `backend/observability.py` |

### Design Principles

1. **Explicit control flow.** Every allowed transition is declared in the graph. The LLM produces *judgments* (route, relevance, grounded, responsive); the graph decides what happens next. This keeps behavior testable.
2. **Separation of trusted and untrusted text.** Application instructions are trusted; documents, web pages, and user text are data.
3. **Defense in depth.** Security checks exist at input, content, and output stages; no single control is assumed sufficient.
4. **Bounded autonomy.** Retries are counted and capped, and external actions (web search) can require human approval.
5. **Measurability.** Every run emits structured telemetry, and evaluation is a first-class component rather than an afterthought.

---

## Formal Workflow Specification

### Notation

| Symbol | Meaning |
|---|---|
| `q` | User question |
| `D = {d₁ … d_k}` | Chunks returned by local retrieval |
| `D' ⊆ D` | Chunks judged relevant by the retrieval grader |
| `W` | Web-search results |
| `C` | Context passed to the generator (`C = D'`, or `W`, or a combination, depending on path) |
| `a` | Generated answer |
| `r ∈ {0, 1, …, R_max}` | Retry counter |
| `s_route ∈ {local, web}` | Router decision |
| `s_ground ∈ {grounded, not grounded}` | Hallucination grader output |
| `s_ans ∈ {useful, not useful}` | Answer grader output |

### Conceptual Graph State

The LangGraph state object carries everything that nodes need to read and write. The following is a *conceptual* description; consult `backend/graph/state.py` for the authoritative field names.

```text
State
├── question            : the (security-checked) user question
├── route               : router decision (local | web)
├── documents           : retrieved / filtered chunks used as context
├── web_search_needed   : flag set by the retrieval grader or router
├── web_search_approved : flag set by the human-approval step
├── generation          : current candidate answer
├── retry_count         : number of regeneration attempts so far
└── (run metadata)      : latency, source counts, grader outputs, HITL usage
```

### Nodes

| Node | File | Input | Output | Description |
|---|---|---|---|---|
| Router (entry decision) | `chains/router.py` | `q` | `s_route` | Decides local knowledge vs. web search. |
| Retrieve | `nodes/retrieve.py` | `q` | `D` | Vector search over the local knowledge base. |
| Grade documents | `nodes/grade_documents.py` | `q, D` | `D'`, fallback flag | Per-chunk relevance judgment using the retrieval grader. |
| Web-search approval | `nodes/web_search_approval.py` | `q` | approve / reject | Optional human approval before external retrieval. |
| Web search | `nodes/web_search.py` | `q` | `W` | External retrieval. |
| Web-search rejected | `nodes/web_search_rejected.py` | – | rejection response | Terminal path when approval is denied. |
| Generate | `nodes/generate.py` | `q, C` | `a` | Context-conditioned answer generation. |
| Increment retry | `nodes/increment_retry.py` | `r` | `r + 1` | Retry bookkeeping. |
| Build response | `nodes/build_response.py` | state | final response | Assembles the answer and sources. |

### Conditional Edges (Decision Table)

| After node | Condition | Next node |
|---|---|---|
| Router | `s_route = local` | Retrieve |
| Router | `s_route = web` | Web-search approval (if enabled) → Web search |
| Grade documents | `D'` non-empty | Generate |
| Grade documents | `D'` empty / insufficient | Web-search approval (if enabled) → Web search |
| Web-search approval | approved | Web search |
| Web-search approval | rejected | Web-search rejected → End |
| Web search | results returned | Generate |
| Generate → hallucination grader | `not grounded`, `r < R_max` | Increment retry → Generate |
| Generate → answer grader | `not useful`, `r < R_max` | Increment retry → Generate |
| Either gate | both pass | Build response |
| Either gate | fail and `r = R_max` | Build response (terminates; see below) |

### Algorithm (Pseudocode)

```text
function ANSWER(q):
    q ← SECURITY_INPUT_CHECK(q)                       # reject / sanitize
    route ← ROUTER(q)

    if route = local:
        D  ← RETRIEVE(q)
        D' ← { d ∈ D : RETRIEVAL_GRADER(q, d) = relevant }
        if D' = ∅:
            route ← web
        else:
            C ← WRAP_UNTRUSTED(D')

    if route = web:
        if APPROVAL_REQUIRED and not HUMAN_APPROVES(q):
            return REJECTION_RESPONSE
        W ← WEB_SEARCH(q)
        C ← WRAP_UNTRUSTED(W)

    r ← 0
    loop:
        a ← GENERATE(q, C)
        grounded ← HALLUCINATION_GRADER(a, C)
        useful   ← ANSWER_GRADER(q, a)
        if grounded and useful:               break
        if r = R_max:                         break   # bounded retry
        r ← r + 1

    a ← SECURITY_OUTPUT_GUARD(a)              # redact secrets, cap length
    RECORD_TELEMETRY(route, |D|, grounded, useful, r, latency, sources, hitl)
    return BUILD_RESPONSE(a, sources)
```

### Termination and Safety Properties

- **Bounded loops.** The only cycle in the graph is *Generate → Validate → Retry → Generate*, and it is governed by a retry counter with an upper bound `R_max`. Therefore every run terminates in a finite number of steps.
- **Single-fallback structure.** Web search is reached at most once per run in the forward direction; there is no edge from generation back to retrieval.
- **No unreviewed external action.** When approval is enabled, the web-search node is reachable only through the approval node.
- **Output always passes the output guard.** The response builder is the single exit point for answers.
- **Behavior on exhausted retries.** The system returns a terminating response instead of looping. Operators should decide, per deployment, whether an answer that failed a gate after `R_max` retries is returned with a warning or replaced by an explicit "could not verify" message.

### Worst-Case LLM Call Count

Let `k` be the number of retrieved chunks graded individually and `R_max` the retry cap:

```text
calls_max ≈ 1 (router)
          + k (retrieval grading, if graded per chunk)
          + (R_max + 1) × 3 (generate + hallucination grade + answer grade)
```

This bound makes the latency and cost envelope explicit, and it is the reason `R_max` and `k` are exposed as tunable parameters.

---

## Core Workflow Components

### 1. Query Routing

The system first determines whether the question can be answered using the available local knowledge or requires external information.

```text
Question
   │
   ▼
Router
   │
   ├── Local knowledge
   │
   └── Web search
```

This prevents unnecessary web searches when relevant uploaded knowledge is available.

**Design notes.**

- The router is a prompted LLM classifier producing a constrained label (local vs. web), not free text.
- Routing is a *prior*, not a guarantee. The retrieval grader provides a second, evidence-based chance to change course, so a routing error toward `local` is recoverable.
- Typical questions routed to web search are time-sensitive or clearly outside the domain of the uploaded documents; typical questions routed to local retrieval refer to uploaded material.

### 2. Local Document Retrieval

Users can provide their own documents rather than relying on hardcoded knowledge sources.

Supported document formats include:

- PDF
- DOCX
- TXT
- Markdown

Documents are processed and added to the application's vector-based retrieval layer. See [Retrieval Layer](#retrieval-layer) for ingestion details.

### 3. Retrieval Grading

Retrieved documents are evaluated for relevance to the user's question.

```text
Question + Retrieved Document
            │
            ▼
      Retrieval Grader
            │
       ┌────┴────┐
       ▼         ▼
     Relevant   Irrelevant
       │         │
       ▼         ▼
   Generation  Web Search
```

This allows the system to recognize when vector retrieval does not provide useful evidence.

**Why grade retrieval at all?** Dense-vector search always returns a nearest neighbor set, even if no neighbor is truly relevant. Similarity score alone is a weak relevance signal because it is not calibrated across queries and corpora. An explicit judgment (`question`, `chunk` → relevant / not relevant) provides a decision that is interpretable, independently testable, and robust to embedding-space idiosyncrasies. It also reduces context noise before generation (see *Lost in the Middle*).

### 4. Web Search Fallback

When local retrieval is insufficient, the workflow can route the request to web search.

```text
Local Knowledge
      │
      ├── Sufficient ──► Generate
      │
      └── Insufficient
               │
               ▼
          Web Search
               │
               ▼
            Generate
```

The project also supports human approval around web-search execution. Web results are wrapped as untrusted content before being passed to the generator (see [Security](#security)).

### 5. Hallucination Detection

Generated answers are checked against their available context.

```text
Generated Answer
       │
       ▼
Hallucination Grader
       │
   ┌───┴───┐
   ▼       ▼
Grounded  Not Grounded
   │          │
   ▼          ▼
 Continue    Retry
```

This provides a quality-control layer between generation and the final response.

**Definition used.** An answer is *grounded* if its factual claims are supported by the context `C` supplied to the generator. This is a **faithfulness** criterion: it measures consistency with the provided evidence, **not** real-world truth. A grounded answer can still be wrong if the evidence is wrong or outdated.

### 6. Answer Quality Validation

The system also evaluates whether the generated answer actually addresses the original question.

```text
Question + Answer
       │
       ▼
 Answer Grader
       │
   ┌───┴───┐
   ▼       ▼
   Yes      No
   │         │
   ▼         ▼
Response    Retry
```

Generation therefore has multiple quality gates rather than simply returning the first LLM response.

**Why two gates?** Groundedness and responsiveness are orthogonal:

| | Responsive | Not responsive |
|---|---|---|
| **Grounded** | Desired outcome | Faithful but off-topic (e.g., summarizes the context instead of answering) |
| **Not grounded** | Plausible but fabricated | Worst case |

A single combined check conflates these cases. Separate gates give separate diagnostics and separate evaluation metrics.

---

## Retrieval Layer

### Ingestion Pipeline

```text
Upload → Validate → Extract text → Chunk → Embed → Index (Chroma)
```

| Stage | Purpose | Notes |
|---|---|---|
| Validate | Enforce supported formats and `MAX_UPLOAD_SIZE_MB` | Defense against oversized or unsupported uploads. |
| Extract | Parse PDF / DOCX / TXT / Markdown to text | Extraction quality bounds downstream quality; scanned PDFs without OCR yield no text. |
| Chunk | Split into retrievable units | Chunking strategy is a primary retrieval-quality lever (see below). |
| Embed | Map chunks to dense vectors | Embedding model choice affects recall and multilingual behavior. |
| Index | Persist in Chroma | Local persistent store under `.chroma/`. |
| Scan for injection | Flag instruction-like content in documents | Part of the security layer; see [Security](#security). |

### Chunking Trade-offs

| Chunk size | Advantage | Disadvantage |
|---|---|---|
| Small | Precise matches, less noise | May lose surrounding context; more chunks to grade |
| Large | Preserves context | Dilutes embeddings; more noise per chunk; larger prompts |

Overlap between consecutive chunks reduces the chance that an answer is split across a boundary, at the cost of index size. Chunk size and overlap should be chosen empirically on the target corpus using the retrieval metrics in [Evaluation Methodology](#evaluation-methodology).

### Retrieval

- Query embedding followed by approximate nearest-neighbor search over the Chroma collection.
- The top-*k* results are passed to the retrieval grader; only chunks judged relevant are used as generation context.
- Source metadata (document name, chunk position) is preserved so answers can cite sources (`backend/sources.py`).

### Known Retrieval Limitations

- Pure dense retrieval can miss exact-match needs (identifiers, rare terms, numbers). Hybrid retrieval (dense + lexical) and reranking are natural extensions (see [Future Work](#future-work)).
- Tables, figures, and complex layouts in PDFs may be flattened or lost during text extraction.
- Retrieval quality depends on the embedding model's coverage of the corpus domain and language.

---

## Quality Gates: LLM-as-Judge Design

The router, retrieval grader, hallucination grader, and answer grader are all **LLM judges**: a prompted model produces a constrained decision that the graph consumes.

### Design Choices

| Choice | Rationale |
|---|---|
| **Constrained output** (binary / enumerated labels, validated with Pydantic) | Makes decisions machine-consumable and prevents free-text parsing failures. |
| **One judgment per grader** | Keeps each grader's prompt focused and its behavior independently testable. |
| **Deterministic decoding for judges** (low temperature) | Improves repeatability of gate decisions. |
| **Judge sees only what it needs** | The hallucination grader sees `(context, answer)`; the answer grader sees `(question, answer)`; this limits cross-contamination between criteria. |
| **Untrusted content stays wrapped** | Judges also receive documents, so they are subject to the same injection risk as the generator. |

### Known Weaknesses of LLM-as-Judge (and Mitigations)

| Weakness | Description | Mitigation |
|---|---|---|
| Judge error | Judges can accept ungrounded answers or reject good ones. | Evaluate graders against labeled examples; track gate pass rates in observability. |
| Self-preference / shared bias | If the generator and judge are the same model, errors may correlate. | Consider a different judge model for gates (see [Future Work](#future-work)). |
| Sensitivity to prompt wording | Small prompt changes can shift decisions. | Pin and version grader prompts; cover them with regression tests. |
| Cost | Each gate adds an LLM call. | Cap retries; batch chunk grading where possible; see [Cost and Latency Analysis](#cost-and-latency-analysis). |
| Injection through judged content | A document can try to influence its own relevance judgment. | Wrap untrusted content; run document-injection detection before grading. |

---

## Human-in-the-Loop

The architecture includes a human approval step for web-search execution.

```text
Agent decides Web Search
          │
          ▼
     Approval Step
       /       \
      /         \
 Approve       Reject
    │             │
    ▼             ▼
Web Search     Rejection
```

This provides an additional control point before external retrieval is executed.

**Why gate web search?** Web search sends (some form of) the user's query to an external service and brings untrusted external content into the agent's context. Gating it provides:

- **Privacy control**: the user or operator decides whether the query may leave the local environment.
- **Reduced excessive agency**: external actions require explicit consent.
- **Reduced exposure to web-borne injection**: no external content enters the context without approval.

**Mechanism.** The approval step is implemented as a node in the graph; the rejection path is a defined terminal branch (`web_search_rejected`) that returns an explicit response instead of silently continuing. HITL usage is recorded in observability data.

---

## Security

Security is integrated into the application rather than treated as an external concern.

### Threat Model

**Assets**

- The hidden system prompt and application configuration
- API keys and environment secrets
- User-uploaded documents
- The integrity of agent behavior and answers
- Service availability and cost

**Adversaries and attack surfaces**

| Surface | Adversary | Example attack |
|---|---|---|
| User question | Malicious or careless user | "Ignore previous instructions…", jailbreak prompts, system-prompt extraction |
| Uploaded documents | Third-party document author or malicious uploader | Hidden instructions embedded in a PDF/DOCX/TXT/MD file |
| Web search results | Authors of web pages | Indirect prompt injection in page content |
| Model output | The model itself (via any of the above) | Leaking secrets, producing unbounded output |
| Upload endpoint | Malicious uploader | Oversized files, unsupported/malformed files |

**Trust boundaries.** Only application-authored instructions are trusted. User text, document text, and web content are all untrusted *data* and may never be treated as instructions.

### Implemented Controls

#### Prompt Injection Detection

Detects malicious instructions attempting to manipulate the agent.

#### System Prompt Extraction Protection

Detects attempts to retrieve hidden system instructions.

#### Jailbreak Detection

Identifies common attempts to bypass model or application restrictions.

#### Document Injection Detection

Uploaded or retrieved content is treated as untrusted information rather than trusted instructions.

#### Untrusted Content Handling

External content is wrapped and handled separately from trusted application instructions.

#### Output Protection

The output layer can redact sensitive information such as:

- API keys
- Groq secrets
- Environment secrets

#### Output Length Protection

Responses are constrained to prevent uncontrolled output.

### Defense-in-Depth Map

| Stage | Control | Module |
|---|---|---|
| Input | Prompt-injection, system-prompt-extraction, and jailbreak detection | `security/prompt_guard.py` |
| Ingestion / retrieval | Document-injection detection; untrusted-content wrapping | `security/content_guard.py` |
| Generation | Trusted instructions kept separate from wrapped untrusted context | prompts in `graph/chains/` |
| Output | Secret redaction; length limits | `security/output_guard.py` |
| Configuration | Central security settings | `security/security_config.py` |

### Mapping to OWASP Top 10 for LLM Applications

| OWASP risk category | Relevant control here |
|---|---|
| Prompt injection | Prompt guard, content guard, untrusted-content wrapping |
| Sensitive information disclosure | Output guard secret redaction; system-prompt-extraction protection |
| Improper output handling | Output guard and length limits |
| Excessive agency | Human approval before web search; fixed graph transitions |
| Unbounded consumption | Output length limits, bounded retries, upload size limit |

### Security Limitations (Honest Scope)

- Pattern- and heuristic-based detection **cannot catch all** prompt-injection or jailbreak variants; it reduces risk rather than eliminating it.
- Wrapping untrusted content lowers, but does not remove, the chance that a model follows embedded instructions.
- Secret redaction protects secrets the guard knows how to recognize; it is not a substitute for keeping secrets out of the model's context in the first place.
- Authentication, authorization, rate limiting, malware scanning, and archive limits are **not yet implemented** (see [Future Work](#future-work)) and are required before any multi-user or public deployment.

---

## Evaluation Methodology

The project includes automated evaluation components for the RAG and agent workflow.

Evaluation covers:

- Retrieval quality
- Hallucination/grounding
- Answer relevance
- Routing behavior
- Generation retry behavior
- Response validation
- Evaluation tracking

Evaluation results can be persisted for later analysis.

### Evaluation Layers

| Layer | Question answered | Typical metrics |
|---|---|---|
| **Component: router** | Does it choose the right source? | Routing accuracy, confusion matrix (local vs. web) |
| **Component: retrieval** | Does retrieval surface the needed evidence? | Recall@k, precision@k, MRR, context precision/recall |
| **Component: graders** | Do the judges agree with human labels? | Accuracy, precision/recall of "relevant", "grounded", "useful" decisions |
| **Generation** | Is the answer faithful and responsive? | Faithfulness, answer relevancy (RAGAS-style) |
| **Agent behavior** | Does the loop behave as designed? | Retry rate, retry success rate, fallback rate, termination behavior |
| **System** | End-to-end cost and speed | Latency distribution, LLM calls per query |
| **Security** | Do controls block attacks without breaking normal use? | Attack block rate, false-positive rate on benign inputs |

### Metric Definitions (RAGAS-Style Families)

| Metric | Intuition |
|---|---|
| **Faithfulness** | Fraction of claims in the answer that are supported by the retrieved context. |
| **Answer relevancy** | How directly the answer addresses the question. |
| **Context precision** | Whether relevant chunks are ranked above irrelevant ones in the retrieved set. |
| **Context recall** | Whether the retrieved context covers the information needed for a reference answer. |

Reference-free metrics scale to unlabeled data but inherit the limitations of LLM-as-judge; they are best combined with a smaller human-labeled benchmark.

### Recommended Experimental Protocol

1. **Benchmark construction.** Build a question set over uploaded documents with (a) questions answerable from the documents, (b) questions *not* answerable from the documents (to exercise fallback), and (c) questions requiring current external information (to exercise routing). Store under `evaluation/benchmarks/`.
2. **Ablations.** Measure the contribution of each component by disabling it in turn:

   | Configuration | Purpose |
   |---|---|
   | Baseline: retrieve → generate | Reference naive RAG |
   | + Router | Effect of adaptive routing |
   | + Retrieval grader | Effect of filtering and fallback |
   | + Hallucination grader | Effect on faithfulness |
   | + Answer grader | Effect on responsiveness |
   | + Retry (full system) | Effect of corrective regeneration |

3. **Report** quality metrics together with cost metrics (LLM calls, latency) so improvements are weighed against overhead.
4. **Judge validation.** Hand-label a sample of grader decisions to estimate each grader's own error rate.
5. **Variance.** Repeat runs to quantify variability caused by stochastic generation; report means and spread, not single runs.
6. **Persist** results under `evaluation/results/` for longitudinal comparison across prompt, model, or chunking changes.

> **Note:** The repository ships the evaluation scaffolding and automated tests. Quantitative benchmark results should be reported only after they are actually measured on a defined dataset; none are claimed here.

---

## Observability

The application includes an observability layer for tracking agent executions.

Recorded information includes:

- Selected route
- Number of retrieved documents
- Grounding results
- Retry count
- Latency
- Source counts
- Human-in-the-loop usage
- Evaluation information

This makes it possible to inspect how the agent arrived at a response rather than treating the LLM as a black box.

### Example Run Record (Illustrative)

```json
{
  "route": "local",
  "retrieved_documents": 4,
  "relevant_documents": 2,
  "web_search_used": false,
  "human_approval_used": false,
  "grounded": true,
  "answer_useful": true,
  "retry_count": 1,
  "source_count": 2,
  "latency_ms": 0
}
```

*(Field names and values above are illustrative of the kind of information recorded, not a verbatim schema.)*

### What the Data Enables

| Question | Signal |
|---|---|
| Is the router over-using web search? | Route distribution over time |
| Is local retrieval failing for some topics? | Fallback rate; relevant-documents ratio |
| Are generations frequently failing gates? | Retry rate; grounded / useful rates |
| Where does latency come from? | Latency vs. retry count and route |
| Is the approval step used? | HITL usage rate |

---

## Cost and Latency Analysis

Total latency is approximately the sum of sequential stages:

```text
L_total ≈ L_security + L_route + L_retrieve
        + L_grade(k) + [L_web]
        + (1 + r) × (L_generate + L_hallucination + L_answer)
        + L_output_guard
```

Key observations:

- **Retries multiply the dominant term.** Each retry repeats generation and both validators, so `R_max` bounds worst-case latency and cost.
- **Per-chunk grading scales with `k`.** Grading each retrieved chunk separately gives interpretable decisions but increases LLM calls; parallelizing or batching grading reduces wall-clock latency.
- **Web search adds a network round trip** and, if enabled, human-approval wait time.
- **Gate overhead is the price of quality.** The ablation protocol above is intended to quantify whether the quality gain justifies the added calls for a given workload.
- **Groq-hosted inference** is typically low-latency, which makes multiple small judge calls more practical than they would be on slower endpoints.

---

## Failure Modes and Limitations

| Area | Limitation | Impact | Possible mitigation |
|---|---|---|---|
| Routing | Router can misclassify a question | Unneeded web search, or missed fresh information | Fallback via retrieval grader; routing evaluation |
| Grading | LLM judges can be wrong or inconsistent | False accepts or rejects | Judge validation; stronger or independent judge model |
| Groundedness | Faithfulness is not truth | Wrong answers from wrong sources | Source quality controls; citations for user verification |
| Retry | Retries may not converge | Extra cost; final answer may still fail a gate | Retry cap; explicit low-confidence response policy |
| Retrieval | Dense-only search; text-extraction loss | Missed exact matches, tables, figures | Hybrid search, reranking, better parsers/OCR |
| Context | Long contexts degrade usage of mid-context information | Lower answer quality | Filter by relevance; cap context size |
| Security | Heuristic detection is incomplete | Residual injection risk | Defense in depth; least privilege; periodic red-teaming |
| Scope | Single-user, no authentication | Not suitable for public exposure | See [Future Work](#future-work) |
| Memory | No persistent conversation memory | Each query handled independently | Add conversation memory |
| Scale | Local vector store | Limited scale and concurrency | Production vector database |

---

## Technology Stack

### Backend

- Python
- FastAPI
- LangGraph
- LangChain
- Groq
- Pydantic

### AI / ML

- Retrieval-Augmented Generation
- Agentic workflows
- LLM-based grading
- Hallucination detection
- Prompt engineering
- Embeddings
- Vector search

### Storage

- Chroma
- Document storage
- Evaluation history
- Observability history

### Frontend

- Next.js
- React
- TypeScript

### Deployment

- Docker
- Docker Compose
- Uvicorn

### Testing

- Pytest

---

## Project Structure

```text
Adaptive-Unified-Retrieval-Reasoning-RAG/
│
├── backend/
│   ├── api.py
│   ├── config.py
│   ├── model.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── response.py
│   ├── sources.py
│   ├── evaluation.py
│   ├── observability.py
│   ├── main.py
│   │
│   ├── graph/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── consts.py
│   │   ├── context.py
│   │   │
│   │   ├── chains/
│   │   │   ├── answer_grader.py
│   │   │   ├── generation.py
│   │   │   ├── hallucination_grader.py
│   │   │   ├── retrieval_grader.py
│   │   │   └── router.py
│   │   │
│   │   └── nodes/
│   │       ├── build_response.py
│   │       ├── generate.py
│   │       ├── grade_documents.py
│   │       ├── increment_retry.py
│   │       ├── retrieve.py
│   │       ├── web_search.py
│   │       ├── web_search_approval.py
│   │       └── web_search_rejected.py
│   │
│   └── security/
│       ├── content_guard.py
│       ├── output_guard.py
│       ├── prompt_guard.py
│       ├── security_config.py
│       └── tests/
│
├── evaluation/
│   ├── benchmarks/
│   ├── results/
│   └── ragas/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   ├── Dockerfile
│   └── next.config.ts
│
├── data/
├── uploads/
│
├── .chroma/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Module Guide

| Path | Role |
|---|---|
| `backend/graph/graph.py` | Defines nodes, edges, and conditional transitions of the workflow |
| `backend/graph/state.py` | Typed state shared across nodes |
| `backend/graph/chains/` | Prompted LLM chains: generator, router, and the three graders |
| `backend/graph/nodes/` | Graph nodes wrapping retrieval, grading, generation, search, approval, retry, and response assembly |
| `backend/security/` | Input, content, and output guards plus configuration and tests |
| `backend/ingestion.py` / `retrieval.py` | Document ingestion and vector retrieval |
| `backend/evaluation.py` / `observability.py` | Evaluation tracking and run telemetry |
| `evaluation/` | Benchmarks, stored results, and RAGAS-based evaluation |

---

## Installation

### Prerequisites

Make sure you have:

- Python 3.11+
- Node.js 20+
- Docker Desktop
- Git
- A Groq API key

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key

MAX_UPLOAD_SIZE_MB=10

FRONTEND_URL=http://localhost:3000
```

Use `.env.example` as the template for the complete configuration.

**Never commit your real API key to Git.**

---

## Running Locally

### 1. Clone the repository

```powershell
git clone <your-repository-url>
cd Adaptive-Unified-Retrieval-Reasoning-RAG
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` and add your API configuration.

### 5. Start the backend

```powershell
uvicorn backend.api:api --reload
```

Backend:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

### 6. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:3000
```

---

## Running with Docker

Docker Compose is provided for running the complete application.

Make sure **Docker Desktop is running** before executing the commands below.

From the project root:

```powershell
docker compose build
```

Start the application:

```powershell
docker compose up
```

Or run in detached mode:

```powershell
docker compose up -d
```

The services are exposed at:

```text
Frontend:
http://localhost:3000

Backend:
http://localhost:8000

API documentation:
http://localhost:8000/docs
```

Check container status:

```powershell
docker compose ps
```

View logs:

```powershell
docker compose logs -f
```

Stop the application:

```powershell
docker compose down
```

---

## API

The FastAPI backend exposes endpoints for:

- Health checks
- Document upload
- Document management
- Chat/query execution
- Evaluation
- Observability

Interactive API documentation is available through:

```text
http://localhost:8000/docs
```

---

## Testing

The project includes automated tests covering the core agent, evaluation, response, and security components.

Run the complete test suite:

```powershell
python -m pytest -v
```

Current verification:

```text
33 passed
```

The test suite covers:

```text
✓ Retrieval grading
✓ Hallucination grading
✓ Answer grading
✓ Query routing
✓ Local RAG path
✓ Web-search fallback
✓ Generation retries
✓ Quality gates
✓ Evaluation tracking
✓ Response validation
✓ Prompt injection protection
✓ System prompt extraction protection
✓ Jailbreak detection
✓ Document injection detection
✓ Untrusted content handling
✓ Secret redaction
✓ Output protection
```

### Testing Philosophy

The project tests the behavior of individual components as well as the agent workflow.

The goal is not simply to verify that functions execute successfully, but to validate important agent behaviors:

```text
Routing
Retrieval
Fallback
Generation
Validation
Retry
Security
Evaluation
```

### Testing Strategy

| Level | What is tested | Approach |
|---|---|---|
| Unit | Individual graders, router, guards | Isolated inputs with expected labels/outcomes |
| Path | Local RAG path; web-search fallback path | Graph runs with controlled component outputs |
| Control flow | Retry loop, quality gates, termination | Forced gate failures verifying bounded retries |
| Security | Injection, extraction, jailbreak, document injection, redaction | Adversarial and benign examples |
| Response | Response building and validation | Structured output checks |

Using controlled (stubbed) component outputs for path and control-flow tests makes graph behavior deterministic and independent of LLM variability. LLM-dependent quality is measured separately through the evaluation layer.

---

## Example Workflow

### Step 1 — Upload a document

Upload a supported document through the application.

```text
PDF / DOCX / TXT / MD
          │
          ▼
      Ingestion
          │
          ▼
      Chunking
          │
          ▼
   Vector Storage
```

### Step 2 — Ask a question

```text
"What are the main conclusions of this document?"
```

### Step 3 — Agent processes the question

```text
Question
   ↓
Security Check
   ↓
Route
   ↓
Retrieve
   ↓
Grade Documents
   ↓
Generate
   ↓
Check Grounding
   ↓
Check Answer Quality
   ↓
Final Response
```

### Step 4 — Inspect execution

Use the observability interface to inspect the execution path and evaluation information.

### Worked Trace Examples (Illustrative)

**Trace A — question answerable from uploaded documents**

```text
Security: pass → Route: local → Retrieve (k chunks) → Grade: ≥1 relevant
→ Generate → Grounded: yes → Useful: yes → Response (with sources)
```

**Trace B — retry after a failed gate**

```text
Route: local → Retrieve → Grade: relevant → Generate
→ Grounded: no → Increment retry → Generate
→ Grounded: yes → Useful: yes → Response (retry_count = 1)
```

**Trace C — local evidence insufficient, web fallback**

```text
Route: local → Retrieve → Grade: none relevant
→ Approval (if enabled): approved → Web search → Generate
→ Grounded: yes → Useful: yes → Response
```

**Trace D — approval rejected**

```text
Route: web → Approval: rejected → Rejection response (no external call made)
```

**Trace E — blocked input**

```text
Security: prompt-injection detected → request refused before routing
```

---

## Why This Project Is Agentic

A conventional RAG pipeline might look like:

```text
Question
   ↓
Vector Search
   ↓
LLM
   ↓
Answer
```

This project introduces decision-making and feedback loops:

```text
                         ┌───────────────┐
                         │    Router     │
                         └───────┬───────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
             Local Retrieval              Web Search
                   │
                   ▼
           Retrieval Grader
                   │
            ┌──────┴──────┐
            ▼             ▼
         Relevant      Not Relevant
            │             │
            └──────┬──────┘
                   ▼
               Generation
                   │
                   ▼
          Hallucination Check
                   │
                   ▼
             Answer Check
                   │
            ┌──────┴──────┐
            ▼             ▼
          Accept         Retry
                            │
                            └──────► Generation
```

The system therefore combines:

- Routing
- Tool selection
- Retrieval
- Evaluation
- Feedback
- Retry
- Human approval

within a stateful LangGraph workflow.

### Comparison with Other Designs

| Property | Fixed RAG pipeline | Free-form tool-using agent | This system |
|---|---|---|---|
| Control flow | Fixed | Chosen by the model at each step | Declared graph with model-produced judgments |
| Source selection | None | Model-chosen | Router + grader-driven fallback |
| Self-checking | None | Optional, ad hoc | Two explicit gates |
| Termination | Guaranteed | Not guaranteed | Guaranteed (bounded retry) |
| Testability | High | Low | High |
| Flexibility | Low | High | Medium |

The system deliberately sits between the two extremes: more adaptive than a fixed pipeline, but more predictable and testable than an unconstrained agent.

---

## Engineering Highlights

### Adaptive Retrieval

Rather than assuming vector search is always sufficient, the system evaluates retrieval quality and can transition to external search.

### Self-Evaluation

The system evaluates both retrieved evidence and generated responses.

### Quality-Controlled Generation

Generation is not automatically accepted. Hallucination and answer-quality checks determine whether the response should be returned or regenerated.

### Security-Aware RAG

Documents and external content are treated as potentially untrusted input, with dedicated controls for prompt injection and output leakage.

### Observability

Agent execution information is recorded so the behavior of the system can be analyzed.

### Containerized Deployment

Both frontend and backend are containerized and orchestrated using Docker Compose.

---

## Future Work

The current implementation focuses on the core Agentic RAG architecture.

### Production Hardening

- Authentication and authorization
- API rate limiting
- Advanced document malware scanning
- More comprehensive file-type validation
- ZIP/archive resource limits
- Dependency vulnerability scanning
- Container image scanning
- Distributed observability
- Production vector database
- Background document processing
- Streaming responses
- Advanced RAG evaluation dashboards
- Persistent conversation memory
- Multi-user support
- Cloud deployment

These are intentionally treated as future production-hardening improvements rather than prerequisites for the current system.

### Research Directions

| Direction | Motivation |
|---|---|
| Hybrid retrieval (dense + lexical) and reranking | Improve recall for exact terms and precision of top results |
| Query rewriting / decomposition | Handle ambiguous or multi-hop questions |
| Independent judge model | Reduce correlated generator/judge errors |
| Calibrated confidence and abstention | Return "cannot verify" instead of a failed-gate answer |
| Claim-level citation verification | Verify each claim against specific supporting passages |
| Adaptive retry strategy | Change the retrieval or prompt on retry rather than regenerating identically |
| Learned router | Replace prompt-based routing with a classifier trained on logged decisions |
| Adversarial red-teaming harness | Systematically measure security control coverage |
| Source credibility scoring for web results | Reduce reliance on low-quality sources |

---

## Project Status

| Component | Status |
|---|---|
| Agentic workflow | ✅ |
| LangGraph orchestration | ✅ |
| Adaptive routing | ✅ |
| Vector retrieval | ✅ |
| Web-search fallback | ✅ |
| Retrieval grading | ✅ |
| Hallucination grading | ✅ |
| Answer grading | ✅ |
| Generation retry | ✅ |
| Human-in-the-loop | ✅ |
| Security layer | ✅ |
| Evaluation | ✅ |
| Observability | ✅ |
| FastAPI backend | ✅ |
| Next.js frontend | ✅ |
| Docker backend | ✅ |
| Docker frontend | ✅ |
| Automated tests | ✅ 33 passed |

---


**Machine Learning Engineer | Generative AI | Agentic AI | LLM Systems**

```text
Python • LangGraph • LangChain • RAG • LLMs
FastAPI • Docker • Vector Search • Generative AI
```

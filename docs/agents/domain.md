# Domain Docs

## Layout: Single-context

This repo uses a single-context layout. All domain knowledge lives in one place:

- **Domain language**: `CONTEXT.md` at the repo root
- **Architectural decisions**: `docs/adr/` directory (ADR = Architecture Decision Record)

## Consumer rules

When a skill reads these files, it should:

1. **Read `CONTEXT.md` first** to understand the domain language — what terms mean, what the bounded contexts and aggregates are, and any ubiquitous-language conventions.
2. **Consult `docs/adr/` when making architectural decisions** — check whether a past ADR already covers the territory before proposing a new approach.
3. **If `CONTEXT.md` doesn't exist yet**, the skill should note that and proceed without it — it can infer domain concepts from the codebase but won't have the canonical definitions.

## Creating these files

- `CONTEXT.md` should be written by the team. It describes the domain in plain language: key entities, their relationships, business rules, and terminology.
- ADRs go in `docs/adr/` using a numbered naming convention (e.g., `001-use-sympy-for-symbolic-math.md`). Each ADR records a decision, its context, the alternatives considered, and the rationale.

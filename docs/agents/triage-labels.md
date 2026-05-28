# Triage Labels

This repo uses the canonical label vocabulary. All five labels should exist in the GitHub repo's label list for the triage skill to work correctly.

| Role                | Label             | Meaning                                   |
|---------------------|-------------------|-------------------------------------------|
| Needs evaluation    | `needs-triage`    | Maintainer needs to assess                |
| Waiting on reporter | `needs-info`      | More details needed from the issue author |
| Ready for agent     | `ready-for-agent` | Fully specified; AFK agent can pick up    |
| Ready for human     | `ready-for-human` | Needs a human to implement                |
| Won't fix           | `wontfix`         | Will not be actioned                      |

## State machine

```
needs-triage  →  needs-info  (needs more detail from reporter)
needs-triage  →  ready-for-agent  (fully specified, small/mechanical)
needs-triage  →  ready-for-human  (needs human judgment or deep domain knowledge)
needs-triage  →  wontfix  (out of scope, duplicate, or not planned)
needs-info    →  needs-triage  (reporter responded — re-evaluate)
ready-for-agent →  (closed when done)
ready-for-human →  (closed when done)
```

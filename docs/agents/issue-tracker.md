# Issue Tracker

- **Provider**: GitHub
- **Repository**: `adxorg/astrodynx`
- **CLI**: `gh`

## Creating issues

Use `gh issue create --repo adxorg/astrodynx --title "..." --body "..."`.

To add labels at creation time: `gh issue create --repo adxorg/astrodynx --title "..." --body "..." --label "needs-triage"`.

## Reading issues

- List open issues: `gh issue list --repo adxorg/astrodynx --state open`
- Read a specific issue: `gh issue view <number> --repo adxorg/astrodynx`
- Search with labels: `gh issue list --repo adxorg/astrodynx --label "ready-for-agent"`

## Updating issues

- Add a label: `gh issue edit <number> --repo adxorg/astrodynx --add-label "needs-info"`
- Remove a label: `gh issue edit <number> --repo adxorg/astrodynx --remove-label "needs-triage"`
- Add a comment: `gh issue comment <number> --repo adxorg/astrodynx --body "..."`

## Closing issues

`gh issue close <number> --repo adxorg/astrodynx`

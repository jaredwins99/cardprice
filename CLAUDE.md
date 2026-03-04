# Cardprice Project Instructions

## Workflow Rules
- **Always parallelize with subagents**: Spawn 5-20 subagents in parallel whenever possible. Don't wait to be asked — proactively parallelize research, testing, implementation, and exploration.
- **Always use Linear for task tracking**: Create and update Linear issues for all work items. Use `scripts/linear.sh` or direct `curl` to the GraphQL API. Never use the Linear MCP server (hangs in WSL2).
- **Be proactive**: Don't wait for explicit prompts to parallelize or track work in Linear.

## Linear Setup
- API: `https://api.linear.app/graphql`
- API Key: `***REDACTED***`
- Team ID: `6853be95-2e18-4b0e-b739-df6047c2e865`
- Script: `scripts/linear.sh` (search, create, update, states)
- Always `export LINEAR_API_KEY` before using the script

## Environment
- WSL2 bridged networking, PostgreSQL via Unix socket peer auth
- DB: `postgresql+psycopg2://godli@/cardprice`
- Card scanner server: `:8888`
- Claude vision subagents: strip all `CLAUDE*` env vars, use direct `subprocess.run()`

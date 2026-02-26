#!/bin/bash
# Lightweight Linear API wrapper - bypasses the slow @linear/sdk
# Usage: linear.sh <action> [args...]

LINEAR_API_KEY="${LINEAR_API_KEY:?Set LINEAR_API_KEY env var}"
LINEAR_API="https://api.linear.app/graphql"
TEAM_ID="${LINEAR_TEAM_ID:-6853be95-2e18-4b0e-b739-df6047c2e865}"

gql() {
  curl -s --max-time 30 -X POST "$LINEAR_API" \
    -H "Content-Type: application/json" \
    -H "Authorization: $LINEAR_API_KEY" \
    -d "$1"
}

case "$1" in
  search)
    shift
    QUERY="${1:-}"
    LIMIT="${2:-10}"
    gql "{\"query\": \"{ issues(filter: { team: { id: { eq: \\\"$TEAM_ID\\\" } } }, first: $LIMIT) { nodes { id identifier title priority state { name } assignee { name } url } } }\"}"
    ;;
  create)
    # Usage: linear.sh create "title" "description" [priority]
    TITLE="$2"
    DESC="${3:-}"
    PRIORITY="${4:-3}"
    gql "{\"query\": \"mutation { issueCreate(input: { title: \\\"$TITLE\\\", description: \\\"$DESC\\\", teamId: \\\"$TEAM_ID\\\", priority: $PRIORITY }) { success issue { id identifier title url } } }\"}"
    ;;
  update)
    # Usage: linear.sh update <issue-id> [title] [description] [priority] [status]
    ISSUE_ID="$2"
    FIELDS=""
    [ -n "$3" ] && FIELDS="$FIELDS title: \\\"$3\\\""
    [ -n "$4" ] && FIELDS="$FIELDS description: \\\"$4\\\""
    [ -n "$5" ] && FIELDS="$FIELDS priority: $5"
    [ -n "$6" ] && FIELDS="$FIELDS stateId: \\\"$6\\\""
    gql "{\"query\": \"mutation { issueUpdate(id: \\\"$ISSUE_ID\\\", input: {$FIELDS }) { success issue { id identifier title url } } }\"}"
    ;;
  teams)
    gql '{"query": "{ teams { nodes { id name } } }"}'
    ;;
  states)
    gql "{\"query\": \"{ team(id: \\\"$TEAM_ID\\\") { states { nodes { id name type } } } }\"}"
    ;;
  viewer)
    gql '{"query": "{ viewer { id name email } }"}'
    ;;
  *)
    echo "Usage: linear.sh <search|create|update|teams|states|viewer> [args...]"
    exit 1
    ;;
esac

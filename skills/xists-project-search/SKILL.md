---
name: xists-project-search
description: Find, compare, and evaluate existing open-source projects before proposing a new implementation. Use when the user asks for a library, framework, repository, self-hosted tool, GitHub alternative, or technology choice, or when existing-project discovery would help a build request. Prefer a configured xists MCP server for candidate discovery.
---

# Xists Project Search

Use xists to discover existing projects, then help the user decide how to use
them. xists supplies indexed project evidence; it does not decide for the user
or guarantee that every need has a match.

## When to search

Search before recommending a new implementation when the user is looking for:

- an open-source project, library, framework, service, CLI, or developer tool;
- a self-hosted option or alternative to an existing product;
- projects for a technical comparison or architecture decision;
- implementation references for a build request.

Do not search for requests that are purely about editing the current codebase,
explaining supplied code, or performing a task with no useful existing-project
discovery component.

## Workflow

1. Use a configured xists MCP server's `search_projects` tool with the user's
   need as a concise natural-language query. Request at most five results by
   default.
2. Preserve the returned ranking and `abstained` state. Do not claim that the
   search covers all of GitHub; it covers the active xists index.
3. Inspect at most the strongest two candidates with `inspect_project` when
   their stored profile is needed to make a comparison. Do not inspect every
   result by default.
4. Use `index_stats` only when index coverage, model compatibility, or corpus
   size materially affects the answer.
5. Synthesize a concise recommendation from the returned evidence and the
   user's stated constraints.

## Answer contract

- Lead with the recommended candidate or a short comparison, not raw tool
  output. Include repository URLs when available.
- For each meaningful candidate, explain what it is best for and its relevant
  limits. Distinguish indexed facts from your own comparison or inference.
- Treat `abstained: true` as: "the current xists index did not contain a
  sufficiently credible match." Do not turn it into a claim that no such
  project exists. Ask whether the user wants to broaden the query or index.
- Keep the default answer to two or three candidates. Expand only when the user
  asks for a larger landscape.
- xists is candidate discovery and initial comparison. For adoption decisions,
  verify current maintenance, license, compatibility, security posture, and
  primary documentation when those factors matter.

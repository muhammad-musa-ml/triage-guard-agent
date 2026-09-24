# triage-guard-agent

A customer-support triage agent. It answers order questions, does basic arithmetic, and files refunds, but it will not let a refund over $50 go through without a human saying yes first.

The idea for the human-approval gate came from a post I saw at https://www.instagram.com/p/DdZQsecgKqv/?img_index=7&stkn=ZHkweDQyYWJwNXg=.

## What it does

The agent is a small state graph with one model node, one tool node, an approval node, and a loop guard. The model can call three tools:

- `calculate`, which evaluates arithmetic through a restricted grammar instead of `eval`.
- `search_order`, which looks up one of three demo orders (`ORD-1001` to `ORD-1003`).
- `refund_order`, which validates a refund and simulates submitting it. No payment processor is contacted.

Routing after the model runs is deterministic, not left to the model. If the refund amount is strictly over $50.00, the graph routes to an approval node that calls LangGraph's `interrupt()` and waits. A human resumes the run with `Command(resume={"approved": True/False, ...})`, and only then does the tool node actually record the refund. A refund of exactly $50.00 goes straight through. The tool node checks the approval decision again before running, so the model cannot talk its way around the gate.

If the model keeps calling tools without ever producing a final answer, a loop guard cuts it off after 8 model calls in one turn and returns a plain response instead of erroring out.

State (message history, call counts, last route, pending refund decision) is a `TypedDict` with an `operator.add` reducer on the messages list, so each node appends to history instead of overwriting it. It is checkpointed to SQLite through `SqliteSaver`, keyed by a `thread_id`, so a session survives closing and reopening the process. The saver uses `JsonPlusSerializer(allowed_msgpack_modules=None)`, which keeps deserialization restricted to LangGraph's built-in safe types instead of trusting arbitrary classes out of the database file.

There is a CLI with two paths: `--demo`, which uses a small deterministic stand-in model and needs no API key, and a real path backed by OpenAI through `langchain-openai`. Both go through the same graph, tools, checkpointing, and approval flow.

## Built with

- Python 3.11+
- `langgraph` for the state graph, routing, checkpointing, and interrupts
- `langchain-core` for messages and tool definitions
- `langgraph-checkpoint-sqlite` for the SQLite checkpointer
- `langchain-openai` for the real model path
- `pytest` for tests

Versions are pinned exactly in `pyproject.toml`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run

No API key needed:

```bash
triage-guard --demo --session approval-demo
```

Then talk to it:

```text
You: Where is ORD-1001?
Agent: ORD-1001 is shipped; total $74.95; tracking 1Z-DEMO-1001.

You: Refund $75 for ORD-1003 because the headphones arrived damaged

--- HUMAN APPROVAL REQUIRED ---
Order:  ORD-1003
Amount: $75.00
Reason:  the headphones arrived damaged
Policy:  refunds over $50.00 require approval
Approve refund? [y/n]: y
Agent: Refund submitted for ORD-1003 in the amount of $75.00.
```

`/state` prints the checkpointed counters, `/quit` exits. Run the same command again with the same `--session` (and `--db`, if you set one) and the history is still there.

For the real model, set `OPENAI_API_KEY` and drop `--demo`:

```bash
export OPENAI_API_KEY="your-key"
triage-guard --session live-support --model gpt-4.1-mini
```

## Tests

```bash
pytest
```

The suite covers the message reducer, the safe arithmetic grammar, order and refund validation, the routing function around the exact $50 boundary, a full model-to-tool-to-model cycle, SQLite state surviving a closed and reopened connection, approve and reject through `Command(resume=...)`, and the loop guard. I wrote these by hand and they line up with what the code does, but I have not been able to run the full suite in every environment I've tried this in, so treat a green run on your machine as the actual confirmation, not this paragraph.

## What's rough or missing

- SQLite is local, single-process state. It is not a multi-tenant boundary and there is no authentication on top of it. Do not commit the checkpoint database; it can contain customer messages.
- There is no retry or backoff around the OpenAI call. A transient API failure just fails the turn.
- The CLI is the only interface. No web frontend, no API server.
- No fine-tuning, no vector search, no ticket categories beyond the three demo orders.
- Postgres checkpointing and tracing (for example LangSmith) are things I'd add next, not things that exist now.

## License

MIT

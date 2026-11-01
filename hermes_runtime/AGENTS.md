# Hermes runtime backend

Applies with the root guide. This package owns shared session contracts, execution events, approvals, cancellation and the private canonical transport. It has no renderer or stdio UI entry point.

Keep behavior in topical sibling modules; server.py is the registration facade. Declare RPC parameters/results and event payloads in contracts before registering handlers. Unknown methods must fail rather than falling back to a legacy UI dispatcher.

Session authority belongs to the gateway. Preserve durable admission identities, generation fencing, replay ordering and explicit unknown-outcome handling. Never replay a side effect automatically after an ambiguous disconnect.

Bind the owning profile scope in background work, callbacks, RPCs, threads and child processes. Credentials, sessions, terminal state and approvals must remain profile scoped. Never switch process environment per request.

Approval and edit-policy checks run before mutation; requester failures deny execution. Restore temporary callbacks after each turn. Cancellation and shutdown must release resources and settle or explicitly classify outstanding work.

Freeze prompts, memories and tool schemas for each conversation, except normal context compression. Configuration and extension changes apply to future sessions.

Use scripts/run_tests.sh with an isolated Hermes home. Exercise production imports and behavior; do not replace security behavior with presentation mocks.

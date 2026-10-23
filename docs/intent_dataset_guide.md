# EAKTP intent annotation guide

This guide defines labels for the current EAKTP workflow. It is derived from
`app.agents.models.Intent`, `AgentWorkflow`, the agent API request model, and the
allowlisted tools. The labels route requests; they are not claims about the content
of a document.

## Supported labels

### `question`

Use when the user wants an answer or explanation grounded in indexed documents.
The workflow retrieves evidence and generates a grounded answer.

Include:

- Questions asking what a policy, procedure, or document says.
- Requests for an explanation or summary that require document retrieval.
- Factual questions whose expected answer comes from indexed evidence.

Exclude:

- A request to locate a passage without asking for an answer (`search`).
- A database report (`sql_report`).
- A draft message (`email_draft`).
- A request to assess an existing answer against evidence (`verify`).

Illustrative examples only:

- `What is the retention policy?`
- `Explain the approval requirements in the handbook.`

### `search`

Use when the primary goal is to find or inspect relevant document evidence,
sections, or passages. The workflow retrieves evidence and returns it rather than
generating a grounded answer.

Include:

- Requests to find a section, clause, passage, or document.
- Requests for relevant source material or citations.
- Requests to search indexed documents for a topic.

Exclude:

- Questions asking the system to answer or explain the topic (`question`).
- A count, grouping, or other allowlisted database report (`sql_report`).

Illustrative examples only:

- `Find the approval section in the handbook.`
- `Search the policies for data retention language.`

### `sql_report`

Use when the user explicitly requests a structured report produced by the
configured read-only SQL tool. In the current implementation, supported reports
are allowlisted names such as `document_counts` and `documents_by_type`; the route
does not execute arbitrary SQL.

Include:

- Counts or grouped summaries over the approved document metadata database.
- A request to report documents by media type when that report is configured.

Exclude:

- A question whose answer should come from document text (`question`).
- A request to search document content (`search`).
- Any request to write, update, delete, or execute arbitrary SQL. Such a request is
  outside the supported label schema and must be escalated for review, not relabeled
  as `sql_report`.

Illustrative examples only:

- `Show document counts by type.`
- `Report how many PDF documents are indexed.`

### `email_draft`

Use when the requested output is an email draft. The workflow creates a draft and
always marks it as requiring approval; it never sends email automatically.

Include:

- Requests to compose, draft, or prepare an email.
- Requests that provide or imply a recipient, subject, and message body.

Exclude:

- A request to send, schedule, or automatically deliver an email. The current
  system does not support sending; annotate the underlying drafting intent only if
  the user explicitly asks for a draft, otherwise flag the record for review.
- A question about email policy with no drafting request (`question`).

Illustrative examples only:

- `Draft an email asking the owner to review this.`
- `Prepare a message to the compliance team summarizing the finding.`

### `verify`

Use when the user asks whether an existing answer or claim is supported by cited
or retrieved evidence. The workflow invokes claim-level verification and can
classify claims as supported, unsupported, contradicted, or uncertain.

Include:

- Requests to check an answer against supplied evidence.
- Requests to validate citations or determine whether a claim is supported.
- Requests to identify contradiction or missing support in an answer.

Exclude:

- A new question that has not supplied an answer or claim to check (`question`).
- A request to find source text (`search`).

Illustrative examples only:

- `Is the answer supported by the cited policy?`
- `Check whether this claim is contradicted by the retrieved evidence.`

## Boundaries and multi-intent requests

Apply the following policy consistently:

1. Label the user's primary requested outcome, not every verb in the message.
2. If the request asks for a draft, `email_draft` takes precedence over the
   informational content of the draft.
3. If the request evaluates an already-written answer or claim, use `verify` even
   when retrieval would be needed to obtain evidence.
4. If the request asks to locate material, use `search`; if it asks the system to
   answer from that material, use `question`.
5. Use `sql_report` only for the approved reporting operation, never as a general
   SQL or data-manipulation label.
6. If two outcomes are equally primary, do not silently invent a new label or split
   one record into multiple labels. Mark the record `needs_review` in the annotation
   workflow metadata, then resolve it with a reviewer. `needs_review` is not a
   training label in the current five-label schema.
7. If no supported label fits, reject or escalate the record rather than forcing a
   label. Do not add an `unknown` label without a versioned schema and routing
   change.

The current API accepts a string intent and the workflow rejects values outside the
five `Intent` enum values. Annotation metadata such as `needs_review`, reviewer
decisions, or disagreement notes must therefore remain outside the training
`text`/`intent` fields.

## Privacy and PII handling

- Collect only approved, necessary user requests.
- Redact email addresses, phone numbers, government identifiers, names, addresses,
  account numbers, credentials, secrets, and other sensitive content before
  annotation export.
- Use the existing privacy/redaction pipeline where applicable; do not copy raw
  private documents into the dataset.
- Preserve only stable redaction placeholders needed to retain intent context.
- Do not put raw examples, evidence, or private provenance details in logs,
  commit messages, or issue comments.
- Keep private source data and annotation workspaces outside Git. Commit only
  sanitized, approved, versioned datasets.
- Review redactions manually for high-risk or ambiguous cases. Redaction is not a
  guarantee that all sensitive information has been removed.

## Collection and provenance workflow

For every candidate example, maintain a controlled annotation record containing:

- A stable, non-sensitive `id`.
- `text` and one of the five supported `intent` values.
- A `group_id` linking paraphrases, copied templates, the same conversation, or
  records derived from one source request.
- Provenance such as source system, collection date, annotator, and dataset
  version, stored in a protected metadata file or system.
- Privacy-review status and redaction status.
- Annotation status: `pending`, `approved`, `needs_review`, or `rejected`.
- Optional reviewer notes kept out of the model input.

Recommended process:

1. Obtain approval for the source and intended use.
2. Extract only the user request, not unnecessary document contents or responses.
3. Assign a stable ID and group ID before annotation.
4. Redact PII and secrets before a reviewer sees or exports the record.
5. Apply the label definitions and record annotator confidence.
6. Run exact and normalized duplicate checks, then review near-duplicates and
   paraphrase groups.
7. Have a second annotator review a representative sample and all low-confidence
   or boundary cases.
8. Resolve disagreements with a documented adjudicator decision.
9. Freeze an approved dataset version and compute its fingerprint before splitting.

## Leakage-safe splitting and quality gates

Split by `group_id`, never by individual row when related examples exist. A
conversation, template family, copied request, or paraphrase cluster must occur in
exactly one of train, validation, or test.

Before a training run is approved:

- Every record has a valid stable ID, group ID, text, and supported intent.
- No exact duplicates remain.
- Near-duplicate and paraphrase groups have been reviewed.
- All sensitive content has been redacted.
- Every required class has adequate examples.
- Train, validation, and test are all non-empty.
- The validation and test sets are large enough to support the intended claims.
- The split seed, group policy, dataset version, and fingerprint are recorded.
- The held-out test set is frozen and reused unchanged for baseline comparison.

The repository's five-record fixture intentionally fails the real-training
validation gate because it has one example per class and an empty validation split.
It remains a deterministic smoke-test fixture only.

## Human review and disagreement resolution

Use two independent annotators for new batches and for all examples marked
ambiguous. Reviewers should cite the applicable label rule and boundary rule, not
just select a preferred label.

When annotators disagree:

1. Compare the primary requested outcome.
2. Check the inclusion/exclusion rules above.
3. Check whether the request is actually unsupported by the current schema.
4. Preserve both decisions and the disagreement reason.
5. An adjudicator assigns the final approved label or rejects the example.
6. Update the guide or a versioned annotation decision log when a recurring
   ambiguity reveals a missing rule.

Do not silently change labels to improve class balance. If a routing ambiguity
requires a new intent, treat it as a separate product/schema decision and update
the workflow, API, validator, tests, and dataset version together.

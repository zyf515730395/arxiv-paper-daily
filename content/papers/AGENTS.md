# Public paper data

- `conference-papers.json` stores public accepted-paper title/URL/edition metadata from official proceedings, with source and collection dates. It is independent of archive review decisions; matching never adds papers or reverses deletions. Raw downloaded HTML and maintenance reports stay in ignored `build/conferences/`.

- `archive.json` is the canonical historical/public paper archive. Preserve historical entries except during an explicitly authorized full recheck: remove only explicitly rejected topic memberships, retain uncertain memberships, and record reasons and original rows in private recheck receipts. Synchronize ledger decisions and annotations through a recoverable, concurrency-checked transaction.
- `arxiv-candidates.json` owns collection cursors and pending/accepted/rejected decisions; only pending candidates are reviewed.
- `paper-annotations.json` owns validated public topic/tag/type/affiliation metadata.
- Conference records store their validated technical tags, type and affiliations in an optional `annotation` field inside `conference-library.json`; its topics follow the record's current topic memberships. Conference IDs never enter the arXiv-only annotation catalog. Original abstracts and model responses remain private evidence.
- Inference source documents, responses, caches, receipts, reports and queue checkpoints never belong here; they remain under ignored `build/paper-summaries/`. Models remain in the configured private runtime location.
- `docs/togos-papers.json` is a generated compatibility export, not an editable input.
- Curated-list refreshes verify paper identity, authors and publication dates against primary sources; list titles and displayed IDs may be wrong or renamed. Reuse existing archive rows and preserve review decisions and collection cursors; unresolved IDs stay in local audit reports.
- The offline pipeline supports modern numeric arXiv IDs only. Do not invent IDs for non-arXiv papers. Archive topic membership does not override validated annotation topics or a ledger's selected topic.
- Apply archive/ledger pairs with the existing lock, concurrent-edit checks and recoverable curation journal. Rebuild with `python -m papers build`; source snapshots and import evidence stay in ignored `build/reports/`, never in public data.

- `conference-library.json` is the public conference intake overlay: stable DOI/official-URL identities, title-screened topics, provenance, partial publication dates, and validated summary output. It joins the website archive without inserting synthetic IDs into the arXiv ledger. Private screening responses, source bodies and retry state remain in `build/conferences/intake/`.
- Explicit conference rechecks may attach `topic_review` decisions. All-false records retain their original metadata and summary as hidden tombstones with empty topics; subsequent intake preserves these decisions rather than resurrecting them.

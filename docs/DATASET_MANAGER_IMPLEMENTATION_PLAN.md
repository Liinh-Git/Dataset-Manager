# Dataset Manager Implementation Plan

> Audit date: 2026-09-24. Correction/simplification pass applied to the existing
> uncommitted working tree; it does not restart from baseline.

## Source snapshot

| Source | Snapshot |
|---|---|
| Dataset-Manager | `main` at `42e4029fc859d99550d1efa26d5fb35715736d70`; initially clean before the master task |
| PBL4 reference | `dev` at `197cb478468d88d7f40365d79ce70d402d53d2f5`; unchanged by these tasks |
| Node Agent design/plan | Drive revisions modified 2026-09-24 |
| DBS design/plan | Drive revisions modified 2026-09-24 |
| Canonical Dataset Manager design/API | Drive revisions modified 2026-09-06/07 |

Historical chunk/prefetch documents were read for rationale and are not
normative where the reviewed Node Agent/DBS designs supersede them.

## Frozen decisions

- Dataset Manager is local/private and has no V1 application-auth framework.
- A durable `ArtifactStore` is required; Hugging Face is the configured
  WAN-readable implementation in the current target deployment.
- Physical batch is Work Unit V1 source; no Work Unit model/API is added.
- Core materialization uses full configured-size batches plus at most one tail.
- The V1 profile, not core batching, owns three-shard and equal-count constraints.
- Shard and Worker are different identity domains.
- The current PBL4 numerical `shard_count == expected_workers` check is a legacy
  V1 compatibility constraint, not architectural identity.
- Partial batches remain valid artifacts and are identifiable by `sample_count`.
- Lifecycle, registration ACK, manifest schema and hash chain remain unchanged.
- Low-memory preprocessing and single-process atomic HF cache fill are retained.

## Current-state audit and artifact-origin finding

Dataset-Manager HF layout is:

```text
dataset-builds/{build_id}/dataset-manifest.json
dataset-builds/{build_id}/{relative_shard_manifest_path}
dataset-builds/{build_id}/{relative_filename}
```

This maps directly to a public/readable Hugging Face `resolve/{revision}` origin.
The HF backend now emits that origin metadata. `DatasetService` consumes generic
published-location fields and has no provider URL knowledge. Local storage still
emits the Dataset Manager `/artifacts/v1/...` origin using `PUBLIC_BASE_URL`.

PBL4 `dev` is not yet direct-HF compatible:

- Runtime `DatasetManifestClient` and `_fetch_shard_manifests` use the configured
  Dataset Manager HTTP routes.
- Worker `ShardDownloader` ignores `root_manifest_path` and hard-codes the same
  Dataset Manager route shapes.
- Therefore a WAN Worker cannot currently consume the emitted HF origin without
  the minimal downloader change described below.

## Done in Dataset Manager

- [x] remove the rejected environment-mode fields and control-authentication
  middleware, configuration, tests, and documentation;
- [x] retain local/private control API and local artifact-serving fallback;
- [x] emit HF `resolve` artifact origin plus `root_manifest_path` for HF storage;
- [x] move HF origin construction from service orchestration into the HF store;
- [x] replace legacy Equal-K spreading with generic fixed-size batches plus tail;
- [x] centralize current profile shape/class/shard and equal-count constraints;
- [x] preserve root REGISTERING verification and READY/DEPRECATED provisioning
  semantics;
- [x] use memory-mapped/indexed CIFAR source and per-batch normalization;
- [x] prove lazy preprocessing equivalence and deterministic new-layout output;
- [x] retain atomic publication and SHA-256 verification;
- [x] serialize same-process HF cache fills and atomically replace verified files;
- [x] test multi-reader local HTTP provisioning and real fake-HF concurrent miss;
- [x] keep manifests free of Worker identity and retain partial batches.

## Required from PBL4 DBS implementation

These are consumer changes, not Dataset Manager work:

- honor `DatasetAssignment.root_manifest_path`;
- for HF origin, join `relative_shard_manifest_path` and `relative_filename`
  instead of synthesizing Dataset Manager-specific routes;
- add `DatasetCache` and `cache_scope=all_shards` provisioning;
- build `WorkUnitRef` catalog from full physical batches;
- add `work_units[]` processing while preserving one StrictBSP contribution;
- stop treating numeric `shard_id` as `worker_id`, while acknowledging that the
  current V1 compatibility check may still require equal counts during migration.

No PBL4 source is modified by this correction pass.

## Keep / modify / remove

### Keep

- manifest/NPZ schemas, lifecycle, registration and immutable publication;
- local and HF storage, cold-start read-through and integrity checks;
- `PUBLIC_BASE_URL` solely for local-storage HTTP artifact origin;
- low-memory build path and useful regression/integration tests.

### Modify

- HF status metadata now advertises HF rather than Dataset Manager as artifact
  origin;
- documentation and claims distinguish artifact provisioning/offline cache from
  full training E2E;
- PBL4 drift wording distinguishes identity semantics, current numerical
  compatibility, and the DBS target.
- physical grouping changes from legacy Equal-K remainder spreading to
  configured-size batches plus at most one tail; the manifest schema remains V1.

### Remove

- production/development application modes;
- control credentials and HTTP authentication behavior;
- production URL validation and Backend-token integration requirements.

## Phase DAG

```text
current diff audit
  -> delete auth/deployment overengineering
  -> audit HF layout and PBL4 URL consumers
  -> emit backend-specific artifact origin metadata
  -> correct docs/tests/claims
  -> full Dataset-Manager verification
  -> read-only PBL4 integration verification
```

## Test matrix

| Requirement | Verification |
|---|---|
| Regressions | full Dataset-Manager pytest suite |
| Low memory | same samples normalize equivalently; lazy/eager new-layout trees match |
| Fixed-size batching | CIFAR-scale U=32/64/128/256 and shard counts 1/2/3/4/8 |
| Profile boundary | invalid shape/shard or unequal physical counts reject without smearing |
| Layout/integrity | root -> shard -> batch corruption and schema tests |
| HF origin metadata | fake-HF store returns exact `resolve` base/root; service source has no vendor routes |
| HF read-through | cold-start/cache-loss/hash verification tests |
| HF concurrency | simultaneous fake-HF cache miss returns identical complete bytes; no temp remains |
| Multi-reader | concurrent local HTTP clients download all shards with identical bytes |
| Offline compatibility | downloaded NPZ remains loadable after service close |
| Worker-free build | serialized root manifest contains no `worker_id` |
| Partial batch | shard manifest retains entries with `sample_count < batch_size` |
| Registration gates | root available REGISTERING; shard/batch require READY/DEPRECATED |

The multi-reader/offline test is an all-shards artifact provisioning and
offline-cache compatibility test, not a full DBS training E2E. Real HF network
smoke is skipped because it requires credentials and remote mutation.

## Definition of done

- [x] removed auth/deployment-only fields, code, docs and tests;
- [x] retained low-memory, deterministic artifacts and HF durability;
- [x] corrected physical batching and isolated V1 profile constraints;
- [x] artifact-origin metadata matches the selected storage backend;
- [x] documented the exact current PBL4 direct-HF blocker;
- [x] no Work Unit/chunk protocol added to Dataset Manager;
- [x] no PBL4 source or remote state changed;
- [x] final lint, format, pytest, diff and status checks recorded.

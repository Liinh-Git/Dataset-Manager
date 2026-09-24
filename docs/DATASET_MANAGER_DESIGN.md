# Dataset Manager Design

> Status: canonical candidate for the Dataset Manager boundary in PBL4 V1.
> Review date: 2026-09-24; corrected deployment review: 2026-09-24.
> Authority: current canonical domain/data/source-structure documents, then
> `NODE_AGENT_DESIGN.md` and `DBS_DESIGN.md`, then the canonical Dataset Manager
> design/API. Historical 14–20 September proposals are rationale only.

## 1. System position and deployment

Dataset Manager is a local/private build-management service. It is not a public
WAN management service and V1 adds no application-level IAM.

```text
                  local / private
Management Backend ------control HTTP------> Dataset Manager
                                             build / verify / lifecycle
                                                       |
                                                       v
                                              durable ArtifactStore
                                      (local or configured Hugging Face)

Worker <======================= DTP/1 =======================> Runtime
```

The architectural invariant is a durable, immutable artifact origin plus a
verified Worker cache. `HuggingFaceArtifactStore` is the current WAN-readable
deployment choice, not a domain invariant. Current PBL4 `dev` still needs the
consumer work in section 6 before Workers can provision directly from that
origin. After provisioning and N/N `SHARD_READY`, no artifact origin is in the
synchronized training step.

For the local storage backend, Dataset Manager's read-only HTTP routes are the
artifact origin fallback. `PUBLIC_BASE_URL` exists only to describe that local
HTTP origin; it is not a production/WAN mode switch.

## 2. Ownership

Dataset Manager owns source ingestion, validation, static preprocessing,
deterministic partitioning, physical batch materialization, manifest and
SHA-256 construction, atomic immutable publication, registration gating,
artifact metadata, and local/Hugging Face storage.

It does not own Worker membership, workload policy, Work Unit assignment,
training steps, model state, gradient/checkpoint traffic, throughput, or Node
orchestration. It never derives Worker identity from a shard.

## 3. Dataset Build lifecycle

```text
CREATED -> QUEUED -> IMPORTING -> VALIDATING -> PREPROCESSING
        -> MATERIALIZING -> VERIFYING -> REGISTERING -> READY -> DEPRECATED
                                      \-> FAILED             \-> DELETING -> DELETED
```

Publication follows successful hash-tree verification. `READY` is reached only
after Backend catalog commit and a valid registration acknowledgement. DBS adds
no Dataset Build state. `READY` and `DEPRECATED` artifacts are immutable;
rebuild creates a new `build_id`.

## 4. Artifact model and identity

```text
dataset-manifest.json
shards/{shard_id:03d}/shard-manifest.json
shards/{shard_id:03d}/batch-{batch_id:06d}.npz
```

The root manifest pins preprocessing, batch size, partition seed/algorithm,
sample count, physical shards and shard-manifest hashes. Shard manifests pin
batch-relative filenames, sample counts, byte sizes and hashes. NPZ contains
pickle-free `x: float32`, `y: int64`, and `sample_ids: int64`.

`shard_id` is physical storage identity, not `worker_id`. Batch identity is
`(shard_id, batch_id)`. Core partitioning, batching, and manifest construction
are parameterized by `shard_count`.

`CNN_IMAGE_CLASSIFICATION_V1` is the current profile boundary. It resolves
`input_shape=(3,32,32)`, `num_classes=10`, and `shard_count=3`, and retains the
canonical V1 requirement that every shard have the same physical batch count.
The value three is a profile/layout compatibility choice, not a Dataset Manager
or DBS invariant.

For each shard, materialization slices its deterministic sample order into
consecutive batches of configured size `U`, followed by at most one non-empty
tail smaller than `U`. It never pads, duplicates, drops, or spreads the remainder
across otherwise-full batches. Profile validation runs after this generic
materialization. If fixed-size output has unequal shard batch counts, the V1
profile rejects that configuration rather than distorting batch sizes.

Current PBL4 V1 still numerically requires `shard_count == expected_workers` in
Runtime and assigns the same numeric shard to the corresponding Worker. This is
a legacy/current V1 compatibility constraint, not an identity equivalence or a
Dataset Manager invariant. DBS Work Unit mode must eventually remove shard
ownership by provisioning all shards, but this repository does not rewrite that
Runtime/Worker work.

## 5. DBS and Work Unit compatibility

Dataset Manager creates no Work Unit entity, ID, table, manifest, or API.
Runtime interprets an eligible physical batch as:

```text
WorkUnitRef { shard_id, batch_id, sample_count }
```

For adaptive V1, `U = dataset.batch_size`; only batches with
`sample_count == U` are eligible. A partial final batch remains a valid,
identifiable artifact in its shard manifest and is not deleted.

Work Unit mode uses Runtime/Worker `cache_scope=all_shards`. Dataset Manager
does not receive this flag and keeps no Worker cursor or assignment state.

## 6. Artifact origins and URLs

`artifact_base_url` identifies the artifact origin, not necessarily the Dataset
Manager process.

With local storage:

```text
artifact_base_url = {PUBLIC_BASE_URL}/artifacts/v1/dataset-builds/{build_id}
manifest_uri      = {artifact_base_url}/manifest.json
```

With Hugging Face storage, the backend (not `DatasetService`) provides:

```text
artifact_base_url = https://huggingface.co/datasets/{repo}/resolve/{revision}/dataset-builds/{build_id}
manifest_uri      = {artifact_base_url}/dataset-manifest.json
```

The Hugging Face layout exactly matches the manifest-relative paths. Generic
service orchestration consumes `artifact_base_url` and `root_manifest_path`
returned by the selected store and contains no Hugging Face hostname or route
construction. The status API can derive its compatibility `manifest_uri` from
those values instead of adding another location field. The repo
must be readable by Workers; only Dataset Manager needs `HF_TOKEN` to publish.
No token is put in manifests or artifact URLs.

The current PBL4 `dev` Worker cannot yet consume the direct Hugging Face origin:
`ShardDownloader` ignores `root_manifest_path` and constructs Dataset Manager-
specific `/manifest.json`, `/shards/{id}/manifest.json`, and
`/batches/{id}` URLs. The minimal consumer correction is to honor
`root_manifest_path` and, for the HF layout, join the root manifest's
`relative_shard_manifest_path` and each shard manifest's `relative_filename`
against `artifact_base_url`. It also needs all-shards provisioning,
`DatasetCache`, and `WorkUnitRef/work_units[]`. Runtime currently fetches root/shard manifests
from its configured local Dataset Manager URL; that remains valid for the
local/private control deployment.

## 7. HTTP API and state gates

The local/private control API remains `/api/v1/dataset-builds/**`. V1 adds no
application authentication middleware. Deployment isolation supplies the boundary.

Dataset Manager also keeps its read-only fallback routes:

- `GET /artifacts/v1/dataset-builds/{build_id}/manifest.json`
- `GET /artifacts/v1/dataset-builds/{build_id}/shards/{shard_id}/manifest.json`
- `GET /artifacts/v1/dataset-builds/{build_id}/shards/{shard_id}/batches/{batch_id}`

State gates are intentional and separate physical readability from logical
eligibility:

- Root manifest is readable from Dataset Manager in `REGISTERING` so Backend
  can independently verify it before catalog commit; it remains readable in
  `READY` and `DEPRECATED`.
- Shard manifests and batches are served by Dataset Manager only for `READY` or
  `DEPRECATED` builds.
- In `REGISTERING`, the root manifest may already be physically readable from
  HF so Backend can verify its hash and commit the catalog.
- `READY` is the control-plane eligibility gate for a new Job/Attempt. HTTP 200
  from an artifact origin does not make a build eligible.
- `DEPRECATED` artifacts remain immutable/readable for existing recovery when
  policy permits, but are not selected for new work.

Responses have correct `Content-Length` and SHA-256 ETag. Paths are resolved
from server-owned manifests. No Work Unit/chunk/step/model endpoint exists.

## 8. Registration

```text
VERIFYING -> publish -> REGISTERING
Backend polls status and reads root manifest
Backend verifies dataset_manifest_hash and commits PostgreSQL
Backend sends stable registration_id + hash ACK
Dataset Manager persists ACK -> READY
```

Dataset Manager never writes Backend DB or performs outbound registration.
Backend/DB outage leaves the build `REGISTERING`; repeated matching ACK is
idempotent and mismatched hash is rejected.

## 9. Storage and concurrency

Local storage publishes by atomic directory rename. Hugging Face is the durable
backend when configured; local disk is staging plus verified read-through cache.
Every remote root, shard and batch is verified against the hash chain.

Within one Dataset Manager process a global re-entrant lock serializes HF cache
fills. Downloads are written to a unique sibling temporary file, flushed and
atomically published with `os.replace`; simultaneous readers cannot observe
partial bytes. The global scope may serialize unrelated misses, but V1 prefers
simple correctness over a lock manager.

## 10. Failure semantics

- Before `SHARD_READY`, origin outage or hash failure blocks readiness.
- After N/N `SHARD_READY`, Dataset Manager/HF outage does not stop training.
- Interrupted downloads are not promoted to Worker cache.
- Backend/DB outage keeps the build `REGISTERING`.
- HF cold-start/cache misses fail closed if bytes do not match the manifest.

## 11. Performance and memory

CIFAR source records are validated and memory-mapped. Only one physical batch's
selected indices are normalized at a time, avoiding the full-dataset `float32`
duplicate. Lazy preprocessing is numerically equivalent to eager preprocessing
for the same sample IDs. Under the new fixed-size algorithm, the same source,
configuration, seed, and implementation produce deterministic artifact bytes
and hashes. Hashes are not claimed compatible with legacy Equal-K grouping.

The manifest schema is unchanged: `batch_size` already denotes the configured
target, and each batch entry already permits `sample_count <= batch_size`.
Materialization semantics changed from legacy Equal-K remainder spreading to
fixed-size batches plus a tail; this intentional grouping change can change
hashes for rebuilt data without requiring a schema-shape version bump.

## 12. Non-goals and superseded proposals

No scheduler, adaptive algorithm, Work Unit database, `work_unit_id`,
`storage_chunk_id`, ChunkCache protocol, `/chunks`, `/work-units`, per-step
Dataset Manager request, prefetch daemon, readiness high-watermark, `data_wait`,
gradient/model traffic, public control IAM, or Node Agent logic is part of V1.

## 13. Constraint classification

- Architectural invariant: deterministic immutable artifacts, hash-chain
  verification, manifest-relative paths, and lifecycle eligibility gates.
- Current profile constraint: CIFAR image shape/classes, three storage shards,
  and equal `batch_count_per_shard` for `CNN_IMAGE_CLASSIFICATION_V1`.
- Current deployment choice: local/private control service with either local or
  Hugging Face durable storage.
- Current implementation limit: PBL4 `dev` still uses Dataset Manager-specific
  routes and assigned-shard caching.
- Future non-goal here: no provider framework, new chunk format, or Dataset
  Manager Work Unit API.

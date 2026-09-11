import time
import uuid

import httpx

from dataset_manager.hashing import sha256_bytes


def main():
    base_url = "http://127.0.0.1:9200"
    client = httpx.Client(base_url=base_url, timeout=10.0)

    # 1. Healthz
    r = client.get("/healthz")
    assert r.status_code == 200, f"Healthz failed: {r.text}"
    health = r.json()["data"]
    assert health["status"] == "ok"
    assert health["storage_writable"] is True
    print("[PASS] 1. GET /healthz")

    # 2. Submit build
    body = {
        "command_id": f"cmd-smoke-{uuid.uuid4().hex[:8]}",
        "source": {
            "type": "cifar10_download",
            "dataset_name": "cifar10",
            "version": "binary-v1",
        },
        "profile": "CNN_IMAGE_CLASSIFICATION_V1",
        "input_shape": [3, 32, 32],
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "batch_size": 2,
        "shard_count": 3,
        "partition_seed": 42,
    }
    r = client.post(
        "/api/v1/dataset-builds",
        json=body,
        headers={"Idempotency-Key": f"smoke-key-{uuid.uuid4().hex[:8]}"},
    )
    assert r.status_code == 202, f"Create build failed: {r.text}"
    build_id = r.json()["data"]["dataset_build_id"]
    print(f"[PASS] 2. POST /api/v1/dataset-builds (id={build_id})")

    # 3. Poll until REGISTERING
    start = time.time()
    while time.time() - start < 10:
        r = client.get(f"/api/v1/dataset-builds/{build_id}")
        assert r.status_code == 200
        st = r.json()["data"]
        if st["state"] == "REGISTERING":
            break
        time.sleep(0.1)
    assert st["state"] == "REGISTERING", f"Build did not reach REGISTERING: {st}"
    manifest_hash = st["dataset_manifest_hash"]
    print(f"[PASS] 3. GET .../{build_id} -> REGISTERING (hash={manifest_hash[:12]}...)")

    # 4. Registration ACK -> READY
    ack_body = {
        "dataset_manifest_hash": manifest_hash,
        "registration_id": "reg-smoke-123",
        "catalog_persisted_at": "2026-09-11T12:00:00Z",
    }
    r = client.post(f"/api/v1/dataset-builds/{build_id}/registration-ack", json=ack_body)
    assert r.status_code == 200, f"Registration ack failed: {r.text}"
    assert r.json()["data"]["state"] == "READY"
    print("[PASS] 4. POST /api/v1/dataset-builds/.../registration-ack -> READY")

    # 5. GET root manifest
    r = client.get(f"/artifacts/v1/dataset-builds/{build_id}/manifest.json")
    assert r.status_code == 200
    root_bytes = r.content
    assert sha256_bytes(root_bytes) == manifest_hash
    root = r.json()
    assert root["dataset_build_id"] == build_id
    assert len(root["shards"]) == 3
    print("[PASS] 5. GET /artifacts/.../manifest.json (verified canonical SHA-256)")

    # 6. GET shard manifest
    shard_ref = root["shards"][0]
    r = client.get(f"/artifacts/v1/dataset-builds/{build_id}/shards/0/manifest.json")
    assert r.status_code == 200
    shard_bytes = r.content
    assert sha256_bytes(shard_bytes) == shard_ref["shard_manifest_sha256"]
    shard = r.json()
    assert shard["shard_id"] == 0
    assert len(shard["batches"]) > 0
    print("[PASS] 6. GET /artifacts/.../shards/0/manifest.json (verified SHA-256)")

    # 7. GET batch artifact
    batch_ref = shard["batches"][0]
    r = client.get(f"/artifacts/v1/dataset-builds/{build_id}/shards/0/batches/0")
    assert r.status_code == 200
    batch_bytes = r.content
    assert len(batch_bytes) == batch_ref["byte_size"]
    assert sha256_bytes(batch_bytes) == batch_ref["sha256"]
    print("[PASS] 7. GET /artifacts/.../shards/0/batches/0 (verified byte_size & SHA-256)")

    # 8. Rebuild
    rebuild_body = {
        "command_id": f"cmd-smoke-rebuild-{uuid.uuid4().hex[:8]}",
        "batch_size": 3,
    }
    r = client.post(
        f"/api/v1/dataset-builds/{build_id}/rebuild",
        json=rebuild_body,
        headers={"Idempotency-Key": f"smoke-rebuild-key-{uuid.uuid4().hex[:8]}"},
    )
    assert r.status_code == 202
    new_build_id = r.json()["data"]["new_dataset_build_id"]
    assert new_build_id != build_id
    print(f"[PASS] 8. POST /api/v1/dataset-builds/.../rebuild -> new_id={new_build_id}")

    # 9. Deprecate
    r = client.post(
        f"/api/v1/dataset-builds/{build_id}/deprecate", json={"reason": "test deprecate"}
    )
    assert r.status_code == 200
    assert r.json()["data"]["state"] == "DEPRECATED"
    print("[PASS] 9. POST /api/v1/dataset-builds/.../deprecate -> DEPRECATED")

    # 10. Purge
    r = client.post(
        f"/api/v1/dataset-builds/{build_id}/purge",
        json={"command_id": f"cmd-purge-{uuid.uuid4().hex[:8]}"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["state"] == "DELETED"
    print("[PASS] 10. POST /api/v1/dataset-builds/.../purge -> DELETED")

    # 11. Deleted build artifact access -> 409
    r = client.get(f"/artifacts/v1/dataset-builds/{build_id}/manifest.json")
    assert r.status_code == 409
    print("[PASS] 11. Deleted build artifact access rejected with 409")

    print("\nALL STANDALONE HTTP SMOKE TESTS PASSED!")


if __name__ == "__main__":
    main()

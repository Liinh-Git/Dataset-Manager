"""FastAPI composition for canonical Dataset Manager control and artifact routes."""

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse

from dataset_manager.config import DatasetManagerConfig
from dataset_manager.schemas import (
    CreateBuildRequest,
    DeprecateRequest,
    PurgeRequest,
    RebuildRequest,
    RegistrationAckRequest,
)
from dataset_manager.service import DatasetService, DatasetServiceError


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-Id") or str(uuid4())


def _success(request: Request, data: object, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        {"request_id": _request_id(request), "data": data, "error": None},
        status_code=status_code,
    )


def create_app(
    config: DatasetManagerConfig | None = None, service: DatasetService | None = None
) -> FastAPI:
    settings = config or DatasetManagerConfig.from_env()
    owned_service = service is None
    dataset_service = service or DatasetService(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owned_service:
            dataset_service.close()

    app = FastAPI(title="PBL4 Dataset Manager", version="1", lifespan=lifespan)
    app.state.dataset_service = dataset_service

    @app.exception_handler(DatasetServiceError)
    async def domain_error(request: Request, exc: DatasetServiceError):
        return JSONResponse(
            {
                "request_id": _request_id(request),
                "data": None,
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                    "retryable": exc.retryable,
                },
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = [
            {key: value for key, value in error.items() if key != "ctx"} for error in exc.errors()
        ]
        return JSONResponse(
            {
                "request_id": _request_id(request),
                "data": None,
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Request validation failed",
                    "details": {"errors": errors},
                    "retryable": False,
                },
            },
            status_code=400,
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _exc: Exception):
        return JSONResponse(
            {
                "request_id": _request_id(request),
                "data": None,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Dataset Manager internal error",
                    "details": {},
                    "retryable": False,
                },
            },
            status_code=500,
        )

    @app.get("/healthz")
    def health(request: Request):
        return _success(request, dataset_service.health())

    @app.post("/api/v1/dataset-builds")
    def create_build(
        request: Request,
        body: CreateBuildRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        return _success(
            request,
            dataset_service.submit(body.model_dump(mode="json"), idempotency_key or ""),
            202,
        )

    @app.get("/api/v1/dataset-builds/{dataset_build_id}")
    def get_build(request: Request, dataset_build_id: str):
        return _success(request, dataset_service.status(dataset_build_id))

    @app.post("/api/v1/dataset-builds/{dataset_build_id}/rebuild")
    def rebuild(
        request: Request,
        dataset_build_id: str,
        body: RebuildRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        return _success(
            request,
            dataset_service.rebuild(
                dataset_build_id, body.model_dump(mode="json"), idempotency_key or ""
            ),
            202,
        )

    @app.post("/api/v1/dataset-builds/{dataset_build_id}/registration-ack")
    def registration_ack(request: Request, dataset_build_id: str, body: RegistrationAckRequest):
        return _success(
            request,
            dataset_service.acknowledge_registration(
                dataset_build_id,
                body.dataset_manifest_hash,
                body.registration_id,
                body.catalog_persisted_at,
            ),
        )

    @app.post("/api/v1/dataset-builds/{dataset_build_id}/deprecate")
    def deprecate(request: Request, dataset_build_id: str, body: DeprecateRequest):
        del body
        return _success(request, dataset_service.deprecate(dataset_build_id))

    @app.post("/api/v1/dataset-builds/{dataset_build_id}/purge")
    def purge(request: Request, dataset_build_id: str, body: PurgeRequest):
        return _success(request, dataset_service.purge(dataset_build_id, body.command_id))

    @app.get("/artifacts/v1/dataset-builds/{dataset_build_id}/manifest.json")
    def root_manifest(dataset_build_id: str):
        path, digest, media_type = dataset_service.artifact(dataset_build_id)
        return FileResponse(
            path,
            media_type=media_type,
            headers={"ETag": f'"{digest}"', "Cache-Control": "public, immutable"},
        )

    @app.get("/artifacts/v1/dataset-builds/{dataset_build_id}/shards/{shard_id}/manifest.json")
    def shard_manifest(dataset_build_id: str, shard_id: int):
        path, digest, media_type = dataset_service.artifact(dataset_build_id, shard_id)
        return FileResponse(
            path,
            media_type=media_type,
            headers={"ETag": f'"{digest}"', "Cache-Control": "public, immutable"},
        )

    @app.get("/artifacts/v1/dataset-builds/{dataset_build_id}/shards/{shard_id}/batches/{batch_id}")
    def batch_artifact(dataset_build_id: str, shard_id: int, batch_id: int):
        path, digest, media_type = dataset_service.artifact(dataset_build_id, shard_id, batch_id)
        return FileResponse(
            path,
            media_type=media_type,
            headers={"ETag": f'"{digest}"', "Cache-Control": "public, immutable"},
        )

    return app

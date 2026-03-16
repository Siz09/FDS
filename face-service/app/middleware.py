"""Request logging and API key authentication middleware."""
import os
import secrets
import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response

log = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    # /docs, /redoc, /openapi.json are intentionally NOT exempt — this is an internal service
    EXEMPT_PATHS = {"/health", "/stats"}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        expected = os.environ.get("FACE_SERVICE_API_KEY", "")
        received = request.headers.get("X-API-Key", "")
        if not expected or not secrets.compare_digest(expected, received):
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
        return await call_next(request)

import os

import httpx

from .prisma_client import Prisma

_http_timeout_s = float(os.getenv("PRISMA_HTTP_TIMEOUT", "15.0"))
_connect_timeout_s = float(os.getenv("PRISMA_CONNECT_TIMEOUT", "5.0"))

prisma = Prisma(
    http={
        "timeout": httpx.Timeout(
            _http_timeout_s,
            connect=min(_connect_timeout_s, _http_timeout_s),
        ),
    },
)

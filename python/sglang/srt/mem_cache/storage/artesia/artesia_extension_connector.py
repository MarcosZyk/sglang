from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from humanfriendly import parse_size

from artesia.connector.connector import ArtesiaConnector
from artesia.control_plane.cache.manager import calculate_model_page_size
from artesia.control_plane.cache.page import split_into_pages
from artesia.service.common import (
    ContextDescription,
    ModelDescription,
    SemanticDescription,
)

logger = logging.getLogger(__name__)


@dataclass
class RegisteredPagesResult:
    page_ids: list[bytes]


@dataclass
class LoadPagesResult:
    num_retrieved: int
    page_ids: list[bytes]


@dataclass
class PageOffloadSpec:
    page_id: bytes
    kv_indices: torch.Tensor


class ArtesiaExtensionConnector:
    """Draft connector for the new Artesia extension-cache interaction model."""

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        model: ModelDescription,
        kv_pool: list[list[torch.Tensor]],
    ):
        self._model = model
        self._page_size: Optional[int] = None
        self._base_connector = ArtesiaConnector(
            local_rank=local_rank,
            device=device,
            model=model,
            kv_pool=kv_pool,
        )

    def open(self) -> None:
        self._base_connector.open()

    def close(self) -> None:
        self._base_connector.close()

    def resolve_model_page_size(self, model: ModelDescription) -> int:
        page_size_env = os.getenv("ARTESIA_PAGE_SIZE")
        if page_size_env is not None:
            page_size = int(page_size_env)
            if page_size <= 0:
                raise RuntimeError(
                    f"Invalid ARTESIA_PAGE_SIZE={page_size_env}. It must be > 0."
                )
            self._page_size = page_size
            return page_size

        page_bytes_env = os.getenv("ARTESIA_PAGE_BYTES_SIZE")
        if page_bytes_env is not None:
            page_bytes_size = parse_size(page_bytes_env, binary=True)
            page_size = calculate_model_page_size(
                page_bytes_size=page_bytes_size,
                dtype=model.dtype,
                kv_shape=model.kv_shape,
                layer_num=model.layer_num,
            )
            if page_size <= 0:
                raise RuntimeError(
                    "Resolved Artesia page size must be > 0. "
                    f"Got {page_size} from ARTESIA_PAGE_BYTES_SIZE={page_bytes_env}."
                )
            self._page_size = page_size
            return page_size

        raise NotImplementedError(
            "ArtesiaExtensionConnector.resolve_model_page_size requires an "
            "Artesia-side handshake API. For early experiments, set "
            "ARTESIA_PAGE_SIZE or ARTESIA_PAGE_BYTES_SIZE so SGLang can derive "
            "the authoritative page size."
        )

    def register_pages(
        self,
        context: ContextDescription,
        semantics: SemanticDescription,
    ) -> RegisteredPagesResult:
        _ = semantics
        if self._page_size is None:
            raise RuntimeError("Artesia page size must be resolved before registration.")

        pages = split_into_pages(context.token_ids, self._page_size)
        if pages:
            logger.warning(
                "ArtesiaExtensionConnector.register_pages is using local page-id "
                "derivation only. Control-plane logical registration still needs "
                "Artesia-side support."
            )
        return RegisteredPagesResult(page_ids=[page.get_id() for page in pages])

    def load_kv_by_suffix(
        self,
        prefix_page_id: bytes,
        suffix_token_ids: list[int],
        semantics: SemanticDescription,
        kv_indices: torch.Tensor,
    ) -> LoadPagesResult:
        _ = prefix_page_id, suffix_token_ids, semantics, kv_indices
        raise NotImplementedError(
            "ArtesiaExtensionConnector.load_kv_by_suffix requires Artesia backend "
            "support for page-id-aware loading."
        )

    def offload_pages(
        self,
        page_specs: list[PageOffloadSpec],
    ) -> None:
        _ = page_specs
        raise NotImplementedError(
            "ArtesiaExtensionConnector.offload_pages requires Artesia backend "
            "support for eviction-time page-id-aware offload."
        )

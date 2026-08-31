from typing import Any

def hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    repo_type: str | None = ...,
    token: str | None = ...,
    **kwargs: Any,
) -> str: ...

def list_repo_files(
    repo_id: str,
    *,
    repo_type: str | None = ...,
    token: str | None = ...,
    **kwargs: Any,
) -> list[str]: ...

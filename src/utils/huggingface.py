from pathlib import Path
from typing import Union
from huggingface_hub import HfApi, hf_hub_url, HfFileSystem


def exists_on_hf(repo_id: str, repo_type: str = "model") -> bool:
    '''
    '''
    fs = HfFileSystem()
    if repo_type == "model":
        return fs.exists(repo_id)
    elif repo_type == "dataset":
        return fs.exists(f"datasets/{repo_id}")
    raise ValueError("Invalid repo_type. Must be either 'model' or 'dataset'")


def upload_to_hf(
    repo_id: str,
    path: Union[str, Path],
    path_in_repo: Union[str, Path],
    repo_type: str,
) -> None:
    api = HfApi()
    if Path(path).is_dir():
        api.upload_folder(
            folder_path=path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    else:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )


def get_file_url_in_hf(
    repo_id: str,
    file_in_repo: str,
    repo_type: str,
) -> str:
    return hf_hub_url(
        repo_id=repo_id,
        filename=file_in_repo,
        repo_type=repo_type,
    )


def get_hf_path(
    repo_id: str,
    file_in_repo: str,
    repo_type: str,
) -> str:
    if repo_type == "dataset":
        return f"datasets/{repo_id}/{file_in_repo}"
    return f"{repo_id}/{file_in_repo}"


def read_file_from_hf(
    repo_id: str,
    file_in_repo: str,
    repo_type: str,
) -> bytes:
    fs = HfFileSystem()
    return fs.read_bytes(path=get_hf_path(repo_id, file_in_repo, repo_type))

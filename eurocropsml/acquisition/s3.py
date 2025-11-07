import io
import logging
import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Literal, cast

import boto3
from botocore.client import BaseClient

from eurocropsml.settings import CONTAINER, ENDPOINT_URL, REGION_NAME, Settings

logger = logging.getLogger(__name__)


def _set_s3_env_variables() -> None:
    credentials = _get_s3_credentials()

    os.environ["AWS_ACCESS_KEY_ID"] = credentials[0]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials[1]
    os.environ["AWS_S3_ENDPOINT"] = ENDPOINT_URL.replace("https://", "").replace("http://", "")
    os.environ["AWS_VIRTUAL_HOSTING"] = "false"
    os.environ["AWS_REGION"] = REGION_NAME
    os.environ["S3_CONTAINER_NAME"] = CONTAINER
    os.environ["CPL_VSIS3_READ_TIMEOUT"] = "60"


def _get_s3_credentials(file_name: str = "eodata-access") -> tuple[str, str]:

    cfg_dir: Path = Settings().cfg_dir
    file_path: Path = cfg_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} was not found. \
            Please first create a file with your EC2 credentials with 'ACCESS_KEY:SECURITY_KEY'."
        )

    with open(file_path, "r") as f:
        credentials = [line.split(":") for line in f][0]

    return credentials[0], credentials[1].strip("\n")


def _establish_s3_client() -> BaseClient:
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_REGION")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        endpoint_url=ENDPOINT_URL,
    )

    return s3_client


def _get_s3_subfolders(
    s3_client: BaseClient, prefix: str, selectionkey: str | None = None
) -> str | list[dict] | None:

    try:
        response: str | dict = s3_client.list_objects_v2(
            Bucket=os.environ.get("S3_CONTAINER_NAME"), Prefix=prefix, Delimiter="/"
        )
        if selectionkey is not None and isinstance(response, dict):
            return cast(list, response[selectionkey])
        return cast(str | list, response)
    except ValueError:
        return None


def _parse_s3_xml(s3_client: BaseClient, key: str) -> ElementTree.Element:
    """Retrieves an XML file from S3 and parses it with ElementTree."""

    # get object from S3
    s3_response = s3_client.get_object(Bucket=os.environ.get("S3_CONTAINER_NAME"), Key=key)

    # read the body content (bytes)
    xml_bytes = s3_response["Body"].read()

    # create a BytesIO buffer (in-memory file) from the bytes
    xml_stream = io.BytesIO(xml_bytes)

    # ElementTree.parse() to read from the in-memory file stream
    tree = ElementTree.parse(xml_stream)

    return tree.getroot()


def _download_s3_prefix(
    s3_client: BaseClient,
    s3_prefix: str,
    local_base_dir: Path,
    file_extension: Literal[".jp2"] = ".jp2",
) -> None:
    """
    Recursively downloads all files under a given S3 prefix
    to a local directory, maintaining the relative path structure.
    """

    # Iterate over all pages of results
    response = _get_s3_subfolders(s3_client, s3_prefix, selectionkey="Contents")

    if isinstance(response, list):
        # Iterate over every object (file) found under the prefix
        for obj in response:
            s3_key = obj["Key"]
            if s3_key.endswith(file_extension):
                # Construct the local file path by appending the relative key
                # to the local base directory.
                # Example: If local_base_dir is /cache/S2B... and s3_prefix is S2B.../IMG_DATA/
                # and s3_key is S2B.../IMG_DATA/B01.jp2, local_file will be /cache/S2B.../IMG_DATA/B01.jp2
                file_name = Path(s3_key).relative_to(Path(s3_prefix))
                local_file = local_base_dir.joinpath(file_name)

                # Ensure the parent directories exist for the local file
                local_file.parent.mkdir(parents=True, exist_ok=True)

                if not local_file.exists():

                    # Download the individual file
                    s3_client.download_file(
                        Bucket=os.environ.get("S3_CONTAINER_NAME"),
                        Key=s3_key,
                        Filename=str(local_file),
                    )
    else:
        logger.info(f"Did not found any files. Skipping copying of {s3_prefix}.")

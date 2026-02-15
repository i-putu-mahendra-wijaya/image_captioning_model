from pathlib import Path
from google.oauth2.service_account import Credentials as Cr


class CredentialAccessor:

    def __init__(
            self,
            credential_path: Path,
            project_id: str,
            location: str
    ) -> None:

        self.credential_path: Path = credential_path

        _str_credential_path: str = str(credential_path)
        self.gcp_sa_credentials: Cr = Cr.from_service_account_file(filename=_str_credential_path)
        self.project_id: str = project_id
        self.location: str = location
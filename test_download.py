import hashlib

import pytest

from wheatkit.data import fetch as dl


class FakeResponse:
    def __init__(self, json_data=None, content=b"", status=200):
        self._json, self.content, self.status_code = json_data, content, status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, n):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    def __init__(self, payload: bytes, md5: str):
        self.payload, self.md5 = payload, md5

    def get(self, url, **kw):
        if "api/records" in url:
            return FakeResponse(
                {
                    "files": [
                        {
                            "key": "x.zip",
                            "size": len(self.payload),
                            "checksum": f"md5:{self.md5}",
                            "links": {"self": "https://zenodo.org/files/x.zip"},
                        }
                    ]
                }
            )
        return FakeResponse(content=self.payload)


def test_licence_gate():
    with pytest.raises(dl.LicenceNotAccepted):
        dl.download("gwhd_2021", accept_licence=False)


def test_tier3_gate(tmp_path):
    with pytest.raises(dl.LicenceNotAccepted):
        dl.download("imag4wheat", dest=tmp_path, accept_licence=True)


def test_zenodo_listing_parses_md5():
    s = FakeSession(b"abc", "900150983cd24fb0d6963f7d28e17f72")
    files = dl.zenodo_files(1, session=s)
    assert files[0].md5 == "900150983cd24fb0d6963f7d28e17f72"


def test_download_verifies_checksum(tmp_path, monkeypatch):
    payload = b"hello wheat"
    md5 = hashlib.md5(payload).hexdigest()
    entry = dl.get_dataset("agicam")
    monkeypatch.setattr(dl, "get_dataset", lambda _id: entry)
    out = dl.download("agicam", dest=tmp_path, accept_licence=True, session=FakeSession(payload, md5))
    assert (out / "x.zip").read_bytes() == payload
    assert "licence" in (out / "LICENCE_NOTICE.txt").read_text().lower()


def test_download_rejects_bad_checksum(tmp_path, monkeypatch):
    entry = dl.get_dataset("agicam")
    monkeypatch.setattr(dl, "get_dataset", lambda _id: entry)
    with pytest.raises(OSError):
        dl.download("agicam", dest=tmp_path, accept_licence=True, session=FakeSession(b"data", "0" * 32))


def test_manual_source_writes_instructions(tmp_path):
    out = dl.download("invita_core_uav", dest=tmp_path, accept_licence=True)
    assert "researchdata.edu.au" in (out / "HOW_TO_GET.txt").read_text()

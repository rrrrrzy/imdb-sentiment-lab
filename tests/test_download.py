"""Regression tests for interrupted range bodies and range integrity."""
import requests
import pytest

from sentiment_lab.data.collect import download


class Response:
    def __init__(self, headers, blocks=(), fail=False):
        self.headers = headers
        self.status_code = 206
        self.blocks = blocks
        self.fail = fail

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, _):
        yield from self.blocks
        if self.fail:
            raise requests.ConnectionError("Connection interrupted mid-body")


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = 0

    def head(self, *_args, **_kwargs):
        return Response({"Content-Length": "3", "ETag": "test-version"})

    def get(self, *_args, **kwargs):
        assert kwargs["headers"]["Range"] == "bytes=0-2"
        assert kwargs["headers"]["If-Range"] == "test-version"
        self.requests += 1
        return next(self.responses)


def test_failed_body_is_retried_without_leaving_partial_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("sentiment_lab.data.collect.time.sleep", lambda _: None)
    client = Client([Response({"Content-Range": "bytes 0-2/3"}, [b"x"], fail=True),
                     Response({"Content-Range": "bytes 0-2/3"}, [b"abc"])])
    target = tmp_path / "reviews.tar.gz"
    download(client, "https://example.org/reviews", target)
    assert target.read_bytes() == b"abc"
    assert client.requests == 2
    assert not target.with_suffix(".part").exists()


def test_wrong_range_is_rejected_and_not_published(tmp_path):
    target = tmp_path / "reviews.tar.gz"
    client = Client([Response({"Content-Range": "bytes 1-2/3"}, [b"bc"])])
    with pytest.raises(ValueError, match="Unexpected Content-Range"):
        download(client, "https://example.org/reviews", target)
    assert not target.exists()
    assert not target.with_suffix(".part").exists()

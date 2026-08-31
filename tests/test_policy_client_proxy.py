from openpi_client import msgpack_numpy

from eval_utils.policy_client import WebsocketClientPolicy


class _FakeConnection:
    def __init__(self, responses=None):
        self.responses = list(responses or [{"model": "test"}])
        self.sent = []

    def recv(self):
        response = self.responses.pop(0)
        if isinstance(response, str):
            return response
        return msgpack_numpy.Packer().pack(response)

    def send(self, payload):
        self.sent.append(msgpack_numpy.unpackb(payload))


def test_policy_client_never_routes_server_connection_through_download_proxy(
    monkeypatch,
):
    calls = []

    def connect(uri, **kwargs):
        calls.append((uri, kwargs))
        return _FakeConnection()

    monkeypatch.setattr("websockets.sync.client.connect", connect)

    client = WebsocketClientPolicy("127.0.0.1", 6105)

    assert client.get_server_metadata() == {"model": "test"}
    assert calls[0][0] == "ws://127.0.0.1:6105"
    assert calls[0][1]["proxy"] is None


def test_policy_client_exposes_snapshot_and_restore_endpoints(monkeypatch):
    connection = _FakeConnection(
        [{"model": "test"}, "snapshot successful", "restore successful"]
    )
    monkeypatch.setattr(
        "websockets.sync.client.connect",
        lambda *args, **kwargs: connection,
    )
    client = WebsocketClientPolicy("127.0.0.1", 6100)

    client.snapshot({"request_key": "request"})
    client.restore({"candidate_label": "candidate"})

    assert connection.sent == [
        {"request_key": "request", "endpoint": "snapshot"},
        {"candidate_label": "candidate", "endpoint": "restore"},
    ]

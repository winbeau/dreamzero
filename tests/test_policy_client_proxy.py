from openpi_client import msgpack_numpy

from eval_utils.policy_client import WebsocketClientPolicy


class _FakeConnection:
    def recv(self):
        return msgpack_numpy.Packer().pack({"model": "test"})


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

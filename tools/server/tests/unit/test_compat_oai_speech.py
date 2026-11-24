import base64
import pytest
import requests
import struct
from utils import *

server: ServerProcess

BODY = {
    "model": "tts",
    "input": "Hello there.",
    "voice": "female",
    "seed": 42,
    "n_predict": 8,
}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylfm3omni()


def post_audio(path: str, **extra) -> requests.Response:
    return requests.post(server.make_url(path), json={**BODY, **extra}, timeout=DEFAULT_REQUEST_TIMEOUT)


def assert_wav(res: requests.Response, data_size: int | None = None):
    assert res.status_code == 200
    assert res.headers["Content-Type"] == "audio/wav"
    assert res.content[:4] == b"RIFF" and res.content[8:12] == b"WAVE"
    assert struct.unpack("<H", res.content[34:36])[0] == 16 # bits per sample
    pos = res.content.index(b"data")
    size = struct.unpack("<I", res.content[pos + 4:pos + 8])[0]
    assert size == (len(res.content) - pos - 8 if data_size is None else data_size)
    assert len(res.content) > pos + 8


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_speech_default_format_is_wav(path: str):
    server.start()
    assert_wav(post_audio(path))


def test_speech_wav_same_as_tts():
    server.start()
    res = post_audio("/v1/audio/speech", response_format="wav")
    assert_wav(res)
    assert res.content == post_audio("/tts").content


def test_speech_pcm_is_signed_16_bit():
    server.start()
    res = post_audio("/v1/audio/speech", response_format="pcm")
    assert res.status_code == 200
    assert res.headers["Content-Type"] == "audio/pcm;rate=24000;encoding=signed-int;bits=16"

    # /tts keeps the float samples
    ref = post_audio("/tts", response_format="pcm")
    assert ref.headers["Content-Type"] == "audio/pcm;rate=24000;encoding=float;bits=32"

    samples = struct.unpack(f"<{len(res.content) // 2}h", res.content)
    floats = struct.unpack(f"<{len(ref.content) // 4}f", ref.content)
    assert len(samples) == len(floats) > 0
    assert any(v != 0 for v in samples)
    for s, f in zip(samples, floats):
        assert abs(s - max(-1.0, min(1.0, f)) * 32767) <= 1


def test_speech_stream_pcm():
    server.start()
    res = post_audio("/v1/audio/speech", response_format="pcm", stream_format="audio")
    assert res.status_code == 200
    assert res.headers.get("Transfer-Encoding") == "chunked"
    assert res.content == post_audio("/v1/audio/speech", response_format="pcm").content


def test_speech_stream_wav_has_unknown_size():
    server.start()
    res = post_audio("/v1/audio/speech", stream_format="audio")
    assert_wav(res, data_size=0xFFFFFFFF)


def test_speech_voice():
    server.start()
    female = post_audio("/v1/audio/speech", voice="female")
    male = post_audio("/v1/audio/speech", voice="male")
    assert_wav(female)
    assert_wav(male)
    assert female.content != male.content
    assert post_audio("/v1/audio/speech", voice={"id": "male"}).content == male.content


def test_speech_multipart():
    server.start()
    fields = {k: (None, str(v)) for k, v in BODY.items()}
    res = requests.post(server.make_url("/v1/audio/speech"), files=fields, timeout=DEFAULT_REQUEST_TIMEOUT)
    assert_wav(res)
    assert res.content == post_audio("/v1/audio/speech").content


@pytest.mark.parametrize("extra", [
    {"speed": 1.0},
    {"instructions": "Speak slowly."},
])
def test_speech_accepted_params(extra: dict):
    server.start()
    assert_wav(post_audio("/v1/audio/speech", **extra))


@pytest.mark.parametrize("extra,message", [
    ({"response_format": "mp3"},  "response_format"),
    ({"response_format": "opus"}, "response_format"),
    ({"stream_format": "sse"},    "stream_format"),
    ({"speed": 1.5},              "speed"),
    ({"voice": "alloy"},          "unknown voice"),
    ({"input": ""},               "input"),
])
def test_speech_invalid_request(extra: dict, message: str):
    server.start()
    res = post_audio("/v1/audio/speech", **extra)
    assert res.status_code == 400
    assert message in res.json()["error"]["message"]


def test_speech_no_tts_model():
    global server
    server = ServerPreset.tinyllama2()
    server.start()
    assert post_audio("/v1/audio/speech").status_code == 501 # ERROR_TYPE_NOT_SUPPORTED


def test_speech_router():
    global server
    model = server.model_hf_repo
    server = ServerPreset.router()
    server.start()
    assert_wav(post_audio("/v1/audio/speech", model=model))


#
# tts only model, with voice cloning
#

def pocket_tts(path: str = "/v1/audio/speech", **extra) -> requests.Response:
    body = {"input": "Hello there.", "seed": 42, "n_predict": 8, **extra}
    return requests.post(server.make_url(path), json=body, timeout=DEFAULT_REQUEST_TIMEOUT)


def test_tts_only():
    global server
    server = ServerPreset.tinypockettts()
    server.start()
    res = server.make_request("GET", "/props")
    assert not res.body.get("voices") # this slice has no predefined voices
    res = server.make_request("GET", "/v1/models")
    assert res.body["data"][0]["architecture"]["output_modalities"] == ["audio"]
    res = pocket_tts()
    assert_wav(res)
    assert res.content == pocket_tts().content


def test_tts_only_speaker_ref():
    global server
    server = ServerPreset.tinypockettts()
    server.start()
    ref_a = make_wav(seconds=2.0, rate=24000, freq=220)
    ref_b = make_wav(seconds=2.0, rate=24000, freq=440)
    res = pocket_tts("/tts", speaker_ref_b64=base64.b64encode(ref_a).decode())
    assert_wav(res)
    assert res.content != pocket_tts("/tts").content
    assert res.content != pocket_tts("/tts", speaker_ref_b64=base64.b64encode(ref_b).decode()).content
    # the same reference as an uploaded file
    fields = {"input": (None, "Hello there."), "seed": (None, "42"), "n_predict": (None, "8"), "speaker_ref": ("ref.wav", ref_a)}
    res_file = requests.post(server.make_url("/tts"), files=fields, timeout=DEFAULT_REQUEST_TIMEOUT)
    assert res_file.content == res.content
    # /v1/audio/speech passes it through
    assert pocket_tts(speaker_ref_b64=base64.b64encode(ref_a).decode()).content == res.content


@pytest.mark.parametrize("speaker_ref_b64,message", [
    ("!!!", "not valid base64"),
    (base64.b64encode(b"not an audio file").decode(), "failed to decode"),
])
def test_tts_only_invalid_speaker_ref(speaker_ref_b64: str, message: str):
    global server
    server = ServerPreset.tinypockettts()
    server.start()
    res = pocket_tts("/tts", speaker_ref_b64=speaker_ref_b64)
    assert res.status_code == 400
    assert message in res.json()["error"]["message"]


def test_tts_only_end_of_speech():
    global server
    server = ServerPreset.tinypockettts()
    server.start()
    # the end-of-speech head stops the generation before n_predict
    res = pocket_tts(n_predict=16)
    assert_wav(res)
    assert res.content == pocket_tts(n_predict=1000).content


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/audio/transcriptions"])
def test_tts_only_no_text_output(path: str):
    global server
    server = ServerPreset.tinypockettts()
    server.start()
    if path == "/v1/audio/transcriptions":
        res = requests.post(server.make_url(path), files={"file": ("audio.wav", make_wav())}, timeout=DEFAULT_REQUEST_TIMEOUT)
    else:
        res = requests.post(server.make_url(path), json={"messages": [{"role": "user", "content": "hello"}]}, timeout=DEFAULT_REQUEST_TIMEOUT)
    assert res.status_code == 501


def test_tts_only_default_speaker_file(tmp_path):
    global server
    ref_a = make_wav(seconds=2.0, rate=24000, freq=220)
    ref_b = make_wav(seconds=2.0, rate=24000, freq=440)
    ref_a_path = tmp_path / "speaker_a.wav"
    ref_a_path.write_bytes(ref_a)

    # baseline outputs without server-side default speaker file
    server = ServerPreset.tinypockettts()
    server.start()
    no_ref = pocket_tts("/tts").content
    with_ref_a = pocket_tts("/tts", speaker_ref_b64=base64.b64encode(ref_a).decode()).content
    with_ref_b = pocket_tts("/tts", speaker_ref_b64=base64.b64encode(ref_b).decode()).content
    server.stop()

    assert with_ref_a != no_ref
    assert with_ref_b != with_ref_a

    # server configured with default --tts-speaker-file and --tts-speaker-text
    log_path = tmp_path / "server.log"
    server = ServerPreset.tinypockettts()
    server.tts_speaker_file = str(ref_a_path)
    server.tts_speaker_text = "Default server speaker transcript."
    server.debug = True
    server.log_path = str(log_path)
    server.start()

    # /tts without speaker_ref uses server default audio + default transcript
    res = pocket_tts("/tts")
    assert_wav(res)
    assert res.content == with_ref_a

    # /v1/audio/speech without speaker_ref (even with OpenAI voice="alloy") uses server default
    res_oai = pocket_tts("/v1/audio/speech", voice="alloy")
    assert_wav(res_oai)
    assert res_oai.content == with_ref_a

    # per-request speaker_ref_text overrides server --tts-speaker-text while keeping server default audio
    res_text_override = pocket_tts("/tts", speaker_ref_text="Request transcript override.")
    assert_wav(res_text_override)
    assert res_text_override.content == with_ref_a

    # per-request speaker_ref_b64 overrides server default and must NOT inherit --tts-speaker-text
    res_override = pocket_tts("/tts", speaker_ref_b64=base64.b64encode(ref_b).decode())
    assert_wav(res_override)
    assert res_override.content == with_ref_b

    res_oai_override = pocket_tts("/v1/audio/speech", speaker_ref_b64=base64.b64encode(ref_b).decode())
    assert_wav(res_oai_override)
    assert res_oai_override.content == with_ref_b

    server.stop()

    tts_lines = [line for line in log_path.read_text(errors="replace").splitlines() if "tts input:" in line]
    assert len(tts_lines) == 5
    assert 'has_speaker_ref=1, speaker_ref_text="Default server speaker transcript."' in tts_lines[0]
    assert 'has_speaker_ref=1, speaker_ref_text="Default server speaker transcript."' in tts_lines[1]
    assert 'has_speaker_ref=1, speaker_ref_text="Request transcript override."' in tts_lines[2]
    assert 'has_speaker_ref=1, speaker_ref_text=""' in tts_lines[3]
    assert 'has_speaker_ref=1, speaker_ref_text=""' in tts_lines[4]


def test_tts_default_speaker_file_router_preset(tmp_path):
    global server
    ref_a = make_wav(seconds=2.0, rate=24000, freq=220)
    ref_a_path = tmp_path / "speaker_router.wav"
    ref_a_path.write_bytes(ref_a)

    # baseline with explicit speaker_ref_b64
    server = ServerPreset.tinypockettts()
    server.start()
    expected = pocket_tts("/v1/audio/speech", speaker_ref_b64=base64.b64encode(ref_a).decode()).content
    server.stop()

    preset_path = tmp_path / "tts_preset.ini"
    preset_path.write_text(
        "[pocket-cloned]\n"
        f"hf-repo = {ServerPreset.tinypockettts().model_hf_repo}\n"
        f"tts-speaker-file = {ref_a_path}\n"
        "tts-speaker-text = Router preset transcript with spaces.\n"
    )
    log_path = tmp_path / "router.log"
    server = ServerPreset.router()
    server.models_preset = str(preset_path)
    server.debug = True
    server.log_path = str(log_path)
    server.start()

    res = pocket_tts("/v1/audio/speech", model="pocket-cloned", voice="alloy")
    assert_wav(res)
    assert res.content == expected
    server.stop()

    log_text = log_path.read_text(errors="replace")
    assert 'has_speaker_ref=1, speaker_ref_text="Router preset transcript with spaces."' in log_text


def test_tts_default_speaker_file_startup_errors(tmp_path):
    global server

    # nonexistent --tts-speaker-file fails fast at startup
    log_missing = tmp_path / "missing.log"
    server = ServerPreset.tinypockettts()
    server.tts_speaker_file = str(tmp_path / "does_not_exist.wav")
    server.log_path = str(log_missing)
    with pytest.raises(RuntimeError):
        server.start()
    server.stop()
    assert "failed to load speaker file" in log_missing.read_text(errors="replace")

    # invalid (non-audio) --tts-speaker-file fails fast at startup
    bad_file = tmp_path / "not_audio.wav"
    bad_file.write_bytes(b"not a valid audio file")
    log_invalid = tmp_path / "invalid.log"
    server = ServerPreset.tinypockettts()
    server.tts_speaker_file = str(bad_file)
    server.log_path = str(log_invalid)
    with pytest.raises(RuntimeError):
        server.start()
    server.stop()
    assert "failed to load speaker file" in log_invalid.read_text(errors="replace")

    # combining --tts-voice and --tts-speaker-file on a predefined-voice model fails fast
    valid_wav = tmp_path / "valid.wav"
    valid_wav.write_bytes(make_wav(seconds=1.0, rate=24000, freq=220))
    log_conflict = tmp_path / "conflict.log"
    server = ServerPreset.tinylfm3omni()
    server.tts_voice = "female"
    server.tts_speaker_file = str(valid_wav)
    server.log_path = str(log_conflict)
    with pytest.raises(RuntimeError):
        server.start()
    server.stop()
    assert "--tts-voice and --tts-speaker-file cannot be used together" in log_conflict.read_text(errors="replace")


from tools import create_ticket, load_sections, search_docs


def test_docs_have_eight_sections():
    assert len(load_sections()) == 8


def test_vpn_question_finds_vpn_section():
    assert search_docs("My VPN is not connecting")[0]["section"] == "VPN connection"


def test_locked_account_finds_password_section():
    result = search_docs("my account is locked")
    assert result[0]["section"] == "Password reset and account lockout"


def test_wifi_spelling_works():
    assert search_docs("How to connect to wifi")[0]["section"] == "Wi-Fi access"


def test_phishing_question_finds_phishing_section():
    result = search_docs("I clicked a link in a suspicious email")
    assert result[0]["section"] == "Phishing and suspicious emails"


def test_unknown_question_returns_nothing():
    assert search_docs("My laptop shows a blue screen after an update") == []


def test_create_ticket_saves_and_numbers(tmp_path):
    path = tmp_path / "tickets.json"
    first = create_ticket("Shared folder", "Restore deleted files", path=path)
    second = create_ticket("Other", "Another issue", path=path)
    assert first["id"] == "IT-0001"
    assert second["id"] == "IT-0002"
    assert path.exists()
"""Auth tests: hash/verify, token create/decode, login flow against test DB."""

import hashlib

import pytest

from app.auth import create_token, hash_password, password_needs_rehash, verify_password


def test_hash_roundtrip():
    stored = hash_password("melorise")
    assert verify_password("melorise", stored)
    assert not verify_password("wrong", stored)


def test_hash_salts_unique():
    a = hash_password("pw")
    b = hash_password("pw")
    assert a != b


def test_token_roundtrip():
    token = create_token(1, "atte")
    import jwt

    from app.config import settings

    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    assert payload["sub"] == "1"
    assert payload["username"] == "atte"


def test_bad_token_rejected():
    import jwt

    from app.config import settings

    with pytest.raises(jwt.PyJWTError):
        jwt.decode("garbage", settings.jwt_secret, algorithms=["HS256"])


def test_scrypt_hash_verifies_and_needs_no_rehash():
    stored = hash_password("pw")
    assert verify_password("pw", stored)
    assert not verify_password("wrong", stored)
    assert not password_needs_rehash(stored)


def test_legacy_hash_verifies_and_flags_rehash():
    salt = "salty"
    digest = hashlib.sha256(f"{salt}:pw".encode()).hexdigest()
    legacy = f"{salt}${digest}"
    assert verify_password("pw", legacy), "legacy SHA-256 должен верифицироваться"
    assert not verify_password("wrong", legacy)
    assert password_needs_rehash(legacy), "legacy-хэш должен помечаться к апгрейду"


def test_validate_security_strict_raises_on_weak_secrets():
    from app.config import Settings, validate_security

    with pytest.raises(RuntimeError):
        validate_security(Settings(strict_secrets=True, jwt_secret="dev-secret-change-me"))
    with pytest.raises(RuntimeError):
        validate_security(Settings(strict_secrets=True, jwt_secret="real-secret", accounts_enc_key=""))


def test_validate_security_lenient_only_warns():
    from app.config import Settings, validate_security

    # non-strict: weak secrets warn but must not raise (so dev/tests keep booting)
    validate_security(Settings(strict_secrets=False, jwt_secret="dev-secret-change-me", accounts_enc_key=""))
    # strict with real secrets passes
    validate_security(Settings(strict_secrets=True, jwt_secret="real-secret", accounts_enc_key="k"))


def test_login_throttle_blocks_after_max_failures():
    from app.auth import check_login_throttle, clear_login_failures, record_login_failure
    from app.errors import ProblemError

    key = "1.2.3.4:brute@example.com"
    clear_login_failures(key)
    for _ in range(5):
        record_login_failure(key)
    with pytest.raises(ProblemError) as e:
        check_login_throttle(key)
    assert e.value.status == 429
    clear_login_failures(key)
    check_login_throttle(key)  # cleared → allowed again

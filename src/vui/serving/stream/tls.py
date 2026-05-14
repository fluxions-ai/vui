"""Self-signed TLS for the streaming server.

`getUserMedia` (mic access) and many other browser APIs are gated to secure
origins. localhost counts, but a phone on the LAN hitting `http://<ip>:8080`
does not — so we generate a self-signed cert (with the host's LAN IPs in the
SubjectAltName) on first run and offer the server over HTTPS as well.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import socket
import ssl
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_DIR = Path.home() / ".vui"
CERT_PATH = CERT_DIR / "cert.pem"
KEY_PATH = CERT_DIR / "key.pem"


def _lan_ipv4s() -> list[str]:
    """Best-effort: all non-loopback IPv4 addresses on this host."""
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    # UDP-connect trick: ask the OS which interface it would use for the
    # default route. Catches IPs that getaddrinfo misses (e.g. when
    # /etc/hosts maps the hostname to 127.0.1.1 on Debian).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(addrs)


def _generate(cert_path: Path, key_path: Path) -> None:
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    san: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv6Address("::1")),
    ]
    # Hostname (e.g. "janus") and the mDNS variant ("janus.local") so phones
    # reaching the box by name don't trip the cert-mismatch warning on top of
    # the self-signed one.
    hostnames: set[str] = set()
    try:
        h = socket.gethostname()
        if h:
            hostnames.add(h)
            if "." not in h:
                hostnames.add(f"{h}.local")
    except OSError:
        pass
    for name in sorted(hostnames):
        san.append(x509.DNSName(name))
    for ip in _lan_ipv4s():
        try:
            san.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
        except ValueError:
            continue

    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "vui-local")]
    )
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _cert_expired_or_missing(cert_path: Path) -> bool:
    if not cert_path.exists():
        return True
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        return cert.not_valid_after_utc <= dt.datetime.now(dt.timezone.utc)
    except Exception:
        return True


def get_ssl_context() -> ssl.SSLContext:
    """Return an SSLContext for the local self-signed cert, generating it
    (and the key) on first call. Re-generates if the cert has expired."""
    if _cert_expired_or_missing(CERT_PATH) or not KEY_PATH.exists():
        _generate(CERT_PATH, KEY_PATH)
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=str(CERT_PATH), keyfile=str(KEY_PATH))
    return ctx


def lan_urls(port: int, scheme: str = "https") -> list[str]:
    urls = [f"{scheme}://localhost:{port}"]
    try:
        h = socket.gethostname()
        if h:
            urls.append(f"{scheme}://{h}:{port}")
            if "." not in h:
                urls.append(f"{scheme}://{h}.local:{port}")
    except OSError:
        pass
    for ip in _lan_ipv4s():
        urls.append(f"{scheme}://{ip}:{port}")
    return urls

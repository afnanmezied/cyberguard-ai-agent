
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qsl


SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure",
    "account", "update", "password"
]

SPECIAL_CHARACTERS = set("@?!#$%&*+=_~")


def calculate_entropy(text):
    """Calculate Shannon entropy of a URL."""
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def extract_url_features(url):
    """Extract lexical features without visiting the URL."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    url = url.strip()

    # Parse URLs with or without a scheme
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        parsed = urlparse(url)
    else:
        parsed = urlparse("//" + url)

    hostname = parsed.hostname

    if not hostname:
        raise ValueError("Invalid URL: hostname not found")

    # Check whether hostname is an IP address
    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    # Count non-empty path segments
    num_subdirs = len([
        part for part in parsed.path.split("/") if part
    ])

    # Count query parameters
    num_params = len(
        parse_qsl(parsed.query, keep_blank_values=True)
    )

    # Count suspicious keywords
    suspicious_words = sum(
        keyword in url.lower()
        for keyword in SUSPICIOUS_KEYWORDS
    )

    # Extract simplified TLD
    if has_ip:
        tld = "ip"
    elif "." in hostname:
        tld = hostname.rsplit(".", 1)[-1].lower()
    else:
        tld = "unknown"

    # Assemble feature dictionary
    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip,
        "num_subdirs": num_subdirs,
        "num_params": num_params,
        "suspicious_words": suspicious_words,
        "tld": tld,
        "special_char_count": sum(
            char in SPECIAL_CHARACTERS for char in url
        ),
        "digits_count": sum(
            char.isdigit() for char in url
        ),
        "entropy": calculate_entropy(url)
    }

    return features


if __name__ == "__main__":
    test_urls = [
        "https://example.com/login/verify?id=123",
        "example.org/about",
        "http://192.168.1.1/login",
        "https://example.com/search?q=hello&lang=en"
    ]

    for url in test_urls:
        print("\nURL:", url)
        print(extract_url_features(url))

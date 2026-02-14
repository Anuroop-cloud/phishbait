"""
url_feature_extractor.py

Extract structural features from URLs found in user input text.
These features mirror the columns used in the PhiUSIIL Kaggle dataset
so that predictions from the URL model are meaningful.

Extracts 20 lightweight features purely from the URL string
(no network requests / page scraping).
"""

import re
import math
from urllib.parse import urlparse, parse_qs


# ── Top-Level Domains commonly seen in legitimate sites ──
LEGIT_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "co", "io", "us", "uk", "in", "de", "fr", "jp",
    "au", "ca", "br", "ru", "cn", "info", "biz", "co.in"
}

# ── Suspicious / free-hosting / shortener domains ────────
SUSPICIOUS_DOMAINS = {
    "bit.ly", "tinyurl.com", "rb.gy", "t.co", "goo.gl",
    "is.gd", "v.gd", "cutt.ly", "shorturl.at", "tiny.cc",
    "000webhostapp.com", "weebly.com", "blogspot.com",
    "herokuapp.com", "firebaseapp.com", "netlify.app",
}

# ── Phishing-bait keywords often embedded in URLs ────────
PHISH_KEYWORDS = {
    "login", "signin", "verify", "secure",
    "update", "confirm", "banking", "password", "credential",
    "suspend", "alert", "limited", "restore", "unlock",
}

URL_REGEX = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE
)


# ──────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────
def extract_urls(text: str) -> list[str]:
    """Return all URLs found in the input text."""
    return URL_REGEX.findall(text)


def extract_url_features(url: str) -> dict:
    """Extract 20 structural features from a single URL string.

    Returns a dict with human-readable keys and numeric values.
    """
    # Normalise
    if not url.startswith("http"):
        url = "http://" + url

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full = url.lower()

    # ── 1. Length-based ──────────────────────
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)

    # ── 2. Structural flags ─────────────────
    is_https = 1 if parsed.scheme == "https" else 0
    is_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(":")[0]) else 0

    # ── 3. Sub-domain analysis ──────────────
    parts = domain.split(".")
    # Remove port if present
    if ":" in parts[-1]:
        parts[-1] = parts[-1].split(":")[0]
    num_subdomains = max(0, len(parts) - 2)  # e.g. a.b.example.com → 2

    # ── 4. TLD analysis ─────────────────────
    tld = parts[-1] if parts else ""
    tld_length = len(tld)
    tld_is_legit = 1 if tld in LEGIT_TLDS else 0

    # ── 5. Character composition ────────────
    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    digit_ratio = num_digits / max(url_length, 1)
    letter_ratio = num_letters / max(url_length, 1)

    # Special characters in URL
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_at = url.count("@")
    num_special = sum(c in "=&?#%~|" for c in url)
    special_ratio = num_special / max(url_length, 1)

    # ── 6. Obfuscation signals ──────────────
    has_obfuscation = 1 if ("%" in url or "@" in parsed.netloc) else 0

    # ── 7. Suspicious domain flag ───────────
    # Exact/suffix match only (avoid substring false positives like
    # "microsoft.com" accidentally matching "t.co").
    is_suspicious_domain = 1 if any(
        domain == sd or domain.endswith(f".{sd}") for sd in SUSPICIOUS_DOMAINS
    ) else 0

    # ── 8. Phishing keyword count ───────────
    phish_keyword_count = sum(1 for kw in PHISH_KEYWORDS if kw in full)

    # ── 9. Query-string complexity ──────────
    num_params = len(parse_qs(query))

    # ── 10. Path depth ──────────────────────
    path_depth = len([seg for seg in path.split("/") if seg])

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "is_https": is_https,
        "is_domain_ip": is_ip,
        "num_subdomains": num_subdomains,
        "tld_length": tld_length,
        "tld_is_legit": tld_is_legit,
        "num_digits_in_url": num_digits,
        "digit_ratio": round(digit_ratio, 4),
        "letter_ratio": round(letter_ratio, 4),
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_at_signs": num_at,
        "num_special_chars": num_special,
        "special_char_ratio": round(special_ratio, 4),
        "has_obfuscation": has_obfuscation,
        "is_suspicious_domain": is_suspicious_domain,
        "phish_keyword_count": phish_keyword_count,
        "path_depth": path_depth,
    }


# ── Feature column order (matches training) ─────────────
URL_FEATURE_COLUMNS = list(extract_url_features("http://example.com").keys())


def get_triggered_features(features: dict) -> list[str]:
    """Return human-readable list of suspicious URL indicators that fired.

    Used by the explanation engine.
    """
    triggered: list[str] = []

    if not features["is_https"]:
        triggered.append("No HTTPS — connection is not encrypted")
    if features["is_domain_ip"]:
        triggered.append("Domain is a raw IP address (common in phishing)")
    if features["num_subdomains"] >= 3:
        triggered.append(f"Excessive subdomains ({features['num_subdomains']})")
    if features["url_length"] > 75:
        triggered.append(f"Unusually long URL ({features['url_length']} characters)")
    if features["is_suspicious_domain"]:
        triggered.append("URL uses a known suspicious or free-hosting domain")
    if features["has_obfuscation"]:
        triggered.append("URL contains obfuscation (percent-encoding or @ symbol)")
    if features["phish_keyword_count"] >= 2:
        triggered.append(
            f"Multiple phishing keywords found in URL ({features['phish_keyword_count']})"
        )
    if features["num_hyphens"] >= 3:
        triggered.append(f"Many hyphens in URL ({features['num_hyphens']} — often used to mimic domains)")
    if features["digit_ratio"] > 0.3:
        triggered.append("High proportion of digits in URL")
    if features["path_depth"] >= 5:
        triggered.append(f"Deep path structure ({features['path_depth']} levels)")

    return triggered


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "http://bank-secure-login.xyz/verify?id=12345",
        "https://www.google.com",
        "http://192.168.1.1/admin/login.php",
        "https://bit.ly/3xYz",
    ]
    for u in test_urls:
        feats = extract_url_features(u)
        triggers = get_triggered_features(feats)
        print(f"\nURL: {u}")
        print(f"  Features: {feats}")
        print(f"  Triggers: {triggers}")

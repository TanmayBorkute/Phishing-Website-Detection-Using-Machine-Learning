"""AI-mode feature extraction for the PhiUSIIL dataset (UCI #967).
The SAME functions compute features for the training URLs (train_ai.py) and for live scans (app.py), so there is no
train/serve mismatch for URL features. Page features are simple yes/no signals read from the fetched HTML."""
import re, math, time, socket, ipaddress
from urllib.parse import urlparse, urljoin
import requests

KEYWORDS = ("login", "signin", "secure", "account", "verify", "update", "bank", "paypal", "confirm", "password", "wallet", "support")
SHORT = re.compile(r"(^|\.)(bit\.ly|goo\.gl|tinyurl\.com|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|tiny\.cc|rebrand\.ly|cutt\.ly|shorturl\.at)$")
TITLES = {
 "url_length": "URL length", "host_length": "Domain length", "path_length": "Path length", "query_length": "Query length",
 "is_ip": "IP address as host", "has_port": "Explicit port", "is_https": "Uses HTTPS", "has_www": "Starts with www",
 "n_subdomains": "Sub-domains", "tld_length": "TLD length", "tld_legit_rate": "TLD legitimacy rate", "n_dots_host": "Dots in domain",
 "n_hyphens_host": "Hyphens in domain", "n_digits_host": "Digits in domain", "host_digit_ratio": "Digit ratio in domain",
 "host_entropy": "Domain randomness (entropy)", "longest_label": "Longest domain label", "n_digits_url": "Digits in URL",
 "digit_ratio": "Digit ratio in URL", "letter_ratio": "Letter ratio in URL", "n_special": "Special characters",
 "n_equals": "'=' characters", "n_qmark": "'?' characters", "n_amp": "'&' characters", "n_at": "'@' characters",
 "n_percent": "Percent-encoded chars", "path_depth": "Path depth", "double_slash_path": "'//' inside path",
 "n_keywords": "Suspicious keywords (URL)", "n_keywords_host": "Suspicious keywords in domain", "is_shortener": "URL shortener",
 "HasPasswordField": "Password field", "HasHiddenFields": "Hidden form fields", "HasSubmitButton": "Submit button",
 "HasExternalFormSubmit": "Form posts to other domain", "HasTitle": "Page title", "HasFavicon": "Favicon",
 "IsResponsive": "Responsive (viewport tag)", "HasDescription": "Meta description", "HasCopyrightInfo": "Copyright notice",
 "HasSocialNet": "Social-network links", "HasIframe": "Iframe",
}
# Model features: ONLY properties of the domain name. In PhiUSIIL every legitimate URL is a bare https://www.<domain> homepage,
# so https / www / path / query features are dataset artifacts (e.g. "has a path => 100% phishing") and are excluded from the models.
# They are still computed by lexical() for the URL-anatomy display and the red-flag rules.
LEX = ["host_length", "is_ip", "has_port", "n_subdomains", "tld_length", "tld_legit_rate", "n_dots_host", "n_hyphens_host",
       "n_digits_host", "host_digit_ratio", "host_entropy", "longest_label", "n_keywords_host", "is_shortener"]
PAGE_BOOL = [k for k in TITLES if k[0].isupper()]

def _entropy(s):
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s)) if s else 0.0

def norm_url(url):
    url = url.strip()
    return url if re.match(r"^https?://", url, re.I) else "https://" + url   # browsers default to https

def registered_domain(host):
    p = host.split(".")
    n = 3 if len(p) > 2 and len(p[-1]) == 2 and p[-2] in ("co", "com", "ac", "org", "gov", "net", "edu") else 2
    return ".".join(p[-n:])

def lexical(url):
    """URL string -> numeric features (tld_legit_rate is added later from the training table)."""
    url = norm_url(url); p = urlparse(url); host = (p.hostname or "").lower()
    bare = host[4:] if host.startswith("www.") else host   # PhiUSIIL legit URLs almost all start with www. -> strip it, or dots/length leak the label
    labels = [l for l in bare.split(".") if l]
    try: ipaddress.ip_address(host); is_ip = 1
    except ValueError: is_ip = 0
    core = [l for l in labels if l != "www"]
    letters = sum(c.isalpha() for c in url); digits = sum(c.isdigit() for c in url)
    hdig = sum(c.isdigit() for c in bare)
    low = url.lower()
    return {"tld": labels[-1] if labels else "",
        "url_length": len(url), "host_length": len(bare), "path_length": len(p.path), "query_length": len(p.query),
        "is_ip": is_ip, "has_port": int(p.port is not None), "is_https": int(p.scheme == "https"), "has_www": int(host.startswith("www.")),
        "n_subdomains": max(0, len(core) - 2), "tld_length": len(labels[-1]) if labels else 0, "n_dots_host": bare.count("."),
        "n_hyphens_host": bare.count("-"), "n_digits_host": hdig, "host_digit_ratio": hdig / len(bare) if bare else 0,
        "host_entropy": _entropy(bare), "longest_label": max((len(l) for l in labels), default=0),
        "n_digits_url": digits, "digit_ratio": digits / len(url), "letter_ratio": letters / len(url),
        "n_special": sum(not c.isalnum() and c not in "/:.-_" for c in url), "n_equals": url.count("="), "n_qmark": url.count("?"),
        "n_amp": url.count("&"), "n_at": url.count("@"), "n_percent": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "path_depth": len([s for s in p.path.split("/") if s]), "double_slash_path": int("//" in p.path),
        "n_keywords": sum(k in low for k in KEYWORDS), "n_keywords_host": sum(k in bare for k in KEYWORDS), "is_shortener": int(bool(SHORT.search(host)))}

def finalize(f, tld_rate, prior):
    f = dict(f); f["tld_legit_rate"] = tld_rate.get(f.pop("tld"), prior); return f

def _public(host):  # SSRF guard
    try:
        for i in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(i[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: return False
        return True
    except OSError: return False

def page_features(html, url):
    host = urlparse(url).hostname or ""
    ext_form = False
    for fm in re.findall(r"<form\b([^>]*)>", html):
        m = re.search(r'action=["\']([^"\']*)', fm)
        if m and m.group(1).startswith("http") and registered_domain(urlparse(m.group(1)).hostname or host) != registered_domain(host): ext_form = True
    return {"HasPasswordField": int(bool(re.search(r"type=[\"']?password", html))), "HasHiddenFields": int(bool(re.search(r"type=[\"']?hidden", html))),
        "HasSubmitButton": int(bool(re.search(r"type=[\"']?submit|<button", html))), "HasExternalFormSubmit": int(ext_form),
        "HasTitle": int(bool(re.search(r"<title>\s*\S", html))), "HasFavicon": int(bool(re.search(r"rel=[\"'][^\"']*icon", html))),
        "IsResponsive": int(bool(re.search(r"name=[\"']viewport", html))), "HasDescription": int(bool(re.search(r"name=[\"']description", html))),
        "HasCopyrightInfo": int("©" in html or "&copy;" in html or "copyright" in html),
        "HasSocialNet": int(bool(re.search(r"facebook\.com|twitter\.com|instagram\.com|linkedin\.com|youtube\.com", html))),
        "HasIframe": int("<iframe" in html)}

def extract(url, tld_rate, prior, fetch=True):
    """Returns (features, page_fetched, log). `log` lists the real steps with timings (replayed in the UI)."""
    log, t0 = [], time.time()
    step = lambda msg: log.append({"t": round((time.time() - t0) * 1000), "msg": msg})
    url = norm_url(url); host = (urlparse(url).hostname or "").lower()
    if not host or "." not in host or len(url) > 2000: raise ValueError("Invalid URL")
    step(f"parsed URL: host={host}")
    f = finalize(lexical(url), tld_rate, prior); step(f"computed {len(LEX)} lexical features")
    html = ""
    if fetch:
        ok = _public(host); step("resolved host to a public address" if ok else "host did not resolve to a public address - not fetching")
        if ok:
            try:
                r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0 PhishGuard"})
                if "html" in r.headers.get("content-type", "html"):
                    html = r.text[:600000].lower(); step(f"fetched page ({len(html):,} chars, status {r.status_code})")
            except requests.RequestException:
                step("page could not be fetched")
    if html:
        f.update(page_features(html, url)); step(f"read {len(PAGE_BOOL)} page signals")
    return f, bool(html), log

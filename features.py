"""URL -> UCI feature vector, implemented from the rules in 'Phishing Websites Features' (Mohammad, Thabtah, McCluskey).
Encoding (as in Training_Dataset.arff): 1 = legitimate-like, 0 = suspicious, -1 = phishing-like.
Only features computable without WHOIS / Alexa / PageRank / Google / PhishTank are extracted; the models for AI mode
are trained on exactly these features (see train.py), so nothing is guessed or neutral-filled."""
import re, socket, ssl, ipaddress
from urllib.parse import urlparse, urljoin
import requests

URL_SET = ["having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol", "double_slash_redirecting",
           "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State", "port", "HTTPS_token", "DNSRecord"]
PAGE_SET = URL_SET + ["Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
                      "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe"]
SHORT = re.compile(r"(^|\.)(bit\.ly|goo\.gl|tinyurl\.com|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|tiny\.cc|rebrand\.ly|cutt\.ly|shorturl\.at)$")
TRUSTED = re.compile(r"geotrust|godaddy|network solutions|thawte|comodo|doster|verisign|digicert|sectigo|globalsign|entrust|amazon|google trust", re.I)

def shift1(X):
    """Maps -1/0/1 -> 0/1/2 for CategoricalNB (module-level so joblib can pickle the pipeline)."""
    return X + 1

def _t(title, **o): return {"title": title, "options": {int(k.replace("m", "-").replace("_", "")): v for k, v in o.items()}}
INFO = {  # label text for each allowed value (shown in the manual form)
 "having_IP_Address": _t("IP address in URL", m1="IP address (decimal/hex)", _1="Domain name"),
 "URL_Length": _t("URL length", m1="> 75 characters", _0="54–75 characters", _1="< 54 characters"),
 "Shortining_Service": _t("URL shortening service", m1="Shortener (bit.ly, tinyurl…)", _1="No shortener"),
 "having_At_Symbol": _t("'@' symbol in URL", m1="'@' present", _1="No '@'"),
 "double_slash_redirecting": _t("'//' redirect", m1="Last '//' after position 7", _1="No redirecting '//'"),
 "Prefix_Suffix": _t("'-' in domain", m1="Dash in domain", _1="No dash"),
 "having_Sub_Domain": _t("Sub-domains", m1="3+ dots (multiple sub-domains)", _0="2 dots (one sub-domain)", _1="≤ 1 dot"),
 "SSLfinal_State": _t("HTTPS / SSL certificate", m1="No HTTPS", _0="HTTPS, untrusted issuer", _1="HTTPS, trusted issuer"),
 "Domain_registeration_length": _t("Domain registration length", m1="Expires within 1 year", _1="Expires in > 1 year"),
 "Favicon": _t("Favicon source", m1="External domain", _1="Same domain"),
 "port": _t("Port", m1="Non-standard / risky port", _1="Standard port"),
 "HTTPS_token": _t("'https' token in domain", m1="Token in domain part", _1="Not in domain"),
 "Request_URL": _t("Request URL (external objects)", m1="≥ 22% external", _1="< 22% external"),
 "URL_of_Anchor": _t("URL of anchors", m1="> 67% unsafe/external", _0="31–67%", _1="< 31%"),
 "Links_in_tags": _t("Links in meta/script/link", m1="> 81% external", _0="17–81%", _1="< 17%"),
 "SFH": _t("Server form handler", m1="Empty / about:blank", _0="Different domain", _1="Same domain / none"),
 "Submitting_to_email": _t("Submits to email", m1="mailto: / mail()", _1="No"),
 "Abnormal_URL": _t("Abnormal URL (WHOIS)", m1="Host name not in URL", _1="Host name in URL"),
 "Redirect": _t("Page redirects", _0="2+ redirects", _1="≤ 1 redirect"),
 "on_mouseover": _t("Status-bar customisation", m1="onMouseOver changes status bar", _1="No"),
 "RightClick": _t("Right click", m1="Disabled", _1="Enabled"),
 "popUpWidnow": _t("Pop-up window", m1="Pop-up with text fields", _1="No"),
 "Iframe": _t("IFrame", m1="iframe used", _1="No iframe"),
 "age_of_domain": _t("Age of domain", m1="< 6 months", _1="≥ 6 months"),
 "DNSRecord": _t("DNS record", m1="No record", _1="Record found"),
 "web_traffic": _t("Website traffic rank", m1="Not ranked", _0="Rank > 100,000", _1="Rank < 100,000"),
 "Page_Rank": _t("PageRank", m1="< 0.2", _1="≥ 0.2"),
 "Google_Index": _t("Google index", m1="Not indexed", _1="Indexed"),
 "Links_pointing_to_page": _t("Links pointing to page", m1="None", _0="1–2 links", _1="> 2 links"),
 "Statistical_report": _t("Statistical report", m1="In top phishing IPs/domains", _1="Not listed"),
}

def _is_ip(h):
    try: ipaddress.ip_address(h); return True
    except ValueError: pass
    parts = h.split(".")  # hexadecimal / decimal forms, e.g. 0x58.0xCC.0xCA.0x62
    return bool(parts) and all(re.fullmatch(r"0x[0-9a-f]+|\d+", p) for p in parts) and (len(parts) == 4 or h.startswith("0x") or h.isdigit())

def _dots(host):  # doc: drop 'www.', drop ccTLD, count remaining dots
    parts = (host[4:] if host.startswith("www.") else host).split(".")
    if len(parts) > 1 and len(parts[-1]) == 2: parts = parts[:-1]
    return len(parts) - 1

def _root(h):
    p = h.split(".")
    n = 3 if len(p) > 2 and len(p[-1]) == 2 and p[-2] in ("co", "com", "ac", "org", "gov", "net", "edu") else 2
    return ".".join(p[-n:])

def _public(host):  # SSRF guard: only fetch hosts that resolve to public addresses
    try:
        for i in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(i[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: return False
        return True
    except OSError:
        return False

def _ssl_state(host, port):
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((host, port or 443), timeout=4) as s, ctx.wrap_socket(s, server_hostname=host) as t:
            issuer = " ".join(v for rdn in t.getpeercert()["issuer"] for _, v in rdn)
        return 1 if TRUSTED.search(issuer) else 0
    except ssl.SSLError:
        return 0           # HTTPS served but certificate not verifiable/trusted
    except OSError:
        return -1

def _grade(pct, lo, hi): return 1 if pct < lo else (0 if pct <= hi else -1)

def extract(url, fetch=True):
    """Returns (features dict, page_fetched)."""
    if not re.match(r"^https?://", url, re.I): url = "http://" + url
    p = urlparse(url); host = (p.hostname or "").lower()
    if not host or ("." not in host and not host.startswith("0x")): raise ValueError("Invalid URL")
    dots = _dots(host)
    f = {"having_IP_Address": -1 if _is_ip(host) else 1,
         "URL_Length": 1 if len(url) < 54 else (0 if len(url) <= 75 else -1),
         "Shortining_Service": -1 if SHORT.search(host) else 1,
         "having_At_Symbol": -1 if "@" in url else 1,
         "double_slash_redirecting": -1 if url.rfind("//") > 7 else 1,
         "Prefix_Suffix": -1 if "-" in host else 1,
         "having_Sub_Domain": 1 if dots <= 1 else (0 if dots == 2 else -1),
         "SSLfinal_State": _ssl_state(host, p.port) if p.scheme == "https" and fetch else (-1 if p.scheme != "https" else 0),
         "port": -1 if p.port not in (None, 80, 443) else 1,
         "HTTPS_token": -1 if "https" in host else 1}
    try: socket.gethostbyname(host); f["DNSRecord"] = 1
    except OSError: f["DNSRecord"] = -1
    html = ""
    if fetch and _public(host):
        try:
            r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0 PhishGuard"})
            if "html" in r.headers.get("content-type", "html"):
                html, hops = r.text[:600000].lower(), len(r.history)
        except requests.RequestException:
            pass
    if not html: return f, False
    ext = lambda us: 100 * sum((_root(urlparse(urljoin(url, u)).hostname or host) != _root(host)) for u in us) / len(us) if us else 0
    attr = lambda tag, a: re.findall(rf'<{tag}\b[^>]*?\b{a}=["\']([^"\']+)', html)
    media = re.findall(r'<(?:img|video|audio|embed|source|iframe)\b[^>]*?\bsrc=["\']([^"\']+)', html)
    anchors = re.findall(r'<a\b[^>]*?\bhref=["\']([^"\']*)', html)
    unsafe = sum(a.startswith("#") or a == "" or a.startswith("javascript") or
                 _root(urlparse(urljoin(url, a)).hostname or host) != _root(host) for a in anchors)
    tagurls = attr("script", "src") + attr("link", "href") + re.findall(r'<meta\b[^>]*?\bcontent=["\'](https?://[^"\']+)', html)
    sfh = 1
    for fm in re.findall(r"<form\b([^>]*)>", html):
        m = re.search(r'action=["\']([^"\']*)', fm)
        if m:
            a = m.group(1)
            sfh = min(sfh, -1 if a in ("", "about:blank") else (0 if a.startswith("http") and _root(urlparse(a).hostname or host) != _root(host) else 1))
    icons = [re.search(r'href=["\']([^"\']+)', t) for t in re.findall(r"<link\b[^>]*>", html) if "icon" in t]
    f.update({"Favicon": -1 if any(m and ext([m.group(1)]) > 0 for m in icons) else 1,
              "Request_URL": 1 if ext(media) < 22 else -1,                      # dataset has only {1,-1}
              "URL_of_Anchor": _grade(100 * unsafe / len(anchors) if anchors else 0, 31, 67),
              "Links_in_tags": _grade(ext(tagurls), 17, 81),
              "SFH": sfh,
              "Submitting_to_email": -1 if ("mailto:" in html or "mail(" in html) else 1,
              "Redirect": 1 if hops <= 1 else 0,                                # dataset has only {0,1}
              "on_mouseover": -1 if re.search(r"onmouseover\s*=\s*[\"'][^\"']*window\.status", html) else 1,
              "RightClick": -1 if re.search(r"event\.button\s*==\s*2|oncontextmenu\s*=\s*[\"']\s*return\s+false", html) else 1,
              "popUpWidnow": -1 if ("window.open(" in html and "<input" in html) else 1,
              "Iframe": -1 if "<iframe" in html else 1})
    return f, True

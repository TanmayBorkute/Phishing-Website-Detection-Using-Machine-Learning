"""Checks extract() against the examples given in the feature-description document (offline: fetch=False)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract
def f(u): return extract(u, fetch=False)[0]
assert f("http://125.98.3.123/fake.html")["having_IP_Address"] == -1
assert f("http://0x58.0xCC.0xCA.0x62/2/paypal.ca/index.html")["having_IP_Address"] == -1
assert f("bit.ly/19DXSk4")["Shortining_Service"] == -1
assert f("http://www.legitimate.com//http://www.phishing.com")["double_slash_redirecting"] == -1
assert f("http://www.Confirme-paypal.com/")["Prefix_Suffix"] == -1
assert f("http://www.hud.ac.uk/students/")["having_Sub_Domain"] == 1 and f("http://www.hud.ac.uk/students/")["Prefix_Suffix"] == 1
assert f("http://https-www-paypal-it-webapps-mpp-home.soft-hair.com/")["HTTPS_token"] == -1
long = "http://federmacedoadv.com.br/3f/aze/ab51e2e319e51502f416dbe46b773a5e/?cmd=_home&dispatch=11004d58f5b74f8dc1e7c2e8dd4105e811004d58f5b74f8dc1e7c2e8dd4105e8@phishing.website.html"
assert f(long)["URL_Length"] == -1 and f(long)["having_At_Symbol"] == -1
assert f("http://a.b.c.example.com/")["having_Sub_Domain"] == -1 and f("http://mail.example.com/")["having_Sub_Domain"] == 0
assert f("http://example.com:8080/")["port"] == -1 and f("http://example.com/")["SSLfinal_State"] == -1
print("all doc examples pass")

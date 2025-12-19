import re
import socket
import requests
import whois
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup


def extract_features_from_url(url: str) -> list:
    features = []

    parsed = urlparse(url)
    domain = parsed.netloc

    # 1. having_IP_Address
    try:
        socket.inet_aton(domain)
        features.append(1)
    except:
        features.append(0)

    # 2. URL_Length
    features.append(1 if len(url) < 54 else 0)

    # 3. Shortining_Service
    shortining_services = r"bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co"
    features.append(0 if re.search(shortining_services, url) else 1)

    # 4. having_At_Symbol
    features.append(0 if "@" in url else 1)

    # 5. double_slash_redirecting
    features.append(0 if url.rfind("//") > 6 else 1)

    # 6. Prefix_Suffix
    features.append(0 if "-" in domain else 1)

    # 7. having_Sub_Domain
    features.append(0 if domain.count('.') > 2 else 1)

    # 8. SSLfinal_State
    features.append(1 if parsed.scheme == "https" else 0)

    # 9. Domain_registeration_length
    try:
        w = whois.whois(domain)
        expiration = w.expiration_date
        creation = w.creation_date
        if isinstance(expiration, list):
            expiration = expiration[0]
        if isinstance(creation, list):
            creation = creation[0]
        features.append(1 if (expiration - creation).days > 365 else 0)
    except:
        features.append(0)

    # 10. Favicon
    features.append(1 if "favicon" in url.lower() else 0)

    # 11. port
    features.append(0 if parsed.port else 1)

    # 12. HTTPS_token
    features.append(0 if "https" in domain else 1)

    # ---- HTML based features ----
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        # 13. Request_URL
        features.append(1)

        # 14. URL_of_Anchor
        features.append(1)

        # 15. Links_in_tags
        features.append(1)

        # 16. SFH
        features.append(1)

        # 17. Submitting_to_email
        features.append(0 if soup.find("a", href=re.compile("mailto")) else 1)

        # 18. Abnormal_URL
        features.append(1)

        # 19. Redirect
        features.append(0 if response.history else 1)

        # 20. on_mouseover
        features.append(0 if "onmouseover" in response.text else 1)

        # 21. RightClick
        features.append(0 if "event.button==2" in response.text else 1)

        # 22. popUpWidnow
        features.append(0 if "window.open" in response.text else 1)

        # 23. Iframe
        features.append(0 if soup.find("iframe") else 1)

    except:
        # Default safe values if page not reachable
        features.extend([0] * 11)

    # 24. age_of_domain
    try:
        age = (datetime.now() - creation).days
        features.append(1 if age > 180 else 0)
    except:
        features.append(0)

    # 25. DNSRecord
    try:
        socket.gethostbyname(domain)
        features.append(1)
    except:
        features.append(0)

    return features
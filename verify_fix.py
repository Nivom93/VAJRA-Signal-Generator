try:
    import defusedxml.ElementTree as ET
    from defusedxml import DefusedXmlException
except ImportError:
    print("defusedxml is not installed. This script requires it to verify the fix.")
    exit(0)

import os

# Malicious XML with XXE
xml_with_xxe = """<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd" >]>
<rss version="2.0">
<channel>
  <item>
    <title>&xxe;</title>
  </item>
</channel>
</rss>
"""

def test_defusedxml():
    print("Testing with defusedxml.ElementTree:")
    try:
        # defusedxml should raise an error when encountering DTD/entities by default
        root = ET.fromstring(xml_with_xxe)
        print("VULNERABLE: defusedxml allowed XXE!")
        for item in root.findall('.//item'):
            title_elem = item.find('title')
            print(f"Title: {title_elem.text}")
    except DefusedXmlException as e:
        print(f"SUCCESS: defusedxml blocked the attack: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")

if __name__ == "__main__":
    test_defusedxml()



import hashlib

pw = '141421'
h = hashlib.sha256(pw.encode()).hexdigest()
target = 'cfb24c91a9b83d9967f5b6a177037f5803abf3c8a84771a62c4fa48ab076434f0'

with open('result.txt', 'w', encoding='utf-8') as f:
    f.write(f"Calculated hash: {h}\n")
    f.write(f"Target hash:     {target}\n")
    f.write(f"Match: {h == target}\n")


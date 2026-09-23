import glob
import re

files = glob.glob('*.py') + glob.glob('src/**/*.py', recursive=True)
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace backslash quote with just quote
    content = content.replace('\"', '"')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print('Fixed quotes in all files')

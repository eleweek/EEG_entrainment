import re

def parse_picks(picks):
    return re.split(r',\s*|\s+', picks) if picks else None
import re

def parse_ddi(raw):
    raw = raw.strip()
    ddi, description = raw.splitlines()[0].split("DD Entity: ")[1].split(" ", 1)
    return int(ddi), description, raw

def read_ddis(ddi_file):
    with open(ddi_file, "r") as f:
        ddi_text = f.read()
    entries = re.split(r"\n\n", ddi_text)
    DDIs = {}
    for entry in entries:
        if entry.strip().startswith("DD Entity"):
            ddi, description, raw = parse_ddi(entry)
            DDIs[ddi] = {"description" : description , "full_description" : raw}
    return DDIs


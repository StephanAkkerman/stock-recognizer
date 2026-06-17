import glob
import json

count = 0
for fp in glob.glob("data/augmented/*.json"):
    for task in json.load(open(fp, encoding="utf-8")):
        text = task["data"]["text"]
        for ann in task.get("annotations") or []:
            for r in ann.get("result", []):
                if r.get("type") == "labels":
                    v = r["value"]
                    span = text[v["start"] : v["end"]]
                    if span.endswith(chr(0x2019) + "s") or span.endswith("'s"):
                        count += 1
                        print(repr(span))
print(f"Total possessive spans in augmented: {count}")

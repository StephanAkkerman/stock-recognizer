from gliner2 import GLiNER2

# Try v5 with the EXACT labels used at training time (no descriptions):
model = GLiNER2.from_pretrained(
    "fastino/gliner2-large-v1", map_location="cuda", quantize=False
)
model.load_adapter("models/reddit_adapter_v5/final")
print(
    model.batch_extract_entities(
        ["I bought $AAPL and Tesla today."],
        ["ticker", "company"],
        threshold=0.1,
        include_spans=True,
    )
)

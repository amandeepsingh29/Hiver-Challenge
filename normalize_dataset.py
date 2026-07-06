import json

with open("dataset.json", "r") as f:
    dataset = json.load(f)

for item in dataset:
    if "customer_email" in item:
        item["incoming"] = item.pop("customer_email")
    if "ideal_reference_reply" in item:
        item["reference_reply"] = item.pop("ideal_reference_reply")
    if "scenario_name" in item:
        item["category"] = item.pop("scenario_name")

with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Dataset normalized.")

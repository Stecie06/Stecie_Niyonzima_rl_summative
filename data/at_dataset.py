import os
import csv

CSV_PATH = os.path.join(os.path.dirname(__file__), "at_data.csv")

COVERAGE_SCORES = {
    "Total coverage":   1.0,
    "Partial coverage": 0.7,
    "No coverage":      0.3,
    "No information":   0.5,
}

FIELD_SCORES = {
    "Yes":                    1.0,
    "No":                     0.0,
    "Information not available": 0.5,
}

def load_country_data(country: str = "Rwanda") -> dict:
    if not os.path.exists(CSV_PATH):
        return {
            "country":       country,
            "summary":       "Partial coverage",
            "cognition":     "No",
            "communication": "No",
            "hearing":       "Yes",
            "mobility":      "Yes",
            "self_care":     "Yes",
            "vision":        "Yes",
        }

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows[3:]:
        if row and row[0].strip('"').strip() == country:
            return {
                "country":       country,
                "summary":       row[1].strip('"').strip(),
                "cognition":     row[2].strip('"').strip(),
                "communication": row[3].strip('"').strip(),
                "hearing":       row[4].strip('"').strip(),
                "mobility":      row[5].strip('"').strip(),
                "self_care":     row[6].strip('"').strip(),
                "vision":        row[7].strip('"').strip(),
            }

    return {"country": country, "summary": "No information"}

def compute_at_weight(country: str = "Rwanda") -> float:
    data = load_country_data(country)

    summary_score = COVERAGE_SCORES.get(data.get("summary", ""), 0.5)

    field_keys = ["cognition", "communication", "hearing",
                  "mobility", "self_care", "vision"]
    field_scores = [FIELD_SCORES.get(data.get(k, ""), 0.5) for k in field_keys]
    avg_fields = sum(field_scores) / len(field_scores)

    vision_score   = FIELD_SCORES.get(data.get("vision",   ""), 0.5)
    mobility_score = FIELD_SCORES.get(data.get("mobility", ""), 0.5)

    weight = (
        0.35 * summary_score +
        0.25 * avg_fields    +
        0.25 * vision_score  +
        0.15 * mobility_score
    )
    return round(float(weight), 4)

AT_WEIGHT = compute_at_weight("Rwanda")

if __name__ == "__main__":
    data = load_country_data("Rwanda")
    print("Rwanda AT data:", data)
    print("AT Weight:", AT_WEIGHT)

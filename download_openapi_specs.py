from pathlib import Path
import requests

OUT_DIR = Path("openapi_specs")
OUT_DIR.mkdir(exist_ok=True)

SPECS = {
    # Adobe Experience Platform APIs
    "catalog.yaml": "https://raw.githubusercontent.com/AdobeDocs/experience-platform-apis/main/static/swagger-specs/catalog.yaml",
    "profile.yaml": "https://raw.githubusercontent.com/AdobeDocs/experience-platform-apis/main/static/swagger-specs/profile.yaml",
    "data-access.yaml": "https://raw.githubusercontent.com/AdobeDocs/experience-platform-apis/main/static/swagger-specs/data-access.yaml",
    "flow-service.yaml": "https://raw.githubusercontent.com/AdobeDocs/experience-platform-apis/main/static/swagger-specs/flow-service.yaml",
    "segmentation.yaml": "https://raw.githubusercontent.com/AdobeDocs/experience-platform-apis/main/static/swagger-specs/segmentation.yaml",
    "unified-tags.yaml": "https://raw.githubusercontent.com/AdobeDocs/experience-platform-apis/main/static/swagger-specs/unified-tags.yaml",
    "schema-registry.yaml": "https://raw.githubusercontent.com/AdobeDocs/experience-platform-apis/main/static/swagger-specs/schema-registry.yaml",

    # Adobe Journey Optimizer APIs
    "journey-retrieve.yaml": "https://raw.githubusercontent.com/AdobeDocs/journey-optimizer-apis/main/static/journey-retrieve.yaml",
}

def download_file(name: str, url: str) -> None:
    path = OUT_DIR / name
    print(f"Downloading {name}...")

    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        print(f"  FAILED: HTTP {response.status_code}")
        return

    text = response.text

    if "openapi:" not in text and "swagger:" not in text:
        print("  WARNING: downloaded file may not be OpenAPI YAML")

    path.write_text(text, encoding="utf-8")
    print(f"  Saved to {path}")

def main():
    for name, url in SPECS.items():
        download_file(name, url)

if __name__ == "__main__":
    main()
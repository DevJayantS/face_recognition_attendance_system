import os
import urllib.request


BOOTSTRAP_CSS_URL = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
BOOTSTRAP_JS_URL = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
FONTAWESOME_CSS_URL = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download(url: str, dest_path: str) -> None:
    print(f"Downloading {url} -> {dest_path}")
    with urllib.request.urlopen(url) as resp:
        content = resp.read()
    with open(dest_path, "wb") as f:
        f.write(content)


def main() -> None:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor_dir = os.path.join(base, "static", "vendor")
    bootstrap_dir = os.path.join(vendor_dir, "bootstrap")
    fa_dir = os.path.join(vendor_dir, "fontawesome", "css")

    ensure_dir(bootstrap_dir)
    ensure_dir(fa_dir)

    download(BOOTSTRAP_CSS_URL, os.path.join(bootstrap_dir, "bootstrap.min.css"))
    download(BOOTSTRAP_JS_URL, os.path.join(bootstrap_dir, "bootstrap.bundle.min.js"))
    download(FONTAWESOME_CSS_URL, os.path.join(fa_dir, "all.min.css"))

    print("Done. Offline assets saved under static/vendor/.")


if __name__ == "__main__":
    main()



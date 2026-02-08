import os
import pdfplumber
from langchain_core.documents import Document


def load_texts(path):
    docs = []

    for root, _, files in os.walk(path):
        for f in files:
            full_path = os.path.join(root, f)

            # ===== TXT =====
            if f.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as file:
                    text = file.read()

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": full_path,
                            "type": "txt"
                        }
                    )
                )

            # ===== PDF =====
            elif f.endswith(".pdf"):
                with pdfplumber.open(full_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()

                        if text and text.strip():
                            docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": full_path,
                                        "page": i + 1,
                                        "type": "pdf"
                                    }
                                )
                            )

    return docs

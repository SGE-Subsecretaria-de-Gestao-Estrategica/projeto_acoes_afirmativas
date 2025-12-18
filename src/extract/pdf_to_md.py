# pdf_to_md.py
from pathlib import Path
from docling.document_converter import DocumentConverter
import unicodedata
import csv


# Configurações
CONVERTER = DocumentConverter()

# Controle de conversão
def controle_conversao(log_path: str):
    processados = set()
    if Path(log_path).exists():
        with open(log_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                if row["status"] == "ok":  # só considera concluídos
                    processados.add(row["pdf_path"])
    
    return processados

def pdf_to_md(log_path:str, pdf_root:str, md_root:str):
    processados = controle_conversao(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        if not Path(log_path).exists():
            writer.writerow(["pdf_path", "md_path", "status", "mensagem"])

        # Itera por todos os PDFs
        pdfs = list(pdf_root.rglob("*.pdf"))
        print(f"Encontrados {len(pdfs)} PDFs")
        for pdf_path in pdf_root.rglob("*.pdf"):
            pdf_path = Path(unicodedata.normalize("NFC", str(pdf_path))).resolve()
            
            if str(pdf_path) in processados:
                print(f"⏩ Pulando (já processado): {pdf_path}")
                continue

            relative_path = pdf_path.relative_to(pdf_root)
            md_path = md_root / relative_path.parent / (pdf_path.stem + ".md")
            md_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                result = CONVERTER.convert(pdf_path)
                md_text = result.document.export_to_markdown()

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_text)

                print(f"✅ Convertido: {pdf_path} → {md_path}")
                writer.writerow([str(pdf_path), str(md_path), "ok", ""])

            except Exception as e:
                print(f"❌ Erro ao processar {pdf_path}: {e}")
                writer.writerow([str(pdf_path), str(md_path), "erro", str(e)])

    print(f"\n📊 Log de conversão atualizado em: {log_path}")



    

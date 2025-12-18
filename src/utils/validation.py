from pathlib import Path

def contar_editais_pdf(base_path: Path) -> int:
    """
    Conta arquivos PDF recursivamente dentro de um diretório.
    """
    return len(list(base_path.rglob("*.pdf")))


def contar_editais_md(base_path: Path) -> int:
    """
    Conta arquivos PDF recursivamente dentro de um diretório.
    """
    return len(list(base_path.rglob("*.md")))
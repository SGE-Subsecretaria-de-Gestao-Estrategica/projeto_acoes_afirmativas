📑 PNAB Ações Afirmativas

Extração inteligente de informações de editais do ciclo 1 da PNAB (Programa Nacional Aldir Blanc de Fomento à Cultura)

🚀 Visão Geral

Este projeto foi desenvolvido para interpretar, estruturar e analisar editais culturais do ciclo 1 da PNAB (Programa Nacional Aldir Blanc de Fomento à Cultura), transformando documentos em PDF em informações computacionais acessíveis.

A pipeline utiliza processamento de texto, pré-processamento estruturado e modelos de linguagem (LLMs) para identificar e organizar dados-chave, com destaque para a detecção de ações afirmativas (cotas, vagas reservadas, critérios de inclusão etc.).

🛠️ Funcionalidades

📂 Leitura de PDFs

Interpretação automática de editais em PDF.

Conversão para linguagem computacional estruturada.

🧹 Pré-processamento inteligente

Criação de filtros para limpeza e padronização de textos.

Divisão dos editais em chunks (partes relevantes para análise).

🤖 Extração de informações com LLMs

Identificação de menções a ações afirmativas.

Retorno estruturado em formato padronizado (ex.: JSON).

📊 Objetivos

Facilitar o monitoramento de políticas públicas de fomento cultural.

Apoiar o Programa Nacional Aldir Blanc na análise de editais.

Fornecer uma base estruturada para dashboards, relatórios e análises.

📂 Estrutura do Projeto
├── data/               # PDFs e arquivos de entrada
├── outputs/            # Resultados processados (JSON, CSV, etc.)
├── src/                # Código-fonte principal
│   ├── pdf_extractor/  # Módulos de extração de PDFs
│   ├── preprocessing/  # Limpeza e chunking dos textos
│   ├── llm/            # Rotinas de interação com modelos de linguagem
│   └── utils/          # Funções auxiliares
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo

⚙️ Como usar

Clonar o repositório

git clone https://github.com/seu-usuario/pnab-edital-extractor.git
cd pnab-edital-extractor


Criar ambiente virtual e instalar dependências

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt


Rodar o pipeline de extração

python src/main.py data/exemplo_edital.pdf

🌱 Contribuições

Contribuições são bem-vindas! 💡
Se você tem sugestões de melhorias, novas features ou encontrou algum problema, fique à vontade para abrir uma issue ou enviar um pull request.

📜 Licença

Este projeto está sob a licença MIT.


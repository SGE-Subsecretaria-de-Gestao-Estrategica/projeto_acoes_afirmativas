# PNAB - Ações Afirmativas

Extração e sistematização de dados dos editais estaduais e das capitais referentes ao ciclo 1 da PNAB (Programa Nacional Aldir Blanc de Fomento à Cultura).


# Visão Geral

Este projeto foi desenvolvido para interpretar, estruturar e analisar editais culturais do ciclo 1 da PNAB (Programa Nacional Aldir Blanc de Fomento à Cultura), transformando documentos em PDF em informações computacionais acessíveis.

A pipeline utiliza processamento de texto, pré-processamento estruturado e modelos de linguagem (LLMs) para identificar e organizar dados-chave, com destaque para a detecção de ações afirmativas (cotas, vagas reservadas, critérios de inclusão etc.).


# Funcionalidades

## Leitura de PDFs

Interpretação automática de editais em PDF.

Conversão para linguagem computacional estruturada.


## Pré-processamento inteligente

Criação de filtros para limpeza e padronização de textos.

Divisão dos editais em chunks (partes relevantes para análise).


## Extração de informações com LLMs

Identificação de menções a ações afirmativas.

Retorno estruturado em formato padronizado (ex.: JSON).


# Objetivos

Facilitar o monitoramento de políticas públicas de fomento cultural.

Apoiar o Programa Nacional Aldir Blanc na análise de editais.

Fornecer uma base estruturada para dashboards, relatórios e análises.



# Estrutura do Projeto
<br>    ├── data/     # PDFs e arquivos de entrada
<br>    ├── outputs/  # Resultados processados (JSON, CSV, etc.)
<br>    ├── src/   # Código-fonte principal
<br>    &nbsp;&nbsp;&nbsp;&nbsp;   ├── main.py   # Código principal
<br>    &nbsp;&nbsp;&nbsp;&nbsp;   ├── regex_patterns.py   # Padrões de regex utilizados nos filtros
<br>    &nbsp;&nbsp;&nbsp;&nbsp;   └── utils.py   # Funções auxiliares
<br>    ├── requirements.txt # Dependências do projeto
<br>    └── README.md # Este arquivo

# Fluxo de extração e transformação dos dados

O fluxo de extração e transformação dos dados adotado pela pesquisa pode ser sintetizado nas seguintes etapas:

## 1. Conversão de PDF para Markdown

Inicialmente, os arquivos em PDF foram convertidos para o formato **Markdown**, um padrão textual leve e legível que preserva a hierarquia e a organização lógica do conteúdo, facilitando a análise posterior.  

Utilizou-se a biblioteca **Docling**, ferramenta de código aberto reconhecida por sua robustez na preservação da estrutura textual e por contar com ampla comunidade de manutenção.

---

## 2. Limpeza e padronização dos dados

O texto convertido passou por um processo de sanitização, com:

- Remoção de espaços excessivos  
- Exclusão de fragmentos soltos sem conexão semântica  
- Eliminação de marcadores residuais de formatação típicos de PDFs  
- Remoção de caracteres de controle ASCII (como `[\x00-\x1f\x7f-\x9f]`), que frequentemente comprometem a leitura automatizada  

---

## 3. Divisão do texto em blocos interpretáveis

Diante da heterogeneidade dos documentos, não foi possível identificar parágrafos de forma consistente. Assim, cada PDF (considerado como unidade de análise) foi segmentado em:

- Blocos de **800 caracteres** (incluindo espaços)  
- Sobreposição (*overlap*) de aproximadamente **100 caracteres**

A sobreposição assegura continuidade contextual entre blocos adjacentes, evitando perda de sentido quando informações relevantes se situam na fronteira entre trechos.

Cada bloco foi programado para se encerrar apenas após:

- Um ponto final (`.`)  
- Um marcador de quebra de linha (`\n`)  

Esse critério preserva a integridade textual dos parágrafos.

---

## 4. Filtragem por expressões regulares

Foram utilizadas **expressões regulares (regex)** para identificar padrões textuais relevantes de forma concisa e flexível.

Com base nessa técnica:

- Elaborou-se um dicionário conceitual a partir de palavras-chave associadas às informações de interesse  
- Definiram-se padrões de busca para localizar conteúdos específicos nos editais  

Esse procedimento permitiu filtrar blocos com precisão, mesmo diante de variações na redação e na estrutura dos documentos.

---

## 5. Aplicação de modelo de linguagem de grande porte (LLM)

Os blocos filtrados foram utilizados como *input* para o modelo de linguagem **GPT-4o-mini**, escolhido pelo equilíbrio entre desempenho e custo.

O modelo foi configurado com **temperatura 0**, reduzindo ao mínimo a variação nas respostas. Na prática, isso significa que, diante do mesmo estímulo, o modelo responde sempre da mesma forma, assegurando maior reprodutibilidade e precisão na extração dos dados.

---

## 6. Avaliação do fluxo de extração

A acurácia do processo foi avaliada por meio de amostragem em **70 editais** (20% da base total de 351).

Utilizou-se como métrica a **F1 Score**, definida pela combinação de:

- **Precisão (Precision):** proporção de acertos entre as previsões positivas  
- **Revocação (Recall):** proporção de acertos em relação às ocorrências reais  

O resultado aproximado de **97%** indica alto equilíbrio entre precisão e completude, confirmando a eficácia do modelo, com extração consistente e incidência mínima de erros.


# Como usar

Clonar o repositório

git clone https://github.com/seu-usuario/pnab-edital-extractor.git
cd pnab-edital-extractor


Criar ambiente virtual e instalar dependências
 
 ```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Rodar o pipeline de extração

```
python src/main.py data/exemplo_edital.pdf
```



# 📜 Licença

Este projeto está sob a licença MIT.



# Ações Afirmativas na Política Nacional Aldir Blanc – Ciclo 1: Análise da Implementação nos Estados, DF e Capitais (2023–2025)


**Acesso aos dados consolidados:** [Repositório de editais levantados na pesquisa Aldir Blanc – Ciclo 1 estados DF e capitais](https://culturagov-my.sharepoint.com/:f:/g/personal/sniic_cultura_gov_br/IgBqI1YVPorFQ7WUo--s2PxZAYnL2nC0J5n5zxgVImNCqlQ?e=Ylf756)  
*Este repositório contém a base de dados simplificada gerada a partir do processo de extração e sistematização descrito abaixo.*


## 📖 Visão Geral

* Este projeto foi desenvolvido para interpretar, estruturar e analisar editais culturais do ciclo 1 da Política Nacional Aldir Blanc de Fomento à Cultura, transformando documentos em PDF em informações computacionais acessíveis.

* A pipeline utiliza processamento de texto, pré-processamento estruturado e modelos de linguagem (LLMs) para identificar e organizar dados-chave, com destaque para a detecção de ações afirmativas (cotas, vagas reservadas, critérios de inclusão etc.).


## 🛠️ Funcionalidades

### Leitura de PDFs

- Interpretação automática de editais em PDF.

- Conversão para linguagem computacional estruturada.


### Pré-processamento inteligente

- Criação de filtros para limpeza e padronização de textos.

- Divisão dos editais em chunks (partes relevantes para análise).


### Extração de informações com LLMs

- Identificação de menções a ações afirmativas.

- Retorno estruturado em formato padronizado (ex.: JSON).


## 🎯 Objetivos

- Facilitar o monitoramento de políticas públicas de fomento cultural.

- Apoiar o Programa Nacional Aldir Blanc na análise de editais.

- Fornecer uma base estruturada para dashboards, relatórios e análises.


## 📂 Estrutura do Projeto
<br>    ├── data/     # PDFs e arquivos de entrada
<br>    ├── outputs/  # Resultados processados (JSON, CSV, etc.)
<br>    ├── src/   # Código-fonte principal
<br>    &nbsp;&nbsp;&nbsp;&nbsp;   ├── main.py   # Código principal
<br>    &nbsp;&nbsp;&nbsp;&nbsp;   ├── regex_patterns.py   # Padrões de regex utilizados nos filtros
<br>    &nbsp;&nbsp;&nbsp;&nbsp;   └── utils.py   # Funções auxiliares
<br>    ├── requirements.txt # Dependências do projeto
<br>    └── README.md # Este arquivo


## 📊 Fluxo de extração e tratamento dos dados

O processo de extração e tratamento dos dados desenvolvido na pesquisa pode ser resumido nas seguintes etapas:

### 1. Conversão de PDF para Markdown

- Inicialmente, os arquivos em PDF foram convertidos para o formato **Markdown**, um padrão textual leve e legível que preserva a hierarquia e a organização lógica do conteúdo, facilitando a análise posterior.  

- Utilizou-se a biblioteca **Docling**, ferramenta de código aberto reconhecida por sua robustez na preservação da estrutura textual e por contar com ampla comunidade de manutenção.

---

### 2. Limpeza e padronização dos dados

O texto convertido passou por um processo de sanitização, com:

- Remoção de espaços excessivos  
- Exclusão de fragmentos soltos sem conexão semântica  
- Eliminação de marcadores residuais de formatação típicos de PDFs  
- Remoção de caracteres de controle ASCII (como `[\x00-\x1f\x7f-\x9f]`), que frequentemente comprometem a leitura automatizada  

---

### 3. Divisão do texto em blocos interpretáveis

Diante da heterogeneidade dos documentos, não foi possível identificar parágrafos de forma consistente. Assim, cada PDF (considerado como unidade de análise) foi segmentado em:

- Blocos de **800 caracteres** (incluindo espaços)  
- Sobreposição (*overlap*) de aproximadamente **100 caracteres**

A sobreposição assegura continuidade contextual entre blocos adjacentes, evitando perda de sentido quando informações relevantes se situam na fronteira entre trechos.

Cada bloco foi programado para se encerrar apenas após:

- Um ponto final (`.`)  
- Um marcador de quebra de linha (`\n`)  

Esse critério preserva a integridade textual dos parágrafos.

---

### 4. Filtragem por expressões regulares

Foram utilizadas **expressões regulares (regex)** para identificar padrões textuais relevantes de forma concisa e flexível.

Com base nessa técnica:

- Elaborou-se um [dicionário conceitual](https://github.com/SGE-Subsecretaria-de-Gestao-Estrategica/projeto_acoes_afirmativas/blob/main/src/regex_patterns.py) a partir de palavras-chave associadas às informações de interesse  
- Definiram-se padrões de busca para localizar conteúdos específicos nos editais  

Esse procedimento permitiu filtrar blocos com precisão, mesmo diante de variações na redação e na estrutura dos documentos.

---

### 5. Aplicação de modelo de linguagem de grande porte (LLM)

Os blocos filtrados foram utilizados como *input* para o modelo de linguagem **GPT-4o-mini**, escolhido pelo equilíbrio entre desempenho e custo.

O modelo foi configurado com **temperatura 0**, reduzindo ao mínimo a variação nas respostas. Na prática, isso significa que, diante do mesmo estímulo, o modelo responde sempre da mesma forma, assegurando maior reprodutibilidade e precisão na extração dos dados.

---

### 6. Avaliação do fluxo de extração

A acurácia do processo foi avaliada por meio de amostragem em **70 editais** (20% da base total de 351).

Utilizou-se como métrica a **F1 Score**, definida pela combinação de:

- **Precisão (Precision):** proporção de acertos entre as previsões positivas  
- **Revocação (Recall):** proporção de acertos em relação às ocorrências reais  

O resultado aproximado de **97%** indica alto equilíbrio entre precisão e completude, confirmando a eficácia do modelo, com extração consistente e incidência mínima de erros.


## 💻 Como usar

### 1. Clonar o repositório

 ```
git clone https://github.com/seu-usuario/pnab-edital-extractor.git
cd pnab-edital-extractor
 ```

### 2. Criar ambiente virtual e instalar dependências
 
 ```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Rodar o pipeline de extração

```
python src/main.py data/exemplo_edital.pdf
```


## 🧮 Cálculo e agregação dos dados

A identificação do número de vagas e dos valores reservados por ente em cada categoria de cota seguiu critérios matemáticos padronizados, organizados em duas etapas complementares: cálculo por edital (nível individual) e agregação por ente federativo (nível consolidado). 

### 1. Cálculo de vagas 

#### 1.1 Cálculo do número de vagas reservadas por edital (nível individual)

Nesta etapa, calculou-se o número de vagas reservadas por categoria de cota em cada edital analisado, a partir dos seguintes procedimentos: 

**1.1.1** Para obter o número de vagas destinadas a cada categoria de cota (vagas_cota_edital), multiplicou-se o total de vagas do edital (vagas_total_edital) pelo percentual de reserva destinado a cada categoria (p_cota_edital).  

$$
\text{vagas-total-edital} \times \text{p-cota-edital} = \text{vagas-cota-edital}
$$

**1.1.2** Quando o resultado foi um número fracionário, aplicou-se a regra prevista no art. 6º, § 2º, da IN MinC nº 10/2023, adotando-se arredondamento para o número inteiro imediatamente superior em frações iguais ou superiores a 0,5 e para o inteiro imediatamente inferior em frações menores que 0,5. 

**1.1.3** Nos casos de editais específicos integralmente destinados a um dos três grupos prioritários (pessoas negras, pessoas indígenas e PCD), a totalidade das vagas foi contabilizada na respectiva categoria de cota. 

#### 1.2 Agregação das vagas reservadas por ente federativo (nível consolidado)  

Após o cálculo por edital, os dados foram consolidados por ente federativo, com o objetivo de estimar o percentual aproximado de vagas reservadas por categoria em cada ente. Para tanto, adotou-se o seguinte fluxo:  

**1.2.1.** Primeiro, apurou-se o total de vagas destinadas a cada categoria nos editais de cada ente (vagas_cota_ente), por meio do somatório (∑) das vagas reservadas em todos os editais publicados (∑ vagas_cota_edital).  

$$
\sum \text{vagas-cota-edital} = \text{vagas-cota-ente}
$$

**1.2.2.** Em seguida, obteve-se o total de vagas dos editais de cada ente (vagas_total_ente), mediante o somatório das vagas previstas em todos os editais publicados (∑ vagas_total_edital). 

\sum \text{vagas-total-ente} = \text{vagas-total-ente}

**1.2.3**. Por fim, o percentual de reserva de cada ente (p_agregado_vagas_ente) foi obtido pela razão entre o total de vagas reservadas e o total geral de vagas: 

$$
\frac{\text{vagas-cota-ente}}{\text{vagas-total-ente}} \times 100 = \text{p-agregado-vagas-cotas-ente}
$$

### 2. Cálculo dos valores 

#### 2.1 Cálculo de valores reservados por edital (nível individual) 

Além do número de vagas, calculou-se o valor destinado a cada categoria de cota por edital, a partir de um procedimento de estimativa, descrito nas etapas a seguir: 

**2.1.1** Inicialmente, estimou-se o valor unitário de cada vaga (valor_vaga_edital) por meio da divisão do valor total do edital (valor_total_edital) pelo total de vagas previstas (vagas_total_edital).  

$$
\frac{\text{valor-total-edital}}{\text{vagas-total-edital}} = \text{valor-vaga-edital}
$$

**2.1.2** Em seguida, calculou-se o valor destinado a cada categoria de cota no edital (valor_cota_edital), multiplicando o número de vagas reservadas (vagas_cota_edital) pelo valor unitário obtido no passo anterior. 

$$
\text{vagas-cota-edital} \times \text{valor-vaga-edital} = \text{valor-cota-edital}
$$

#### 2.2 Agregação dos valores reservadas por ente federativo (nível consolidado)

Após o cálculo individual por edital, os dados foram consolidados por ente federativo, a fim de estimar o percentual aproximado de valores reservados a cada categoria de cota em cada ente. Para isso, adotou-se o seguinte procedimento: 

**2.2.1.** Primeiro, apurou-se o valor total reservado por ente a cada categoria (valor_cota_ente), mediante o somatório dos valores correspondentes nos editais por ele publicados (∑valor_cota_edital). 

$$
\sum \text{valor-cota-edital} = \text{valor-cota-ente}
$$

**2.2.2.** Em seguida, obteve-se o total dos valores dos editais de cada ente (valor_total_ente), mediante o somatório dos valores previstaos em todos os editais publicados (∑ valores_total_ente). 

$$
\sum \text{valores-total-ente} = \text{valor-total-ente}
$$

**2.2.3** Por fim, o percentual do valor reservado para cada categoria de cota (p_ agregado_valor_cota_ente) foi calculado, em cada ente, pela razão entre o valor total destinado às cotas (valor_cota_ente) e o valor total destinado (valor_total_ente), multiplicada por 100. 

$$
\left( \frac{\text{valor-cota-ente}}{\text{valor-total-ente}} \right) \times 100 = \text{p-agregado-valor-cota-ente}
$$


## 🔎 Interpretação dos resultados

A análise considerou, de forma agregada, os percentuais de vagas e de recursos previstos nos editais de cada ente. A identificação de índices ligeiramente inferiores ao mínimo normativo não implica, por si só, descumprimento das regras de cotas. Essas variações podem decorrer de fatores técnicos, como: 

- **Uso de editais específicos**: a publicação de editais exclusivos para determinados grupos prioritários pode alterar a proporção de reservas quando se observa o conjunto dos editais. Nesses casos, a ação afirmativa se dá por meio de instrumentos focalizados, sem que haja descumprimento das diretrizes no resultado global.
- **Regras de arredondamento**: a conversão de frações em números inteiros pode produzir percentuais levemente inferiores ao parâmetro previsto. Trata-se, muitas vezes, de efeito da aplicação correta da regra de arredondamento estabelecida no art. 6º, § 2º, da IN MinC nº 10/2023, sem indicar necessariamente descumprimento da reserva.


## 📈 Análise e consolidação de dados 

A análise da implementação das cotas nos estados, capitais e DF baseou-se nos índices e cálculos de distribuição detalhados na seção anterior, necessários para mensurar a aplicação das cotas dentro de editais universais. Já o exame do uso de editais específicos não exigiu cálculos de partilha, visto que são destinados a um público exclusivo.  

Para este grupo, realizou-se uma triagem manual dos 496 editais levantados, fundamentada nos parâmetros dispostos no art. 14 da IN MinC nº 10/2023, para identificar: i. a configuração do certame (se específico para grupos e territórios vulnerabilizados); e ii. o público-alvo do instrumento. Dados como valores, número de vagas e ente responsável foram obtidos via extração automatizada, o que permitiu uma análise quantitativa padronizada com os critérios de reserva de vagas. 

Esse percurso metodológico consolidou um acervo disperso em uma base de dados estruturada, permitindo que o processamento quantitativo das duas modalidades de ação afirmativa seguisse quatro frentes principais: 

- **Adesão:** Implementação de cotas e editais específicos em estados, DF e capitais  

- **Alcance:** Quantidade de vagas destinadas a cada grupo e percentuais.  

- **Investimento:** Volume de recursos financeiros direcionados por grupo e percentuais.  

- **Comparação demográfica:** análise comparada dos percentuais de cotas com a composição demográfica dos estados e DF, apenas;

Essa estrutura permitiu analisar a dinâmica das ações afirmativas no Ciclo 1 da Política Nacional Aldir Blanc, revelando pistas sobre o grau de incorporação de cotas e editais específicos por estados, capitais e Distrito Federal. 


## 📜 Licença

Este projeto está sob a licença MIT.

# Guia Rápido do Projeto Kedro – Spaceflights

Este documento explica, passo a passo e em linguagem simples, como o projeto **Spaceflights** foi estruturado com o Kedro para repetir o fluxo de tratamento de dados, criação de features, balanceamento de classes, treinamento do modelo e geração do `submission.csv` do notebook `project002.ipynb`.

## 1. Visão geral da estrutura de pastas

```
spaceflights/
├── conf/                 # Configurações do Kedro (datasets, parâmetros, logs)
│   └── base/
│       ├── catalog.yml   # Onde declaramos os conjuntos de dados usados no pipeline
│       ├── parameters.yml            # Parâmetros globais (ex.: threshold)
│       ├── parameters_data_science.yml  # Parâmetros de ML, incluindo RandomForest
│       └── ...
├── data/                 # Dados brutos, processados e relatórios
├── src/spaceflights/
│   ├── pipelines/
│   │   ├── data_processing/   # Pipeline de ingestão e pré-processamento
│   │   │   ├── nodes.py       # Funções Python (nodes)
│   │   │   └── pipeline.py    # Orquestração dos nodes
│   │   └── data_science/      # Pipeline de modelagem
│   │       ├── nodes.py
│   │       └── pipeline.py
│   └── pipeline_registry.py   # Registro de todos os pipelines
└── README_KEDRO.md            # (este arquivo)
```

## 2. Pipelines criadas

### 2.1 `data_processing`

1. **Node `load_public_dataset`**
   * Lê `data/01_raw/public.csv` e mostra informações básicas.
2. **Node `preprocess_public_dataframe`**
   * Realiza todas as etapas do notebook:
     * Preenche `metaf` faltantes.
     * Converte `hora_ref` para datetime e extrai `hora` & `dia`.
     * Cria flag `precisa_troca`.
     * Aplica *one-hot encoding* em `origem` e `destino`.
     * Separa linhas com/sem a coluna‐alvo `espera` ⇒ entrega:
       * `X_train_rf`, `y_train_rf` (treino)
       * `X_to_predict_rf`, `flightid_pred` (para submissão)

Resultado: dados prontos para o pipeline de ciência de dados.

### 2.2 `data_science`

1. **Node `train_random_forest`**
   * Aplica *SMOTE* para balancear classes.
   * Treina `RandomForestClassifier` seguindo hiperparâmetros em `params:model_rf`.
2. **Node `predict_espera`**
   * Usa o modelo treinado para prever `espera` em `X_to_predict_rf`.
3. **Node `build_submission`**
   * Combina `flightid_pred` + `espera_pred` ⇒ `submission`.

O dataset `submission` é salvo automaticamente em `data/08_reporting/submission.csv`.

## 3. Configurações importantes (`conf/base`)

| Arquivo | Papel |
|---------|-------|
| `catalog.yml` | Declara datasets, por exemplo `public` (CSV) e `submission` (CSV de saída). |
| `parameters_data_science.yml` | Bloco `model_rf` com `n_estimators`, `test_size` e `random_state`. |
| `parameters.yml` | Apenas o parâmetro global `threshold` (sem duplicidades!). |

## 4. Registro de pipelines

No `src/spaceflights/pipeline_registry.py`:
```python
return {
    "data_processing": data_processing.create_pipeline(),
    "data_science"   : data_science.create_pipeline(),
    "__default__"    : data_processing.create_pipeline() + data_science.create_pipeline(),
}
```
* O comando `kedro run` executa `__default__`, portanto roda os dois pipelines em sequência.
* `kedro viz` exibe o fluxo completo.

## 5. Como executar

1. **Instalar dependências** (ex.: num ambiente virtual):
   ```bash
   pip install -r requirements.txt
   ```
2. **Executar pipeline completo:**
   ```bash
   kedro run
   ```
3. **Visualizar pipeline:**
   ```bash
   kedro viz
   ```
4. **Arquivo de submissão gerado:**
   * `data/08_reporting/submission.csv`

## 6. Principais funções (nodes)

| Node | Local | O que faz |
|------|-------|-----------|
| `load_public_dataset` | `pipelines/data_processing/nodes.py` | Carrega CSV. |
| `_prepare_feature_matrix` | idem | Helper privado para engenharia de features. |
| `preprocess_public_dataframe` | idem | Produz X/Y para treino e previsão. |
| `train_random_forest` | `pipelines/data_science/nodes.py` | Treina modelo com SMOTE. |
| `predict_espera` | idem | Gera previsões. |
| `build_submission` | idem | Cria DataFrame final de submissão. |

## 7. Próximos passos

* Ajustar hiperparâmetros em `conf/base/parameters_data_science.yml` para melhorar desempenho.
* Adicionar testes unitários para os novos nodes, seguindo o padrão em `tests/`.
* Habilitar versionamento para mais datasets se desejar auditar históricos.

---
Com este guia qualquer pessoa, mesmo sem experiência prévia com Kedro, pode entender a lógica do projeto, localizar os arquivos relevantes e reproduzir o fluxo completo. Bom trabalho! 🚀 
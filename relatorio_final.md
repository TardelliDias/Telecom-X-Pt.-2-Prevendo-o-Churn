# <h1 align='center'>Análise e Previsão de Evasão de Clientes (Churn)</h1>

## 1. Introdução

Este relatório apresenta uma análise aprofundada sobre a evasão de clientes na **Telecom X**, com o objetivo de identificar os principais fatores que influenciam o churn e desenvolver modelos preditivos eficazes. A compreensão desses fatores é crucial para a formulação de estratégias de retenção de clientes.

## 2. Metodologia

### Pré-processamento e Engenharia de Features:

1.  **Limpeza de Dados**: Colunas irrelevantes como `id_cliente` foram removidas. Análises de colunas constantes, quase constantes e de alta cardinalidade foram realizadas, levando à remoção da coluna `cobrança_total` e de algumas features altamente correlacionadas para mitigar a multicolinearidade.

2.  **Codificação de Variáveis**: Variáveis categóricas foram transformadas utilizando `OneHotEncoder` (com `drop='if_binary'`), e a variável alvo `churn` foi codificada para numérica utilizando `LabelEncoder` (0 = Não Evadiu, 1 = Evadiu).

3.  **Balanceamento de Classes**: Detectado um desequilíbrio significativo (73.42% 'Ativo', 26.58% 'Churn'), foram aplicadas duas técnicas de balanceamento:

    *   **Oversampling (SMOTE)**: Aumentou o número de amostras da classe minoritária, resultando em 5163 amostras para ambas as classes (total de 10326).
    *   **Undersampling (RandomUnderSampler)**: Reduziu o número de amostras da classe majoritária, resultando em 1869 amostras para ambas as classes (total de 3738).  

4.  **Padronização de Features**: As features numéricas dos conjuntos de dados balanceados (`SMOTE` e `RandomUnderSampler`) foram padronizadas usando `StandardScaler` para otimizar o desempenho dos modelos baseados em distância.

### Análise e Comparação dos Resultados de Balanceamento:

**1. Revisão das Distribuições de Classes:**

*   **Distribuição Original (`y`):**
    *   Classe 0 (Ativo): 5163 amostras (73.42%)
    *   Classe 1 (Churn): 1869 amostras (26.58%)
    *   Total de amostras: 7032

*   **Distribuição Após SMOTE (`y_smote`):**
    *   Classe 0 (Ativo): 5163 amostras
    *   Classe 1 (Churn): 5163 amostras
    *   Total de amostras: 10326

*   **Distribuição Após RandomUnderSampler (`y_rus`):**
    *   Classe 0 (Ativo): 1869 amostras
    *   Classe 1 (Churn): 1869 amostras
    *   Total de amostras: 3738

**2. Principais Diferenças Observadas:**

*   **Número Total de Amostras:**
    *   **SMOTE** aumentou o número total de amostras de 7032 para 10326. Ele conseguiu isso gerando amostras sintéticas para a classe minoritária ('churn').

    *   **RandomUnderSampler** reduziu o número total de amostras de 7032 para 3738. Ele balanceou as classes removendo amostras da classe majoritária ('ativo').

*   **Método de Balanceamento:**
    *   **SMOTE** (Oversampling) cria novas amostras sintéticas da classe minoritária. Isso ajuda a combater o desequilíbrio sem perder informações da classe majoritária, mas pode introduzir ruído ou amostras irrealistas.

    *   **RandomUnderSampler** (Undersampling) remove aleatoriamente amostras da classe majoritária. É um método simples e eficaz, mas a principal desvantagem é a potencial perda de informações valiosas contidas nas amostras descartadas da classe majoritária.

**3. Implicações para o Treinamento de um Modelo Preditivo:**

*   **Volume de Dados para Treinamento:**
    *   **SMOTE:** Resulta em um conjunto de dados de treinamento maior. Isso geralmente é benéfico, pois mais dados podem levar a modelos mais robustos e capazes de capturar melhor os padrões subjacentes, especialmente em problemas complexos.

    *   **RandomUnderSampler:** Resulta em um conjunto de dados de treinamento menor. Embora simplifique o treinamento e possa acelerar o processo, a redução drástica do volume de dados pode levar a um modelo menos informativo e com menor poder de generalização, pois menos exemplos da classe majoritária foram utilizados.

*   **Risco de Overfitting:**
    *   **SMOTE:** Embora crie amostras sintéticas, o SMOTE pode, em alguns casos, levar a um certo grau de overfitting se as amostras sintéticas forem muito semelhantes entre si ou se não representarem bem a diversidade da classe minoritária real. O modelo pode aprender demais sobre esses exemplos sintéticos.

    *   **RandomUnderSampler:** O risco de overfitting à classe minoritária é geralmente menor, mas há um risco maior de o modelo não aprender a fundo as características da classe majoritária devido à perda de dados, o que pode levar a um underfitting ou a um viés no modelo.

*   **Perda de Informação:**
    *   **SMOTE:** A principal preocupação é a criação de amostras sintéticas que podem não ser totalmente representativas do espaço de características real, ou que podem introduzir ruído se a classe minoritária for intrinsecamente ruidosa.

    *   **RandomUnderSampler:** A perda de informação é uma desvantagem intrínseca, pois amostras reais da classe majoritária são simplesmente descartadas. Isso pode ser problemático se as amostras removidas contiverem padrões ou características importantes que distinguem a classe majoritária de forma sutil.

**Conclusão:**

A escolha entre `SMOTE` e `RandomUnderSampler` depende do contexto específico do problema e dos recursos disponíveis. 

* **SMOTE** é geralmente preferível quando se deseja manter o máximo de informações da classe majoritária e o volume de dados é uma preocupação, pois aumenta o tamanho do conjunto de treinamento. No entanto, deve-se ter cautela para evitar a criação de amostras sintéticas excessivamente homogêneas. 

*   **RandomUnderSampler** é mais simples e rápido, mas deve ser usado com cuidado para garantir que a remoção de amostras da classe majoritária não leve à perda de informações cruciais para o aprendizado do modelo.

### Modelagem Preditiva:

Foram explorados dois modelos de Machine Learning:

*   **Regressão Logística**: Um modelo linear, utilizado como baseline.
*   **Random Forest**: Um modelo de ensemble baseado em árvores, mais complexo e robusto.

Ambos os modelos foram treinados e avaliados em conjuntos de dados separados (treino e teste, com 70/30 de divisão, estratificada), utilizando métricas como `Acurácia`, `Precisão`, `Recall` e `F1-Score`, que são mais adequadas para dados desbalanceados.

## 3. Fatores que Mais Influenciam a Evasão

Com base na análise exploratória (boxplots) e na importância das features pelos modelos, os principais fatores que influenciam a evasão são:

### 3.1. Análise Visual (Boxplots):

*   **Tempo de Contrato**:
    *   Clientes que **evadem** (`churn=1`) tendem a ter um **tempo de contrato significativamente menor** (mediana em torno de 10 meses). 
    *   Clientes que **não evadem** (`churn=0`) possuem contratos de **maior duração** (mediana proximo de 40 meses).
    *   **Insight**: Clientes mais novos são mais propensos a sair.

![tempo_de_contrato_vs_churn.png](https://github.com/TardelliDias/Telecom-X-Pt.-2-Prevendo-o-Churn/blob/main/img/tempo_de_contrato_vs_churn.png)

*   **Cobrança Mensal**:
    *   Clientes que **evadem** (`churn=1`) tendem a ter **cobranças mensais mais elevadas** (mediana visivelmente maior).
    *   Clientes que **não evadem** (`churn=0`) têm uma distribuição de cobrança mensal mais ampla, mas com uma mediana menor.
    *   **Insight**: Clientes com planos mais caros são mais propensos a evadir.

![cobranca_mensal_vs_churn.png](https://github.com/TardelliDias/Telecom-X-Pt.-2-Prevendo-o-Churn/blob/main/img/cobranca_mensal_vs_churn.png)

### 3.2. Importância das Features pelos Modelos:

*   **Regressão Logística (Coeficientes)**:
    Os coeficientes da Regressão Logística apontam para a direção e força da relação. Geralmente, as features mais relevantes aqui incluem:

    *   **`tempo_contrato`**: Coeficiente negativo, indicando que quanto maior o tempo de contrato, menor a chance de churn.

    *   **`cobrança_mensal`**: Coeficiente positivo, indicando que maior cobrança mensal, maior a chance de churn.

    *   **Tipo de Contrato (`onehotencoder__contrato_mensal`, `onehotencoder__contrato_dois anos`)**: Contratos mensais costumam ter coeficientes positivos (maior churn), enquanto contratos de longo prazo (dois anos) costumam ter coeficientes negativos (menor churn).

    *   **Serviço de Internet (`onehotencoder__servico_internet_fibra ótica`)**: Fibra ótica pode ter um coeficiente positivo, sugerindo que clientes com esse serviço são mais propensos ao churn.

    *   **Método de Pagamento 
    (`onehotencoder__metodo_pagamento_cheque eletrônico`)**: Pode ter um coeficiente positivo, associado a maior churn.

*   **Random Forest (Importância das Variáveis)**:

    O Random Forest, um modelo não linear, geralmente identifica relações mais complexas. As features mais importantes costumam ser:

    *   **`tempo_contrato`**: Consistentemente uma das features mais importantes, refletindo a duração do relacionamento do cliente.

    *   **`cobrança_mensal`**: Também muito relevante, indicando o impacto do custo do serviço.

    *   **`onehotencoder__contrato_mensal` / `onehotencoder__contrato_dois anos`**: O tipo de contrato é um fator crucial, com contratos mensais impulsionando o churn e contratos anuais reduzindo-o.

    *   **`onehotencoder__servico_internet_fibra ótica`**: A presença de serviço de fibra ótica frequentemente aparece como um fator de churn, talvez devido a altas expectativas ou concorrência.

    *   **`onehotencoder__metodo_pagamento_cheque eletrônico`**: Clientes que pagam via cheque eletrônico são frequentemente associados a maior churn.

**Conclusão sobre Fatores:** As variáveis `tempo_contrato`, `cobrança_mensal`, `tipo de contrato`, `serviço de internet (especialmente fibra ótica)` e `método de pagamento (cheque eletrônico)` são os **principais fatores da evasão de clientes**.

## 4. Desempenho dos Modelos

A tabela abaixo resume o desempenho dos modelos testados no conjunto de teste, utilizando o F1-Score como métrica principal devido ao desequilíbrio de classes:

| Modelo                      | Estratégia de Balanceamento | Acurácia | Precisão | Recall | F1-Score |
| :-------------------------- | :-------------------------- | :------- | :------- | :----- | :------- |
| Regressão Logística         | SMOTE                       | 0.7631   | 0.7439   | 0.8025 | 0.7720   |
| Regressão Logística         | RandomUnderSampler          | 0.7594   | 0.7358   | 0.8093 | 0.7708   |
| Random Forest (Padrão)      | SMOTE                       | 0.8525   | 0.8645   | 0.8360 | 0.8500   |
| Random Forest (Otimizado)   | SMOTE                       | 0.8483   | 0.8483   | 0.8483 | 0.8483   |
| Random Forest (Otimizado)   | RandomUnderSampler          | 0.7656   | 0.7388   | 0.8217 | 0.7781   |

**Melhor Modelo**: O **Random Forest com dados balanceados por SMOTE** (seja com parâmetros padrão ou otimizados) demonstrou o melhor desempenho, com F1-Scores consistentemente na faixa de **0.84 a 0.85**. Isso representa uma melhoria significativa em relação à Regressão Logística.

**Impacto do Balanceamento**: O oversampling (SMOTE) combinado com o Random Forest foi a estratégia mais eficaz para este problema, resultando em um modelo com alta capacidade de generalização. O undersampling, embora mais rápido, resultou em perda de informação e menor desempenho para o Random Forest.

## 5. Estratégias de Retenção Baseadas nos Resultados

Com base nos fatores identificados e no desempenho dos modelos, as seguintes estratégias de retenção são propostas:

1.  **Foco em Clientes Novos (Tempo de Contrato Curto)**:
    *   **Estratégia**: Implementar programas de 'onboarding' robustos e proativos nos primeiros 6-12 meses de contrato. Oferecer suporte dedicado, checar a satisfação com o serviço e resolver proativamente quaisquer problemas.

    *   **Ação**: Contato personalizado (e-mail, telefone) nos primeiros meses, ofertas de benefícios exclusivos após 3-6 meses de lealdade (ex: upgrade de plano, bônus de dados).

2.  **Gerenciamento da Cobrança Mensal Elevada**:
    *   **Estratégia**: Clientes com mensalidades altas são mais sensíveis ao churn. Monitorar a satisfação e o valor percebido por esses clientes. Oferecer análises de uso para justificar o valor ou propor planos mais adequados.

    *   **Ação**: Revisão periódica de planos, ofertas de pacotes personalizados que ofereçam mais valor (ex: serviços adicionais gratuitos), comunicação transparente sobre o valor agregado.

3.  **Incentivo a Contratos de Longo Prazo**:
    *   **Estratégia**: Reduzir a atratividade dos contratos mensais e incentivar a migração para contratos de 1 ou 2 anos, que mostram menor propensão ao churn.

    *   **Ação**: Oferecer descontos significativos, benefícios extras ou 'freebies' para clientes que optarem por contratos anuais/bienais no momento da contratação ou renovação.

4.  **Atenção aos Usuários de Fibra Ótica e Cheque Eletrônico**:
    *   **Estratégia**: Clientes com fibra ótica podem ter expectativas mais altas ou ser mais exigentes. Clientes com método de pagamento via cheque eletrônico podem estar menos 'engajados' ou mais propensos a descontinuar o serviço. Criar campanhas direcionadas a esses segmentos.

    *   **Ação**: Para fibra ótica, garantir qualidade de serviço impecável e comunicar inovações. Para cheque eletrônico, investigar razões para a escolha do método e oferecer incentivos para métodos de pagamento mais 'sticky' (débito automático, cartão de crédito).

## 6. Conclusão

A análise revelou que a evasão de clientes é um problema complexo influenciado por fatores como `tempo de contrato`, `cobrança mensal`, `tipo de contrato`, `serviço de internet` e `método de pagamento`. O modelo **Random Forest treinado com oversampling (SMOTE)** se destacou como o mais eficaz na previsão de churn, fornecendo uma ferramenta poderosa para identificar clientes em risco.

As estratégias de retenção propostas são direcionadas aos principais fatores que influenciam o churn identificados na análise, com foco na proatividade, personalização e incentivos para construir relacionamentos mais duradouros e satisfatórios com os clientes. A implementação contínua e o monitoramento dessas estratégias, juntamente com a reavaliação periódica do modelo, serão fundamentais para reduzir a taxa de evasão.





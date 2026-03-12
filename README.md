# <h1 align='center'>Telecom X - Prevendo o Churn</h1>

## 📌 Descrição

Este projeto foi desenvolvido para criar modelos preditivos capazes de prever quais clientes terão propensão a cancelar os serviços da **Telecom X**, baseados em variáveis relevantes.

## 🏆 Objetivos

* Preparar os dados para a modelagem (tratamento, encoding, normalização).

* Realizar análise de correlação e seleção de variáveis.

* Treinar dois ou mais modelos de classificação.

* Avaliar o desempenho dos modelos com métricas.

* Interpretar os resultados, incluindo a importância das variáveis.

* Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.

## ✅ Status do Projeto

✔️ Projeto concluído! 

## 💻 Tecnologias Utilizadas

* `Python` - Linguagem base para o projeto

* `Pandas` - Biblioteca para análise e exploração dos dados

* `Seaborn` e `Matplotlib` - Bibliotecas para visualização dos dados

* `Scikit-learn` - Biblioteca para criar os modelos preditivos

* `Markdown` - Apresentação e documentação

* `Jupyter Notebook` - Ambiente de desenvolvimento e execução 

## 📊 Resultados

Com base na análise exploratória (boxplots) e na importância das features pelos modelos, os principais fatores que influenciam a evasão são:

  * **Tempo de Contrato**: Clientes mais novos são mais propensos a sair.
  *  **Cobrança Mensal**: Clientes com planos mais caros são mais propensos a evadir.
  *  **Método de pagamento (`Cheque eletrônico`)**: Clientes que pagam via cheque eletrônico são frequentemente associados a maior churn.
  *  **Serviço de Internet (`Fibra ótica`)**: A presença de serviço de fibra ótica frequentemente aparece como um fator de churn, talvez devido a altas expectativas ou concorrência.
  

## 📜 Conclusão

A análise revelou que a evasão de clientes é um problema complexo influenciado por fatores como `tempo de contrato`, `cobrança mensal`, `tipo de contrato`, `serviço de internet` e `método de pagamento`. O modelo **Random Forest treinado com oversampling (SMOTE)** se destacou como o mais eficaz na previsão de churn, com uma acurácia de 85%, se apresentando como uma ferramenta poderosa para identificar clientes em risco de evasão.

* [Acessar o Relatório Final](https://github.com/TardelliDias/Telecom-X-Pt.-2-Prevendo-o-Churn/blob/main/relatorio_final.md)

## 😁 Contato

**Nome:** Tardelli Dias <br>
**E-mail:** tardelli.dias@gmail.com <br>
**LinkedIn:** https://www.linkedin.com/in/tardelli-dias/ <br>
**Discord:** https://discord.com/users/tardellif
- - -
📚 ***Projeto com fins educacionais.***





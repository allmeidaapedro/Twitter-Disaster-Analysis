# Análise de Tweets - Desastre ou não?

<img src="logo_twitter.webp" width="100%" height=350px>

- Este é um projeto de <b>classificação binária</b>, utilizando <b>NLP</b>, para analisar um conjunto de <b>tweets</b> e <B>determinar quais estão relacionados a desastres reais e quais não estão.</b>
- Este projeto é a resolução de um case para uma vaga como cientista de dados.
- Para entender todo o processo, incluindo o problema de negócio, análise exploratória de dados, modelagem e deploy, acesse a <b>documentação do projeto acima no arquivo 'analise_tweets_desastres.pdf' ou no arquivo 'analise_tweets_desastres.pptx'.</b>

# Estrutura de pastas do projeto
- <B>Artifacts:</b> Contém os artefatos do modelo de machine learning (model .pkl, preprocessor .pkl, dados brutos, de treino e teste), após a execução do pipeline de treinamento.
- <B>Input:</b> Contém os dados brutos, utilizados em todos os notebooks e arquivos do projeto.
- <B>Notebooks:</b> Contém os notebooks de análise exploratória de dados e modelagem preditiva.
- <B>Src:</b> Contém todos os scripts .py, incluindo utils, exceções, logger, componentes de ingestão de dados, transformação de dados e treinamento de modelo, e pipelines de treino e predição para deploy do modelo.
- <B>Templates:</b> Contém as páginas de predição da probabilidade de um tweet estar relacionado a um desastre.
- <B>Arquivos importantes:</b> 
    - <B>setup.py:</b> Contém metadados acerca do projeto, além de dependências.
    - <B>requirements.txt:</b> Contém as bibliotecas utilizadas no projeto e suas versões, a fim de prover reprodutibilidade em qualquer dispositivo.
    - <B>app.py:</b> API em Flask, responsável pela requisição de um tweet de formulário e utilização deste para predição.
    - <B>analise_tweets_desastre.py e .pptx:</b> Documentação do projeto contendo todo o pipeline detalhado, desde a definição do problema de negócio ao deploy.

# Execute o projeto na sua máquina local
- Pré-requisitos:

- Antes de começar, certifique-se de ter os seguintes itens instalados em sua máquina:

    - Python 3.11.4
    - pip (gerenciador de pacotes Python)
    - Git (ferramenta de controle de versão)

- Após instalar esses requisitos, abra um terminal em sua máquina local e execute os seguintes comandos:

1. Clonar o repositório:
<pre>
git clone https://github.com/allmeidaapedro/Twitter-Disaster-Analysis.git
</pre>

2. Navegar até o diretório do repositório clonado:
<pre>
cd Twitter-Disaster-Analysis
</pre>

3. Criar um ambiente virtual:
<pre>
python -m venv venv
</pre>

4. Ativar o Ambiente Virtual:
- Ative o ambiente virtual usado para isolar as dependências do projeto.
<pre>
source venv/bin/activate  # No Windows, use 'venv\Scripts\activate'
</pre>

5. Instalar Dependências:
- Use o pip para instalar as dependências necessárias listadas no arquivo requirements.txt.
<pre>
pip install -r requirements.txt
</pre>

6. Executar a Aplicação:
- Para prever a probabilidade de um tweet estar relacionado a um desastre, execute:
<pre>
python app.py
</pre>

7. Acessar o Projeto Localmente:
- Após executar com sucesso, você pode acessar o projeto localmente. Abra um navegador da web e vá para http://127.0.0.1:5000/
- Em seguida, vá para a página de predição, cole o tweet e clique em enviar. A probabilidade de o tweet estar relacionado a um desastre aparecerá no lado direito.

8. Desligar a Aplicação:
- Para parar a aplicação, normalmente você pode pressionar Ctrl+C no terminal onde a aplicação está em execução.

9. Desativar o Ambiente Virtual:
- Quando terminar com o projeto, desative o ambiente virtual.
<pre>
deactivate
</pre>

# Contato
- <b>Linkedin:</b> https://www.linkedin.com/in/pedro-henrique-almeida-oliveira-77b44b237/
- <b>Github:</b> https://github.com/allmeidaapedro
- <b>Gmail:</b> pedrooalmeida.net@gmail.com
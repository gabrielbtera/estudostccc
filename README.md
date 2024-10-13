# Como criar uma aplicação de Recuperação Aumentada por Geração (RAG) on-premise utilizando Llama3-8b, MongoDB e uma GPU Nvidia?

Para prosseguir com o tutorial de desenvolvimento de uma aplicação RAG on-premise utilizando Llama3-8b, MongoDB e uma GPU Nvidia, é fundamental garantir que os pré-requisitos estejam atendidos:

1. **Sistema Operacional Baseado em Linux**: O desenvolvimento será realizado em um sistema Linux, que proporciona um suporte otimizado para bibliotecas de machine learning e facilita a configuração de drivers e frameworks essenciais, como o CUDA, necessários para utilizar a GPU Nvidia com eficiência.

2. **GPU com no mínimo 4GB de VRAM**: A presença de uma GPU com pelo menos 4GB de VRAM é crucial para o processamento eficiente dos modelos de linguagem, garantindo a capacidade de realizar inferências no Llama3-8b.

3. **Processador Core i5 ou Superior**: Um processador Intel Core i5 ou equivalente é recomendado para lidar com a carga de trabalho, especialmente no pré-processamento dos dados e em operações que não utilizam a GPU. Processadores mais potentes podem melhorar ainda mais o desempenho.

4. **Pelo Menos 16GB de RAM**: Para processar grandes volumes de dados e executar tarefas intensivas, é necessário ter pelo menos 16GB de RAM, o que ajudará a manter o sistema responsivo e eficiente durante as operações.

Com esses requisitos atendidos, você estará preparado para configurar o ambiente e seguir com a execucão da aplicação RAG on-premise, garantindo um desempenho robusto e uma experiência de desenvolvimento otimizada.

## Etapa 1: Configurando a GPU

Vamos começar configurando os drivers CUDA na máquina. Essa etapa é essencial para garantir que a GPU possa ser utilizada de forma eficiente durante o processamento dos modelos de linguagem.

Para configurar o ambiente com suporte à GPU da NVIDIA em sua distribuição Linux, siga os passos abaixo. É recomendado utilizar o gerenciador de pacotes do sistema para instalar o driver, conforme a recomendação oficial da NVIDIA.

### 1. Instale o Driver NVIDIA

A forma mais recomendada de instalar o driver é através do gerenciador de pacotes da sua distribuição. Para mais informações sobre a instalação utilizando um gerenciador de pacotes, consulte o [Guia Rápido de Instalação do Driver da NVIDIA](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

> **Alternativa:** Você também pode instalar o driver baixando um instalador `.run`. Consulte a [página oficial de drivers da NVIDIA](https://www.nvidia.com/Download/index.aspx).

### 2. Configuração do Repositório com Apt

Caso utilize uma distribuição baseada em Debian/Ubuntu, siga os passos abaixo para configurar o repositório de pacotes de produção da NVIDIA:

- Adicione a chave GPG e configure o repositório:

  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  ```

- **Opcional:** Ative o uso de pacotes experimentais, se desejar:

  ```bash
  sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
  ```

### 3. Atualize a Lista de Pacotes

Após configurar o repositório, atualize a lista de pacotes disponíveis:

```bash
sudo apt-get update
```

### 4. Instale o NVIDIA Container Toolkit

Por fim, instale o toolkit da NVIDIA para containers com o comando:

```bash
sudo apt-get install -y nvidia-container-toolkit
```

---

Pronto! Agora o driver da NVIDIA está instalado e o sistema está preparado para utilizar a GPU em aplicações que requerem aceleração por hardware. Para validar a instalação, você pode rodar o comando `nvidia-smi` e verificar se a GPU é reconhecida corretamente.

### Dica Extra

Para garantir que o driver está sempre atualizado, configure o sistema para receber atualizações automáticas dos pacotes da NVIDIA diretamente do repositório configurado. Isso manterá o ambiente sempre otimizado para o uso com GPUs.

## Etapa 2: Executando o Ollama na GPU

Vamos configurar e rodar o Ollama em um contêiner Docker com suporte a GPU. Siga os passos abaixo para deixar tudo funcionando.

### Pré-requisitos:

1. Verifique se o Docker está instalado e configurado corretamente no seu sistema.
2. Certifique-se de que o driver NVIDIA e o NVIDIA Container Toolkit estão instalados para usar a GPU.

### Passo 1: Inicie o Contêiner com Suporte a GPU

Abra o terminal e execute o comando abaixo para rodar o contêiner Docker com a imagem `ollama/ollama`:

```bash
docker run -it --gpus=all -v /home/ollama:/root/.ollama:z -p 11435:11434 --name ollama ollama/ollama
```

- O contêiner será iniciado de forma interativa, com suporte a todas as GPUs disponíveis (`--gpus=all`).
- O volume `/home/ollama` será montado em `/root/.ollama` no contêiner para manter os dados persistentes.
- A porta 11434 do contêiner será mapeada para a porta 11435 do host.
- O contêiner será nomeado como "ollama" para facilitar o uso.

> **Observação:** Ao iniciar, o contêiner exibirá alguns logs no terminal, o que é esperado.

### Passo 2: Acesse o Terminal do Contêiner

Abra um novo terminal e rode o seguinte comando para entrar no terminal do contêiner:

```bash
docker exec -it ollama /bin/bash
```

Isso permitirá que você interaja diretamente com o ambiente dentro do contêiner.

### Passo 3: Baixe o Modelo Llama3

No terminal paralelo, execute o comando abaixo para baixar o modelo Llama3:

```bash
docker exec -it ollama ollama pull llama3
```

Isso iniciará o download do modelo, deixando-o disponível para uso dentro do contêiner.

No seu terminal aparecerá algo assim:

[imagem do container llamma 3 em execucao]

---

Seguindo esses passos, o Ollama estará configurado para rodar na GPU, com o modelo Llama3 pronto para ser utilizado.

## Etapa 3: Configurando e Executando o MongoDB com Docker Compose

Nesta etapa, vamos configurar o MongoDB utilizando o Docker Compose. Primeiro, siga as instruções abaixo para clonar o repositório e configurar o ambiente.

### Passo 1: Clone o Repositório

Clone o repositório necessário com o comando abaixo:

```bash
git clone https://github.com/gabrielbtera/estudostccc
```

### Passo 2: Navegue até a Pasta `database`

Entre na pasta `database` dentro do repositório clonado:

```bash
cd estudostccc/database
```

### Passo 3: Instale as Dependências do Projeto

Se houver um arquivo `requirements.txt` na pasta, execute o comando abaixo para instalar as dependências necessárias:

```bash
pip install -r requirements.txt
```

Isso garantirá que todas as bibliotecas necessárias para o ambiente estejam instaladas.

### Passo 4: Verifique o Arquivo `docker-compose.yml`

Dentro da pasta `database`, certifique-se de que o arquivo `docker-compose.yml` está configurado com o seguinte conteúdo:

```yaml
services:
  mongodb:
    image: mongodb/mongodb-atlas-local
    environment:
      - MONGODB_INITDB_ROOT_USERNAME=user
      - MONGODB_INITDB_ROOT_PASSWORD=pass
    ports:
      - 27019:27017
    volumes:
      - data:/data/db

volumes:
  data:
```

- **Explicação das Configurações:**
  - `image`: Especifica a imagem do MongoDB a ser usada.
  - `environment`: Configura as variáveis de ambiente para definir o usuário e a senha do banco de dados.
  - `ports`: Mapeia a porta 27017 do contêiner para a porta 27019 do host.
  - `volumes`: Monta o volume `data` para persistência dos dados do MongoDB.

### Passo 5: Execute o MongoDB com Docker Compose

Agora, execute o comando para iniciar o MongoDB:

```bash
docker-compose up -d
```

- O parâmetro `-d` faz com que os serviços sejam executados em segundo plano.

> **Nota:** Para verificar se o MongoDB está rodando corretamente, use o comando `docker-compose ps` e certifique-se de que o serviço `mongodb` está "UP".

### Passo 6: Acessar o MongoDB

Com o MongoDB em execução, você pode se conectar ao banco de dados utilizando o cliente de sua preferência (como MongoDB Compass ou Mongo Shell) com a URL de conexão:

```
mongodb://user:pass@localhost:27019
```

Essa URL usa o usuário e a senha configurados no arquivo `docker-compose.yml` e mapeia a porta 27019 para o host.

---

Seguindo esses passos, o MongoDB estará configurado e rodando em um contêiner Docker, pronto para ser utilizado na aplicação RAG com o Llama3-8b.

## Etapa 4: Alimentando o Banco de Dados

Agora vamos adicionar documentos PDF ao banco de dados MongoDB. Siga os passos abaixo para configurar e executar o processo de inserção.

### Passo 1: Adicione Documentos PDF

Navegue até a pasta `database` e, dentro dela, entre na subpasta `pdf`. Coloque pelo menos um documento PDF nesta pasta:

```bash
cd estudostccc/database/pdf
# ATENCAO: Adicione um ou mais arquivos .pdf na pasta
```

### Passo 2: Crie um Ambiente Virtual

Volte para a pasta `database` e crie um ambiente virtual para o Python:

```bash
cd ..
python3 -m venv venv
```

### Passo 3: Ative o Ambiente Virtual

Ative o ambiente virtual criado:

- No Linux ou macOS:

  ```bash
  source venv/bin/activate
  ```

### Passo 4: Instale as Dependências

Com o ambiente virtual ativo, instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Passo 5: Execute o Script para Inserir os Documentos no Banco de Dados

Agora, execute o script `main.py` para inserir os documentos PDF no MongoDB:

```bash
python3 main.py
```

Esse comando irá processar os documentos PDF na pasta `pdf` e inseri-los no banco de dados.

---

Seguindo esses passos, o banco de dados MongoDB será alimentado com os documentos PDF fornecidos, prontos para serem utilizados na aplicação RAG.

### Comentários: Como a inserção funciona?

A função `load_docs_from_directory` tem a responsabilidade de carregar documentos PDF de um diretório especificado (`PDF_PATH`), convertê-los em formato Markdown e dividi-los em chunks para serem posteriormente inseridos em um banco de dados.

1. **Listar os arquivos PDF no diretório**:

   - A função começa criando uma lista (`doc_list`) que contém o caminho completo de todos os arquivos PDF presentes no diretório especificado (`PDF_PATH`). Isso é feito verificando se o caminho corresponde a um arquivo.

   ```python
   doc_list = [join(PDF_PATH, f) for f in os.listdir(PDF_PATH) if isfile(join(PDF_PATH, f))]
   ```

2. **Processar cada arquivo PDF**:

   - Para cada arquivo PDF na lista, o conteúdo é convertido para Markdown usando a biblioteca `pymupdf4llm`, que transforma o PDF em uma lista de chunks de texto e metadados (por exemplo, páginas). Essa lista de chunks é armazenada em `all_docs_list`.

   ```python
   for file_path in doc_list:
       all_docs_list += pymupdf4llm.to_markdown(file_path, page_chunks=True)
   ```

3. **Dividir o texto em chunks menores (text splitting)**:

   - Utiliza o `MarkdownTextSplitter` para dividir o texto e seus metadados em documentos menores. Isso é feito para tornar os dados mais fáceis de indexar e buscar.

   ```python
   text_splitter = MarkdownTextSplitter()
   return text_splitter.create_documents([i['text'] for i in all_docs_list], [i['metadata'] for i in all_docs_list])
   ```

   Nesse ponto, a função retorna uma lista de documentos, onde cada documento contém um trecho do texto original e os metadados associados.

#### Inserção no Banco de Dados

A inserção dos documentos no banco de dados ocorre na função `setup_atlas`. Veja como isso é feito:

1. **Configurar a conexão com o MongoDB**:

   - A função conecta-se ao MongoDB usando a string de conexão fornecida (`ATLAS_CONNECTION_URL`), acessando o banco de dados (`ATLAS_DB_NAME`) e a coleção (`ATLAS_COLLECTION_NAME`).

   ```python
   client = MongoClient(ATLAS_CONNECTION_URL)
   atlas_collection = client[ATLAS_DB_NAME][ATLAS_COLLECTION_NAME]
   ```

2. **Verificar se o banco de dados já existe**:

   - A função verifica se o banco de dados chamado "search_db" já existe. Se sim, a função retorna `True` e não realiza a inserção novamente.

3. **Carregar e inserir os documentos**:

   - Caso o banco de dados não exista, os documentos são carregados usando a função `load_docs_from_directory` e inseridos no banco de dados MongoDB com vetores de embeddings associados.

   ```python
   docs = load_docs_from_directory()
   MongoDBAtlasVectorSearch.from_documents(
       documents=docs,
       embedding=embeddings,
       collection=atlas_collection,
       index_name=ATLAS_SEARCH_INDEX_NAME
   )
   ```

   Aqui, a função `MongoDBAtlasVectorSearch.from_documents` é utilizada para indexar os documentos no MongoDB Atlas com embeddings, criando uma pesquisa vetorial que permitirá buscar documentos por similaridade.

Essa abordagem facilita a recuperação de documentos relevantes com base na similaridade de embeddings, possibilitando consultas eficientes no banco de dados.

## Etapa 5: Executando o Backend Assíncrono em Python

Nesta etapa, vamos configurar e executar o backend em Python para lidar com requisições assíncronas.

### Passo 1: Navegue até a Pasta `app`

A partir da pasta `database`, volte para a raiz do projeto e entre na pasta `app`:

```bash
cd ../app
```

### Passo 2: Execute o Backend

Execute o servidor HTTP com o comando abaixo:

```bash
python3 serverhttp.py
```

Isso iniciará o servidor backend assíncrono em Python, que ficará disponível em [http://localhost:8000/](http://localhost:8000/).

### Passo 3: Configuração do Backend

O backend é configurado para fornecer duas rotas principais:

1. **`/gemini` (POST)**: Rota para interagir com o serviço `Gemini`. Envia um prompt e recebe uma resposta do modelo.
2. **`/` (POST)**: Rota principal para lidar com requisições e processar o prompt.

### Exemplo de Requisição para a Rota `/`

Você pode enviar uma requisição para a rota `/` utilizando `curl` ou um cliente de API, como o Postman:

#### Usando `curl`

```bash
curl -X POST http://localhost:8000/ \
-H "Content-Type: application/json" \
-d '{
    "prompt": "Digite aqui sua pergunta",
    "parametro": "algum_parametro_opcional"
}'
```

#### Usando um Cliente de Teste de API

Alternativamente, você pode usar um cliente de teste de API, como o Postman, para enviar a requisição. Basta configurar a URL como `http://localhost:8000/`, definir o método como `POST`, e enviar o corpo da requisição no formato JSON, como no exemplo acima:

```json
{
  "prompt": "Digite aqui sua pergunta",
  "parametro": 0
}
```

### Integrando com o Gemini

Se você quiser integrar com o `Gemini` ('gemini-1.0-pro') para enviar requisições à API, basta configurar o arquivo `.env` com a chave de API gerada. Para isso, adicione a propriedade `GOOGLE_API_KEY` no arquivo `.env`:

```
GOOGLE_API_KEY=seu_token_do_google_gemini
```

Esse token é necessário para autenticação e acesso aos serviços da API do Gemini.

---

Seguindo esses passos, o backend estará configurado e rodando em [http://localhost:8000/](http://localhost:8000/), pronto para processar requisições.

### Comentários: Como a inferencia do LLama foi configurada em termos gerais?

A inferência do Llama é realizada através da função `chat`, que utiliza a biblioteca `ollama` para interagir com o modelo. Veja os principais passos:

1. **Conexão com o Cliente Assíncrono**:  
   A função `connect_to_aioprompt` cria uma conexão com o cliente assíncrono do Ollama, que será usado para a comunicação com o modelo Llama:

   ```python
   async def connect_to_aioprompt():
       client = AsyncClient(host="http://localhost:11435")
       return client
   ```

2. **Formatação da Consulta e Execução do Chat**:  
   A função `chat` utiliza o cliente para enviar uma consulta ao modelo Llama. A consulta é formatada por meio da função `formatPrompt`, que inclui o resultado da função `run_search_query` para melhorar o contexto:

   ```python
   query = formatPrompt(message, run_search_query(message))
   ```

3. **Envio da Solicitação ao Modelo Llama**:  
   A solicitação é feita usando o método `client.chat` com o modelo `llama3`, transmitindo a mensagem formatada. A inferência é realizada em streaming, retornando as partes do resultado à medida que são geradas:

   ```python
   stream = await client.chat(model='llama3', messages=[{"role": "user", "content": query}], stream=True, options={'num_gpu':1})
   ```

4. **Processamento das Respostas**:  
   O código itera sobre o stream de respostas do modelo, imprimindo cada parte:

   ```python
   async for part in stream:
       print(f"Streamed part: {json.dumps(part)}")
       yield f"{json.dumps(part)}\n"
   ```

Esse fluxo assíncrono permite receber a resposta do modelo Llama em partes, processando-a em tempo real.

## Etapa 6: Executando a Interface Angular

Nesta etapa, vamos configurar e rodar a interface Angular disponível no repositório para que o frontend possa se comunicar com o backend.

### Antes de Começar

O arquivo `app/serverhttp.py`, que configura o servidor HTTP do backend, está preparado com o CORS para aceitar requisições de clientes provenientes do endereço `http://localhost:4200`. Isso permite que a interface Angular possa se comunicar com o backend sem restrições de origem.

### Passo 1: Clone o Repositório da Interface Angular

Clone o repositório necessário para configurar a interface Angular:

```bash
git clone https://github.com/gabrielbtera/assistente-da-gente-front
```

### Passo 2: Navegue até a Raiz do Projeto Angular

Entre no diretório raiz do projeto clonado:

```bash
cd assistente-da-gente-front
```

### Passo 3: Construa a Imagem Docker para a Interface Angular

Com o Docker instalado e executando, rode o comando abaixo para criar a imagem Docker:

```bash
docker build -t angular-ssr-websocket-client .
```

Esse comando cria uma imagem Docker chamada `angular-ssr-websocket-client` a partir do `Dockerfile` localizado no diretório atual, que contém as instruções para a construção do projeto Angular com suporte para renderização no servidor e WebSocket.

### Passo 4: Execute o Contêiner com a Interface Angular

Agora, execute o contêiner utilizando o comando:

```bash
docker run -d -p 4200:4200 angular-ssr-websocket-client
```

- O contêiner será executado em segundo plano (`-d`), com a porta 4200 mapeada para o host, tornando o frontend acessível em [http://localhost:4200](http://localhost:4200).

---

Seguindo esses passos, o projeto Angular estará rodando e disponível na porta 4200 do seu dispositivo, pronto para interagir com o backend que está na 8000.

[Imagem da tela]

### Comentários: Como o front end recebe os pedaços das respostas?

Aqui estão os trechos principais para o recebimento e processamento de dados em tempo real:

1. Recebimento de Dados (`getStreamData`)

Configura a solicitação HTTP para receber dados em "chunks":

```typescript
getStreamData(prompt: string, isGemini = false): Observable<any> {
  const endpoint = this.url + (isGemini ? '/gemini' : '');
  return this.http
    .post(endpoint, { prompt, parametro: 0 }, {
      headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
      responseType: 'text' as 'json',
      observe: 'events' as 'events',
      reportProgress: true
    })
    .pipe(map((event) => isGemini ? this.getDataGemini(event) : this.obterDadosDoEvento(event)));
}
```

2. Processamento de "Chunks" (`obterDadosDoEvento`)

Atualiza a resposta conforme os "chunks" chegam:

```typescript
private obterDadosDoEvento(event: any): string | void | boolean {
  if (event.type === HttpEventType.DownloadProgress) {
    const objeto = JSON.parse(event.partialText.trim().split('\n').pop()!);
    if (!objeto.done) {
      this.response += objetosJson.message.content;
      return objetosJson.message.content;
    }
  } else if (event.type === HttpEventType.Response) {
    return true;
  }
}
```

3. Atualização do Histórico do Chat (`updateCurrentPrompt`)

Concatena os "chunks" ao histórico do chat:

```typescript
updateCurrentPrompt(chunk: string) {
  const element = this.historyChat[this.historyChat.length - 1];
  element.outputChat += chunk;
  element.flagLoading = false;
}
```

4. A resposta é renderizada na tela usando o ngx-markdown, que interpreta e exibe o conteúdo em formato Markdown, permitindo uma apresentação mais rica e formatada das respostas recebidas do backend. O ngx-markdown é uma biblioteca usada para renderizar conteúdo Markdown no Angular.

```html
<markdown
  style="font-size: 14px; color: #4c566a"
  ngPreserveWhitespaces
  [data]="textResponse"
>
</markdown>
```

Esses trechos permitem o processamento incremental das respostas, atualizando o chat em tempo real.

## Conclusão

Esse tutorial apresentou um guia detalhado para configurar uma aplicação de Recuperação Aumentada por Geração (RAG) on-premise, utilizando Llama3-8b, MongoDB e uma GPU Nvidia. Passamos por todas as etapas necessárias, desde a configuração da GPU com os drivers NVIDIA, execução do Ollama com suporte a GPU, configuração do MongoDB com Docker Compose, processamento e inserção de documentos PDF no banco de dados, até a execução do backend assíncrono em Python e a interface Angular. Seguindo os passos propostos, você terá um ambiente completo e funcional para experimentação e desenvolvimento de aplicações RAG.

<!-- # estudostccc

## Para executar o modelo ollma na nvidia faça:

Configure as ferramentas da nvidia em https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation :

#### Passo 1:

```curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### Passo 2:

```
sudo apt-get update

```

#### Passo 3:

```
sudo apt-get install -y nvidia-container-toolkit
```

#### Passo 4:

Agora, com o docker instalado execute

```
- docker run -it  --gpus=all -v /home/ollama:/root/.ollama:z -p 11435:11434 --name ollama ollama/ollama
- docker exec -it <container_id> /bin/bash


```

Obs.: O container vai iniciar exibindo logs

#### Passo 5:

Abra um terminal paralelo e execute

```
docker exec -it ollama ollama pull llama3

```

### Passo 6

Agora eh so executar o trecho de codigo que invoca o llama3

## Soluções para problemas na execução do notebook

- se tiver um problema com en_core_web_sm, execute: `python -m spacy download en_core_web_sm`

https://medium.com/@blackhorseya/running-llama-3-model-with-nvidia-gpu-using-ollama-docker-on-rhel-9-0504aeb1c924

https://dev.to/berk/running-ollama-and-open-webui-self-hosted-4ih5 -->

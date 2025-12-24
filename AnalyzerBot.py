import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import os
    from langchain_core.tools import tool
    from langchain_mistralai import ChatMistralAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    from dotenv import load_dotenv
    from langgraph_supervisor import create_supervisor
    from langgraph.checkpoint.memory import InMemorySaver

    from pathlib import Path
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.tools.retriever import create_retriever_tool
    from langchain_core.documents import Document
    from langchain_core.documents.base import Blob
    return (
        ChatMistralAI,
        Document,
        HuggingFaceEmbeddings,
        InMemoryVectorStore,
        Path,
        create_react_agent,
        create_retriever_tool,
        os,
        tool,
    )


@app.cell
def _():
    from marimo import cache
    import subprocess

    @cache                    # la prima esecuzione avvierà il server; le successive restituiranno lo stesso processo  
    def avvia_mlflow_server():
        cmd = [
            "mlflow", "server",
            "--backend-store-uri",   "sqlite:///mlflow.db",
            "--default-artifact-root", "./artifacts",
            "--host",                "0.0.0.0",
            "--port",                "1414",
        ]
        p = subprocess.Popen(cmd)
        return f"MLflow UI avviato su http://localhost:1414 (PID={p.pid})"

    print(avvia_mlflow_server())

    import mlflow
    mlflow.set_tracking_uri("http://localhost:1414")
    mlflow.config.enable_async_logging()
    mlflow.langchain.autolog(exclusive=False)
    return


@app.cell
def _(ChatMistralAI, getpass, os):
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass()

    model = ChatMistralAI(model_name="mistral-small")
    return (model,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Definizione Tools""")
    return


@app.cell
def _(tool):
    @tool
    def create_flowchart(flow_model: dict) -> str:
        """
        Converte un modello di flusso strutturato in un diagramma Mermaid (flowchart TD).

        Il tool riceve in input un dizionario che rappresenta la struttura logica
        del flusso di un programma (tipicamente estratto da codice COBOL) e genera
        una stringa in sintassi Mermaid pronta per essere renderizzata.

        Struttura attesa di `flow_model`:
        {
            "flow": [
                {
                    "id": "N1",
                    "type": "start" | "end" | "process" | "decision",
                    "label": "Descrizione del nodo"
                },
                ...
            ],
            "edges": [
                {
                    "from": "N1",
                    "to": "N2"
                },
                {
                    "from": "N2",
                    "to": "N3",
                    "condition": "Sì | No | EOF | ecc."
                },
                ...
            ]
        }
        Regole di rendering:
        - I nodi di tipo "start" e "end" vengono rappresentati come nodi terminali.
        - I nodi di tipo "decision" vengono rappresentati come rombi.
        - Tutti gli altri nodi vengono rappresentati come processi standard.
        - Gli archi possono includere una condizione opzionale visualizzata sul collegamento.

        Args:
            flow_model (dict): Modello di flusso strutturato contenente nodi e collegamenti.

        Returns:
            str: Stringa in sintassi Mermaid (flowchart TD), senza blocchi Markdown,
                 pronta per il rendering automatico.
        """
        lines = ["flowchart TD"]

        for node in flow_model["flow"]:
            if node["type"] in ("start", "end"):
                lines.append(f'{node["id"]}(["{node["label"]}"])')
            elif node["type"] == "decision":
                lines.append(f'{node["id"]}{{"{node["label"]}"}}')
            else:
                lines.append(f'{node["id"]}["{node["label"]}"]')

        for edge in flow_model["edges"]:
            if "condition" in edge:
                lines.append(
                    f'{edge["from"]} -->|{edge["condition"]}| {edge["to"]}'
                )
            else:
                lines.append(
                    f'{edge["from"]} --> {edge["to"]}'
                )

        return "\n".join(lines)
    return (create_flowchart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## RAG""")
    return


@app.cell
def _(HuggingFaceEmbeddings):
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return (embedding_model,)


@app.cell
def _(Path):
    async def load_cobol_sources(
        base_dir: str,
        extensions: tuple[str, ...] = (".cob", ".cbl")
    ):
        """
        Carica file COBOL locali
        """
        base_path = Path(base_dir)
        blobs = []

        for path in base_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                content = path.read_text(encoding="utf-8", errors="ignore")
                blobs.append({
                    "content": content,
                    "source": path.name,
                    "path": str(path)
                })

        return blobs
    return (load_cobol_sources,)


@app.cell
def _(Document):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    def cobol_chunk_documents(blobs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=[
                "\n       IDENTIFICATION DIVISION.",
                "\n       DATA DIVISION.",
                "\n       PROCEDURE DIVISION.",
                "\n       ",
                "\n"
            ],
        )

        documents = []

        for blob in blobs:
            chunks = splitter.split_text(blob["content"])
            for i, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": blob["source"],
                            "path": blob["path"],
                            "chunk": i,
                            "language": "cobol"
                        }
                    )
                )

        return documents

    return (cobol_chunk_documents,)


@app.cell
def _(
    InMemoryVectorStore,
    cobol_chunk_documents,
    embedding_model,
    load_cobol_sources,
):
    async def create_in_memory_retriever():
        print("Ingesting COBOL sources...")

        blobs = await load_cobol_sources(base_dir="./cobol/")
        documents = cobol_chunk_documents(blobs)

        print(f"Loaded {len(documents)} COBOL chunks")

        vector_store = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=embedding_model
        )

        return vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
    return (create_in_memory_retriever,)


@app.cell
async def _(create_in_memory_retriever, create_retriever_tool):
    retriever = await create_in_memory_retriever()

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="cobol_code_retriever",
        description=(
            "Recupera porzioni di codice COBOL locali "
            "per analisi, debug e spiegazione."
        )
    )
    return (retriever_tool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Definizione Prompt""")
    return


@app.cell
def _():
    analyzer_prompt = """
    You are Analyzer, an expert assistant specialized in the analysis of legacy enterprise applications,
    with a strong focus on COBOL-based systems.

    Your responsibilities:

    1. Source Code Analysis
       - Analyze ONLY the COBOL source code retrieved through the available tools.
       - The user will indicate the name of the COBOL file to analyze.
       - You MUST rely on the retriever to obtain the source code.
       - You MUST NOT invent, assume, or extrapolate code that is not explicitly retrieved.

    2. From the analyzed COBOL code, you may be asked to produce one or more of the following outputs:

       A. Functional Description
          - Explain the business purpose of the program in clear, non-technical language.
          - Describe the main functional flow of the application.
          - Identify inputs, outputs, and key processing steps.
          - Highlight any relevant business rules.

       B. Component Inventory (Censimento delle componenti)
          - COBOL Programs:
            - Name of the program analyzed.
          - Files:
            - Input files (READ, START)
            - Output files (WRITE, REWRITE, DELETE)
          - Databases:
            - Indexed or sequential files
            - Any reference to database access (EXEC SQL), if present.
          - JCL:
            - Indicate whether the program is batch-oriented.
            - Identify expected JCL elements (DD statements, input/output datasets),
              even if the JCL itself is not provided.
          - Maps / UI:
            - Identify usage of BMS, CICS MAPs, or screen handling (ACCEPT / DISPLAY).
          - External Components:
            - CALL to external programs, subprograms, or utilities.

       C. Program Flow Structure
          - When requested, extract the control flow of the program (PROCEDURE DIVISION).
          - Represent the flow as a structured model suitable for diagram generation.
          - Do NOT generate diagrams directly.

    3. Output Rules (VERY IMPORTANT)

       - You MUST ALWAYS return a SINGLE valid JSON object as the final output.
       - The JSON object MUST follow this structure:

         {
           "type": "<output_type>",
           "content": <output_content>
         }

       - Allowed values for "type":
         - "text"    → for functional descriptions and component inventories
         - "json"    → for structured data or intermediate models
         - "mermaid" → ONLY if the final result is a Mermaid diagram produced via a tool

       - If the output is:
         - "text": content MUST be a Markdown-formatted string in Italian.
         - "json": content MUST be a valid JSON object.
         - "mermaid": content MUST be valid Mermaid syntax ONLY (no Markdown fences).

    4. Diagram Generation Rule

       - You MUST NOT generate Mermaid syntax directly.
       - When a diagram is requested:
         1. Extract the program flow as a structured JSON model.
         2. Call the tool "create_flowchart" to convert the model into Mermaid syntax.
         3. Return ONLY the tool result, wrapped in:
            {
              "type": "mermaid",
              "content": "<mermaid syntax>"
            }

    5. Analysis Constraints
       - Your analysis must be based strictly on what is present in the retrieved code.
       - If information is missing or cannot be determined, explicitly state:
         "Non inferibile dal codice sorgente disponibile."
       - Do NOT execute code.
       - Do NOT modify code.
       - Do NOT hallucinate missing components.
       - Do NOT assume runtime environment details unless explicitly stated in the code.

    6. Language and Style
       - Write all human-readable content in Italian.
       - Use clear section titles and bullet points where appropriate.
       - Keep the style concise, professional, and suitable for technical documentation.
    """
    return (analyzer_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Definizione Agente""")
    return


@app.cell
def _(
    analyzer_prompt,
    create_flowchart,
    create_react_agent,
    mo,
    model,
    retriever_tool,
):
    analyzer_agent = create_react_agent(
        model=model,
        tools=[retriever_tool, create_flowchart],
        prompt=analyzer_prompt,
        name="Analyzer",
    )

    mo.mermaid(analyzer_agent.get_graph(xray=True).draw_mermaid())
    return (analyzer_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Esecuzione""")
    return


@app.function
def print_messages(messages):
    for msg in messages["messages"]:
        tipo = type(msg).__name__  # es: 'HumanMessage', 'AIMessage'
        nome = getattr(msg, "name", "N/A")
        contenuto = msg.content

        # Se il contenuto � strutturato (lista di dict), estrai i testi
        if isinstance(contenuto, list):
            testo = "\n".join(block.get("text", "") for block in contenuto if block.get("type") == "text")
        else:
            testo = contenuto

        print(f"[Tipo: {tipo}] [Agente: {nome}]\nContenuto: {testo}\n{'-'*50}")


@app.cell
def _(mo):
    user_prompt_1 = mo.ui.text()
    run_button_1 = mo.ui.run_button()
    user_prompt_1, run_button_1
    # Analizza il file BUY_ROUTINE.COB
    # Mi crei il flowchart del file BUY_ROUTINE.COB
    return run_button_1, user_prompt_1


@app.cell
def _(analyzer_agent, mo, run_button_1, user_prompt_1):
    mo.stop(not run_button_1.value, mo.md("Click 👆 to run this cell"))
    config = {"configurable": {"thread_id": "983eb4db-579d-c844-783f-c3a9bcec929f"}}
    turn_1  = analyzer_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt_1.value
                }
            ]
        }, 
        config,
    )
    return (turn_1,)


@app.cell
def _(turn_1):
    print_messages(turn_1)
    return


@app.cell
def _(turn_1):
    turn_1["messages"][-1].content
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analisi Risultato""")
    return


@app.cell
def _():
    import json
    import re
    from typing import Union, Dict

    # funzione per sostituire doppi apici interni con singoli
    def fix_mermaid_quotes(code):
        # cerca doppi apici interni dentro le parentesi quadre
        # cattura tutto ciò che è tra ["] e "]"
        pattern = re.compile(r'\["(.*?)"\]')

        def replacer(match):
            content = match.group(1)
            # sostituisci i doppi apici interni con singoli
            fixed_content = content.replace('"', "'")
            return f'["{fixed_content}"]'

        return pattern.sub(replacer, code)

    def parse_llm_output(content: str) -> Dict[str, Union[str, dict]]:
        """
        Normalizza l'output di un LLM che può restituire:
        - JSON fenced (```json)
        - Mermaid fenced (```mermaid)
        - Testo semplice

        Restituisce sempre:
        {
            "type": "json" | "mermaid" | "text",
            "content": ...
        }
        """

        text = content.strip()

        # Caso 1: fenced JSON
        if text.startswith("```json"):
            cleaned = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL)
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {
                "type": "json",
                "content": parsed
            }

        # Caso 2: fenced Mermaid
        if text.startswith("```mermaid"):
            cleaned = re.sub(r"^```mermaid\s*|\s*```$", "", text, flags=re.DOTALL)
            fixed_code = fix_mermaid_quotes(cleaned.strip())
            return {
                "type": "mermaid",
                "content": fixed_code
            }

        # Caso 4: testo normale
        return {
            "type": "text",
            "content": text
        }

    return (parse_llm_output,)


@app.cell
def _(parse_llm_output, turn_1):
    result = parse_llm_output(turn_1["messages"][-1].content)
    return (result,)


@app.cell
def _(mo, result):
    from IPython.display import display

    if result['type'] == 'mermaid':
        display(mo.mermaid(result['content']))
    elif result['type'] == 'text':
        display(mo.md(result['content'])) 

    return


if __name__ == "__main__":
    app.run()

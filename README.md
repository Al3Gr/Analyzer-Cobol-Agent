# Analyzer – COBOL Code Analysis Agent

Analyzer is an AI-based agent designed to analyze legacy **COBOL code** and provide structured insights based on user requests.  
The COBOL sources are expected to be placed inside the dedicated `cobol/` directory.

The agent can generate:
- Functional analyses
- Component inventories
- Flowcharts representing the program logic

The project is built using **LangChain** and **LangGraph** for agent orchestration and relies on **Mistral-Small** as the underlying Large Language Model (LLM).

---

## Features

Analyzer supports the following types of analysis:

- **Functional Analysis**  
  Provides a high-level explanation of what the COBOL program does, its business logic, and main responsibilities.

- **Component Inventory**  
  Extracts and lists key components such as:
  - Programs
  - Sections and paragraphs
  - Files and data structures
  - External dependencies

- **Flowchart Generation**  
  Produces a logical flowchart of the COBOL program, typically represented using **Mermaid** syntax for easy visualization.

---

## Architecture Overview

The system is implemented as an **agent-based workflow**:

- **Analyzer Agent**  
  Acts as the main reasoning unit, interpreting user requests and deciding which type of analysis to perform.

- **Retrieval Layer**  
  COBOL source files are ingested, chunked, and retrieved to provide the agent with relevant context for analysis.

- **LLM Integration**  
  The agent uses **Mistral-Small** to reason over the retrieved code and generate structured outputs.

- **LangGraph Orchestration**  
  LangGraph is used to define and manage the execution flow of the agent and its tools.

---

## Project Structure

- cobol/ # COBOL source files to be analyzed
- AnalyzerBot.py # Main application and agent definition
- README.md


---

## Technologies Used

- **Python**
- **LangChain**
- **LangGraph**
- **Mistral-Small (LLM)**
- **Marimo** (for interactive execution and visualization)
- **Mermaid** (for flowchart rendering)

---

## Example Outputs

Depending on the request, Analyzer can return:
- A textual functional description of the COBOL program
- A structured inventory of program components
- A Mermaid-based flowchart describing the execution logic

---

## Purpose

This project is intended to support:
- Understanding and documenting legacy COBOL applications
- Assisting in system analysis and modernization efforts
- Reducing the manual effort required to interpret complex legacy codebases

# Iraqi-LLM-Dataset-Generator
Iraqi LLM Dataset Generator

# Overview
This project focuses on building a specialized Iraqi Arabic dataset generation pipeline for training and fine-tuning Large Language Models (LLMs).
The main goal is to create an AI model that can understand and respond using natural Iraqi dialect while also being specialized in technical and educational domains.
The system automatically converts educational PDF documents into structured Question-and-Answer datasets in both:
*	Formal Arabic 
*	Iraqi Dialect Arabic 

The generated output is saved in JSONL format, which is the standard format used for LLM fine-tuning.

# Project Goal
Most existing LLMs are trained mainly on:

* English
* Modern Standard Arabic (MSA)

However, Iraqi dialect data is very limited.

This project solves that problem by automatically generating high-quality Iraqi instruction datasets from books, lectures, and educational materials.

The final objective is to train or fine-tune an Iraqi-specialized LLM capable of:

* Understanding Iraqi dialect
* Answering in Iraqi dialect
* Understanding technical terminology
* Supporting educational and scientific topics

# Pipeline Workflow

The system works through the following stages:

## 1. PDF Processing
Educational PDFs such as:

* Books
* Lecture notes
* Technical documents
* Academic material

are processed automatically.

## 2. Text Extraction
Text content is extracted page-by-page from the PDF documents.

The extracted text is cleaned and prepared for AI processing.

## 3. AI-Powered Question Generation

Using a local LLM through Ollama and Qwen2.5, the system generates:

* Formal Arabic Questions
* Formal Arabic Answers

based on the educational content.

## 5. JSON Repair and Validation

Since local LLMs may sometimes generate malformed JSON, the pipeline includes:

* JSON cleaning
* Error correction
* Validation
* Filtering corrupted outputs

This ensures stable large-scale dataset generation.

# Technologies Used
## AI Models
* Qwen2.5
* Ollama

# Python Libraries
* PyMuPDF (fitz)
* requests
* tqdm
* json
* concurrent.futures

# Features
* Automatic PDF-to-Dataset conversion
* Iraqi dialect generation
* Formal Arabic QA generation
* JSON auto-repair
* Local offline AI processing
* JSONL export for fine-tuning
* Multi-page dataset generation
* Educational dataset specialization

# Why This Project Matters
This project addresses a major limitation in Arabic AI systems:

The lack of high-quality Iraqi dialect instruction datasets.

By generating Iraqi educational QA datasets, this project enables the development of:

* Iraqi educational assistants
* Iraqi technical chatbots
* Iraqi tutoring systems
* Iraqi domain-specific LLMs

# Future Goals
## Future development may include:
* Larger Iraqi datasets
* Speech-to-text Iraqi datasets
* Iraqi conversational datasets
* Fine-tuning larger LLMs
* Iraqi voice assistants
* Multi-domain Iraqi AI systems

# Final Objective
The ultimate goal is to build a specialized Iraqi Large Language Model capable of:

* Understanding Iraqi users naturally
* Responding using Iraqi dialect
* Handling technical and educational questions
* Supporting Arabic AI research and development

"""
summarizer.py — T5 / LLaMA research-paper summarizer + chatbot.

Merges the former model.py, database_handler.py, llama_model.py, data_pre.py,
chatbot.py and main.py (T5 pipeline) into one module.

Pipeline:
  1. build-db    Extract text from PDFs → SQLite 'works' table
  2. summarize   T5 summarizes every unsummarized work
  3. fine-tune   Fine-tune T5 on the (full_text → summary) pairs
  chat           Interactive LLaMA chatbot grounded in the stored summaries

CLI:
  python summarizer.py build-db
  python summarizer.py summarize
  python summarizer.py fine-tune
  python summarizer.py pipeline      # build-db → summarize → fine-tune
  python summarizer.py chat

Config via environment variables (defaults preserve the original hardcoded paths):
  T5_DB_PATH          (default C:\\codes\\t5-db\\researchers.db)
  T5_PDF_FOLDER       (default C:\\codes\\t5-db\\download_pdfs)
  T5_MODEL            (default t5-small)
  T5_FINETUNE_DIR     (default C:\\codes\\t5-db\\fine_tuned_t5)
  LLAMA_MODEL_PATH    (default C:\\codes\\llama32\\Llama-3.2-1B-Instruct)
  LLAMA_FINETUNE_DIR  (default C:\\codes\\llama32\\fine_tuned_llama)

Deliberate changes made during the merge (documented, not silent):
  * The three duplicate clear_memory() copies are now one.
  * T5 and LLaMA are loaded LAZILY (on first use) instead of at import time,
    so importing this module — or running a T5-only command — no longer loads
    both models onto the GPU.
  * The SQLite connection is opened lazily instead of at import time, so the
    module imports cleanly on machines without the hardcoded DB path.
  * extract_text_from_pdf is implemented here (pdfminer → pypdf) as a drop-in
    replacement for the original `pdf_pre` module, which was not provided.
"""

import gc
import os
import sqlite3

# ── Config (env-overridable; defaults preserve the original hardcoded paths) ──
DB_PATH            = os.getenv("T5_DB_PATH",         r"C:\codes\t5-db\researchers.db")
PDF_FOLDER         = os.getenv("T5_PDF_FOLDER",      r"C:\codes\t5-db\download_pdfs")
T5_MODEL_NAME      = os.getenv("T5_MODEL",           "t5-small")
T5_FINETUNE_DIR    = os.getenv("T5_FINETUNE_DIR",    r"C:\codes\t5-db\fine_tuned_t5")
LLAMA_MODEL_PATH   = os.getenv("LLAMA_MODEL_PATH",   r"C:\codes\llama32\Llama-3.2-1B-Instruct")
LLAMA_FINETUNE_DIR = os.getenv("LLAMA_FINETUNE_DIR", r"C:\codes\llama32\fine_tuned_llama")


# ═══════════════════════════════════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════════════════════════════════

def clear_memory():
    """Clear GPU and CPU memory (single shared copy)."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Lazy model loaders  (were module-level loads in model.py / llama_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

_T5 = None
_LLAMA = None


def _get_t5():
    """Load and cache (model, tokenizer, device) for T5 on first use."""
    global _T5
    if _T5 is None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading T5 model '{T5_MODEL_NAME}' on {device}...")
        model     = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(device)
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
        _T5 = (model, tokenizer, device)
    return _T5


def _get_llama():
    """Load and cache (model, tokenizer, device) for LLaMA on first use."""
    global _LLAMA
    if _LLAMA is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading LLaMA model from:", LLAMA_MODEL_PATH)
        model     = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH).to(device)
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
        # Gradient checkpointing to reduce VRAM during fine-tuning
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            model.config.use_cache = False
        _LLAMA = (model, tokenizer, device)
    return _LLAMA


# ═══════════════════════════════════════════════════════════════════════════════
# PDF text extraction  (replaces the missing pdf_pre module)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF (pdfminer.six, then pypdf fallback).

    Drop-in replacement for the original `pdf_pre.extract_text_from_pdf`,
    which was not among the provided files.
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
        if text and text.strip():
            return text.strip()
    except ImportError:
        print("pdfminer.six not installed — run: pip install pdfminer.six")
    except Exception as e:
        print(f"pdfminer failed for {file_path}: {e}")

    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages  = [p.extract_text() or "" for p in reader.pages]
        text   = "\n\n".join(p.strip() for p in pages if p.strip())
        if text and text.strip():
            return text.strip()
    except Exception as e:
        print(f"pypdf failed for {file_path}: {e}")

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Data preprocessing  (was data_pre.py)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_text_for_t5(text, model_name=None):
    """Prepare text for T5 summarization (task prefix + truncation)."""
    from transformers import T5Tokenizer
    tokenizer  = T5Tokenizer.from_pretrained(model_name or T5_MODEL_NAME)
    input_text = "summarize: " + text[:1000]  # Truncate to the first 1000 characters
    return tokenizer(input_text, truncation=True, max_length=512, padding="max_length")


# ═══════════════════════════════════════════════════════════════════════════════
# T5 summarization + fine-tuning  (was model.py)
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_text(text, idx=None, total=None):
    """Generate a summary for the given text using T5."""
    model, tokenizer, device = _get_t5()
    if idx is not None and total is not None:
        print(f"Summarizing text [{idx}/{total}]...")
    input_text = "summarize: " + text[:1000]  # Truncate to the first 1000 characters
    inputs  = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    clear_memory()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def fine_tune_t5_on_papers(dataset, output_dir=None):
    """Fine-tune the T5 model on the papers dataset (needs input_text + summary)."""
    output_dir = output_dir or T5_FINETUNE_DIR
    model, tokenizer, device = _get_t5()
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    print("Preparing dataset for fine-tuning...")
    if 'input_text' not in dataset.columns or 'summary' not in dataset.columns:
        raise ValueError("Dataset must contain 'input_text' and 'summary' columns!")

    def tokenize_function(examples):
        inputs  = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(examples['summary'],    padding="max_length", truncation=True, max_length=150)
        inputs['labels'] = targets['input_ids']
        return inputs

    hf_dataset        = Dataset.from_pandas(dataset)
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=2,
        save_steps=500,
        evaluation_strategy="no",
        logging_dir="./logs",
        logging_steps=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)

    print("Starting fine-tuning on GPU...")
    trainer.train()
    print(f"Saving the fine-tuned model to {output_dir}...")
    trainer.save_model(output_dir)
    print("Model fine-tuned and saved successfully!")
    clear_memory()
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# LLaMA answering + fine-tuning  (was llama_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

def chatbot_answer(question, context):
    """Generate an answer to a question given background context, using LLaMA."""
    model, tokenizer, device = _get_llama()
    prompt  = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        **inputs, max_length=256, do_sample=True,
        temperature=0.7, top_p=0.9, num_return_sequences=1,
    )
    clear_memory()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def fine_tune_llama_on_papers(dataset, output_dir=None):
    """Fine-tune LLaMA on a dataset with columns input_text + target_text."""
    output_dir = output_dir or LLAMA_FINETUNE_DIR
    model, tokenizer, device = _get_llama()
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    if 'input_text' not in dataset.columns or 'target_text' not in dataset.columns:
        raise ValueError("Dataset must contain 'input_text' and 'target_text' columns!")

    def tokenize_function(examples):
        prompts      = [f"Research Paper: {inp}\nSummary:" for inp in examples['input_text']]
        model_inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_text'], truncation=True, padding="max_length", max_length=150)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    hf_dataset        = Dataset.from_pandas(dataset)
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_dataset, eval_dataset=tokenized_dataset,
    )

    print("Starting fine-tuning for LLaMA model...")
    trainer.train()
    print(f"Saving fine-tuned LLaMA model to {output_dir}...")
    trainer.save_model(output_dir)
    clear_memory()
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite handler  (was database_handler.py — now lazily connected)
# ═══════════════════════════════════════════════════════════════════════════════

_CONN = None


def _get_conn():
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(DB_PATH)
    return _CONN


def setup_database():
    """Create the 'works' table if it doesn't exist."""
    conn = _get_conn()
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS works (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT UNIQUE,
            full_text TEXT NOT NULL,
            summary TEXT,
            summary_status TEXT DEFAULT 'unsummarized',
            progress INTEGER DEFAULT 0
        )
    """)
    conn.commit()


def remove_duplicates():
    """Remove duplicate entries in the 'works' table."""
    conn = _get_conn()
    conn.cursor().execute("""
        DELETE FROM works
        WHERE id NOT IN (SELECT MIN(id) FROM works GROUP BY file_name)
    """)
    conn.commit()
    print("Duplicates removed from works table.")


def insert_work(file_name, full_text, summary=None, summary_status="unsummarized", progress=0):
    """Insert a new work into the database, ensuring no duplicates."""
    conn = _get_conn()
    try:
        conn.cursor().execute("""
            INSERT INTO works (file_name, full_text, summary, summary_status, progress)
            VALUES (?, ?, ?, ?, ?)
        """, (file_name, full_text, summary, summary_status, progress))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Skipping duplicate file: {file_name}")


def fetch_unsummarized_works(limit=None):
    """Fetch all unsummarized works from the database."""
    cursor = _get_conn().cursor()
    query = """
        SELECT id, full_text FROM works
        WHERE summary_status = 'unsummarized' AND progress = 0
    """
    if limit:
        cursor.execute(query + " LIMIT ?", (limit,))
    else:
        cursor.execute(query)
    return cursor.fetchall()


def update_summary(work_id, summary):
    """Update the summary for a specific work."""
    conn = _get_conn()
    conn.cursor().execute("""
        UPDATE works
        SET summary = ?, summary_status = 'summarized', progress = 1
        WHERE id = ?
    """, (summary, work_id))
    conn.commit()


def count_entries_in_table():
    """Count the total number of entries in the database."""
    cursor = _get_conn().cursor()
    cursor.execute("SELECT COUNT(*) FROM works")
    return cursor.fetchone()[0]


def check_missing_files_in_db(pdf_files):
    """Check which files in the folder are missing in the database."""
    cursor = _get_conn().cursor()
    cursor.execute("SELECT file_name FROM works")
    db_files = {row[0] for row in cursor.fetchall()}
    return set(pdf_files) - db_files


def close_connection():
    """Close the database connection."""
    global _CONN
    if _CONN is not None:
        _CONN.close()
        _CONN = None


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline steps  (was main.py — T5 pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def populate_database_from_pdfs():
    """Step 1: Process all PDFs in the folder, store text in DB."""
    setup_database()
    remove_duplicates()

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    if total_files == 0:
        print("No PDF files found in the directory.")
        return

    print(f"Found {total_files} PDF files in the folder.")
    missing_files = check_missing_files_in_db(pdf_files)
    print(f"{len(missing_files)} files missing from the database. Processing these files...")

    for idx, file_name in enumerate(missing_files, start=1):
        file_path = os.path.join(PDF_FOLDER, file_name)
        print(f"[{idx}/{len(missing_files)}] Processing: {file_name}")
        extracted_text = extract_text_from_pdf(file_path)
        if extracted_text:
            insert_work(file_name=file_name, full_text=extracted_text,
                        summary=None, summary_status="unsummarized", progress=0)

    print(f"Database populated with all PDFs. Total entries in database: {count_entries_in_table()}")


def generate_summaries_for_database():
    """Step 2: Summarize all works in the database."""
    unsummarized_works = fetch_unsummarized_works()
    if not unsummarized_works:
        print("No unsummarized works found.")
        return

    print(f"Found {len(unsummarized_works)} unsummarized works. Generating summaries...")
    for idx, (work_id, full_text) in enumerate(unsummarized_works, start=1):
        try:
            summary = summarize_text(full_text, idx=idx, total=len(unsummarized_works))
            update_summary(work_id, summary)
            print(f"Summary updated for work ID: {work_id}")
        except Exception as e:
            print(f"Error summarizing work ID {work_id}: {e}")
        clear_memory()


def fine_tune_model_on_summaries():
    """Step 3: Fine-tune T5 on all summarized data."""
    import pandas as pd
    print("Preparing data for fine-tuning...")
    conn = _get_conn()
    query = """
        SELECT full_text, summary
        FROM works
        WHERE summary_status = 'summarized' AND progress = 1
    """
    papers_df = pd.read_sql_query(query, conn)

    if papers_df.empty:
        print("No summarized data available for fine-tuning.")
        return

    if 'full_text' in papers_df.columns and 'summary' in papers_df.columns:
        papers_df = papers_df.rename(columns={"full_text": "input_text", "summary": "target_text"})
        papers_df = papers_df.rename(columns={"target_text": "summary"})
        print(f"Fine-tuning on {len(papers_df)} summarized entries...")
        output_dir = fine_tune_t5_on_papers(papers_df, T5_FINETUNE_DIR)
        print(f"Fine-tuned model saved at: {output_dir}")
    else:
        print("Error: Dataset must contain 'input_text' and 'summary' columns!")

    clear_memory()


def run_pipeline():
    """Full T5 pipeline: build-db → summarize → fine-tune."""
    try:
        print("STEP 1: Populating the database from PDFs...")
        populate_database_from_pdfs()
        print("STEP 2: Generating summaries...")
        generate_summaries_for_database()
        print("STEP 3: Fine-tuning the model on summaries...")
        fine_tune_model_on_summaries()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        close_connection()
        print("Pipeline completed and database connection closed.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chatbot  (was chatbot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_context():
    """Concatenate all stored summaries (summarized + progress=1) into a context string."""
    cursor = _get_conn().cursor()
    cursor.execute("""
        SELECT summary FROM works
        WHERE summary_status = 'summarized' AND progress = 1
    """)
    results = cursor.fetchall()
    return "\n".join(row[0] for row in results if row[0])


def run_chatbot():
    """Interactive chatbot loop grounded in the stored paper summaries."""
    print("Chatbot is ready! Type your research question (or 'exit' to quit):")
    context = retrieve_context()
    if not context:
        print("No summarized papers found in the database. Please run the pipeline to generate summaries first.")
        return

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        answer = chatbot_answer(question, context)
        print("Chatbot:", answer)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="T5/LLaMA research-paper summarizer + chatbot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-db",  help="Extract text from PDFs into the SQLite DB")
    sub.add_parser("summarize", help="T5-summarize all unsummarized works")
    sub.add_parser("fine-tune", help="Fine-tune T5 on stored (text, summary) pairs")
    sub.add_parser("pipeline",  help="build-db → summarize → fine-tune")
    sub.add_parser("chat",      help="Interactive LLaMA chatbot over stored summaries")
    args = parser.parse_args()

    if args.command == "build-db":
        populate_database_from_pdfs(); close_connection()
    elif args.command == "summarize":
        generate_summaries_for_database(); close_connection()
    elif args.command == "fine-tune":
        fine_tune_model_on_summaries(); close_connection()
    elif args.command == "pipeline":
        run_pipeline()
    elif args.command == "chat":
        try:
            run_chatbot()
        finally:
            close_connection()


if __name__ == "__main__":
    _cli()

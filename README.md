# 🚀 Fine-Tuning Falcon-RW-1B with QLoRA

This project demonstrates how to **fine-tune a large language model (Falcon-RW-1B)** using **QLoRA** on a **custom instruction dataset**.  
The goal was to build a resource-efficient chatbot capable of answering questions in a conversational style.

---

## 📌 Project Overview

- **Base Model** → Falcon-RW-1B 
- **Dataset Used** → Guanaco-LLaMA2-1K (instruction-tuning dataset)
- **Fine-Tuning Method** → **QLoRA** (Quantized LoRA) 
- **Frameworks & Tools** → Hugging Face Transformers, TRL, PEFT, BitsAndBytes, TensorBoard
- **Output** → A fine-tuned chatbot model saved as falcon-1b-finetune


---

## ⚡ Why QLoRA?

- Fine-tuning large models is **memory heavy.**
- LoRA trains only **small adapter layers,** but still requires model in full precision.
- **QLoRA** combines LoRA +**4-bit quantization,** allowing fine-tuning on **smaller GPUs** with much lower VRAM usage.

---

## 🛠 Tech Stack

- **Hugging Face Transformers** → Load & manage LLMs.
- **TRL (Transformers Reinforcement Learning)** → Provides SFTTrainer for supervised fine-tuning.
- **PEFT (Parameter-Efficient Fine-Tuning)** → Implements LoRA adapters. 
- **BitsAndBytes (bnb)** → Enables 4-bit quantization for QLoRA.
- **TensorBoard** → Training visualization (loss curves, logs).
- **PyTorch** → Core deep learning framework.

---

## 🔧 Training Configuration

### LoRA Parameters
- **lora_r = 64** → Rank of LoRA matrices
- **lora_alpha = 16** → Scaling factor
- **lora_dropout = 0.1** → Prevents overfitting

### 🔹 Step 2: Semantic Chunking
- Uses **SemanticChunker** with **Google Embeddings**  
- Breaks the document into **meaningful sections** instead of fixed-size chunks

### 🔹 Step 3: Vector Database (ChromaDB)
- Stores embeddings in **ChromaDB (persistent)**  
- Automatically reuses existing DB if available

### 🔹 Step 4: Query Retrieval
- User query → Expanded into multiple queries using **MultiQueryRetriever**  
- Fetches **top k=3** most relevant chunks from Chroma

### 🔹 Step 5: Prompt + LLM
- Builds a **prompt template** with:  
  - Conversation history  
  - Retrieved context  
  - Latest question  
- Sends it to **Google Gemini (`gemini-1.5-flash`)**

### 🔹 Step 6: Streaming Response
- LLM response streamed **word-by-word**  
- Typing effect for a **natural chat experience**


---
s
## 📊 Example Workflow

### 1. Start the App
- Loads **PDF** + **Chroma DB**

### 2. Ask a Question
- Example: *“Summarize key points”*

### 3. Retrieve Relevant Chunks
- Retriever fetches **top matching sections** from the document

### 4. Generate Response
- **Gemini LLM** creates a contextual answer

### 5. Display Answer
- Chat interface shows the response with a **typing animation**

---

## 📷 Screenshots  

### 1. Welcome Page  
![Welcome Page](screenshots/welcome.png)  

### 2. User Asking Question  
![User Asking Question](screenshots/user_question.png)  

### 3. Bot Responding with Typing Effect  
![Bot Response](screenshots/bot_response.png)  


---

## 📌 Dependencies

### 🐍 Python
- Python **3.9+**

### 📦 Core Libraries
- **Streamlit**  
- **LangChain**  
- **LangChain-Community**  
- **LangChain-Chroma**

### 🗄️ Optional
- **FAISS** → optional backup vector database

### 🤖 AI Model
- **Google Generative AI (Gemini)**

3. **Install via:**
   ```bash
   pip install -r requirements.txt

---


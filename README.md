# SnapChef 👨‍🍳🤖

An AI-powered prototype that blends *Retrieval-Augmented Generation (RAG)* with *Computer Vision (CNN)* to deliver intelligent recipe recommendations. Built with *Flask, **Pathway, and **Trae*, this project demonstrates how multimodal AI can transform food discovery and recipe generation for real-world applications.

![SnapChef](rag_service_pathway/static/images/demo_ui.png)

## 🌟 Features

- *Image-to-Recipe*: Upload a food photo → CNN model detects dish/ingredients.
- *Smart Recipe Retrieval*: Pathway-powered RAG fetches recipes from trusted Indian food websites (like Hebbar’s Kitchen).
- *Live Web Scraping*: Automatically fetches the latest recipes from sitemaps instead of relying only on static seeds.
- *Modern UI*: Responsive frontend with TailwindCSS + dark/light themes.
- *Search Functionality: Text-based queries (e.g., *“Paneer Butter Masala”) return recipes with links and snippets.
- *Demo Ready*: Lightweight index + real-time retriever for hackathon showcase.

## 🛠 Tech Stack

- *Backend*: Python 3.8, Flask
- *Retrieval Engine*: [Pathway](https://pathway.com) (streaming data + RAG pipeline)
- *Embedding & Indexing*: FAISS + SentenceTransformers
- *Live Recipe Fetching*: Custom scrapers + Hebbar’s Kitchen sitemap integration
- *Frontend*: HTML5, TailwindCSS, JavaScript
- *Deployment Ready*: Dockerized setup with docker-compose

## 📋 Prerequisites

Before running the app, ensure you have:

- Python 3.8 (recommended for model compatibility)
- Docker + Docker Compose (for containerized RAG service)
- Model files for the CNN-based image classifier stored in Foodimg2Ing/data/:
  1. modelbest.ckpt
  2. ingr_vocab.pkl
  3. instr_vocab.pkl

## 🚀 Getting Started

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/SnapChef.git
cd SnapChef

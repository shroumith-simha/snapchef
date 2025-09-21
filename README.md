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

# SnapToCook 🍳

An AI-powered web application that transforms food images into detailed recipes using advanced deep learning. Built with Flask and PyTorch, this project demonstrates the power of multi-modal AI in culinary applications.

![SnapToCook](Foodimg2Ing/static/images/UISS.png)

## 🌟 Features

- **Real-time Food Analysis**: Upload any food image to get instant recipe suggestions
- **AI-Powered Recognition**: Advanced CNN and Transformer models for ingredient detection
- **Detailed Recipes**: Step-by-step cooking instructions with ingredient lists
- **Modern UI**: Clean, responsive design with dark/light theme support
- **Sample Images**: Test the system with pre-loaded food images
- **Accessibility**: Keyboard navigation and screen reader support

## 🛠️ Tech Stack

- **Backend**: Python 3.8, Flask, PyTorch
- **Frontend**: HTML5, TailwindCSS, JavaScript
- **AI Models**: Custom CNN and Transformer architectures
- **Deployment**: Heroku-ready configuration

## 📋 Prerequisites

Before running the application, you need to download the following model files and place them in the `Foodimg2Ing/data/` directory:

1. **Model File**:
   - Download [Modelbest.ckpt](https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt)
   - Place in: `Foodimg2Ing/data/modelbest.ckpt`

2. **Ingredients Vocabulary**:
   - Download [ingr_vocab.pkl](https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl)
   - Place in: `Foodimg2Ing/data/ingr_vocab.pkl`

3. **Instructions Vocabulary**:
   - Download [instr_vocab.pkl](https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl)
   - Place in: `Foodimg2Ing/data/instr_vocab.pkl`

## 🚀 Getting Started

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/SnapToCook.git
cd SnapToCook
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download and place model files as described in Prerequisites

5. Run the application
```bash
python run.py
```

6. Open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
SnapToCook/
├── run.py                      # Application entry point
├── Foodimg2Ing/               # Main application package
│   ├── Templates/             # HTML templates
│   │   ├── base.html         # Base template with common elements
│   │   ├── index.html        # Home page
│   │   ├── about.html        # About page
│   │   └── layout.html       # Layout template
│   ├── static/               # Static assets
│   │   ├── css/             # Stylesheets
│   │   ├── js/              # JavaScript files
│   │   └── images/          # Image assets
│   ├── modules/             # Core model modules
│   ├── utils/               # Utility functions
│   └── data/                # Data and models directory
└── requirements.txt          # Project dependencies
```

## 🤖 How It Works

1. **Image Processing**
   - User uploads a food image
   - Image is preprocessed and normalized
   - Features are extracted using CNN

2. **Ingredient Recognition**
   - CNN features are processed by Transformer
   - Ingredients are identified and listed
   - Confidence scores are calculated

3. **Recipe Generation**
   - Identified ingredients are encoded
   - Recipe steps are generated
   - Instructions are formatted and displayed

## 🎯 Usage

1. Visit the home page
2. Upload a food image or select from samples
3. Wait for AI processing
4. View the generated recipe
5. Follow the cooking instructions

## 🔧 Configuration

The application can be configured through `args.py`:
- Model parameters
- Image processing settings
- Recipe generation options


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**P SHOUMITH SIMHA**
- LinkedIn: [P SHOUMITH SIMHA](https://www.linkedin.com/in/shroumith-simha)

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by food recognition research
- Built with open-source tools

## 📫 Contact

For questions and feedback, please reach out through:

- GitHub Issues
- LinkedIn: [P SHROUMITH SIMHA](https://www.linkedin.com/in/shroumith-simha)
- 📧 Email: [simhashroumith@gmail.com](mailto:simhashroumith@gmail.com)

## 🔄 Future Improvements

- [ ] Add more recipe variations
- [ ] Implement user accounts
- [ ] Add recipe saving feature
- [ ] Enhance mobile responsiveness
- [ ] Add more language support

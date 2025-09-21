# demo_index.py -- create a tiny demo FAISS index + meta.pkl in /app/data
import os, pickle, numpy as np, faiss

DATA_DIR="/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

meta = [
  {"title":"Paneer Butter Masala - Demo","url":"https://hebbarskitchen.com/paneer-butter-masala-recipe/","snippet":"Paneer Butter Masala recipe: cubes of paneer in a rich tomato gravy..."},
  {"title":"Dal Tadka - Demo","url":"https://www.vegrecipesofindia.com/dal-tadka/","snippet":"Dal Tadka recipe: yellow lentils tempered with cumin, garlic and ghee..."},
  {"title":"Masala Dosa - Demo","url":"https://hebbarskitchen.com/masala-dosa-recipe/","snippet":"Masala Dosa recipe: crispy dosa with potato masala filling..."}
]

dim = 384
vecs = np.random.randn(len(meta), dim).astype("float32")
vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

idx = faiss.IndexFlatIP(dim)
idx.add(vecs)

faiss.write_index(idx, os.path.join(DATA_DIR, "vectors.index"))
pickle.dump(meta, open(os.path.join(DATA_DIR, "meta.pkl"), "wb"))

print("Demo index written to", DATA_DIR)

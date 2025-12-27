from datetime import datetime
import os


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


import faiss
import numpy as np
import os

DIM = 512
NLIST = 512         
NPROBE = 16

INDEX_PATH = "./vectors/faces.index"
IDS_PATH = "./vectors/ids.npy"

# -------------------------
# LOAD OR INIT
# -------------------------
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    person_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
else:
    # START WITH FLAT INDEX (NO TRAINING NEEDED)
    index = faiss.IndexFlatIP(DIM)
    person_ids = []

# -------------------------
# ADD PERSON
# -------------------------
def add_person(embedding, person_id):
    global index, person_ids

    embedding = embedding.reshape(1, -1).astype("float32")

    # -------------------------
    # UPGRADE TO IVF IF ENOUGH DATA
    # -------------------------
    if isinstance(index, faiss.IndexFlatIP) and len(person_ids) >= NLIST:
        print("Upgrading FAISS index to IVF...")
        upgrade_to_ivf()

    index.add(embedding)
    person_ids.append(person_id)

    save_db()
    print(f"Person added: {person_id}, total={index.ntotal}")

# -------------------------
# SEARCH
# -------------------------
def search_person(embedding, threshold=0.50):
    if index.ntotal == 0:
        return None, 0.0

    embedding = embedding.reshape(1, -1).astype("float32")
    scores, idxs = index.search(embedding, 1)
    score = float(scores[0][0])

    if score < threshold:
        return None, score

    return person_ids[idxs[0][0]], score

# -------------------------
# UPDATE PERSON (REBUILD)
# -------------------------
def update_person(embedding, person_id):
    if person_id not in person_ids:
        raise ValueError("Person not found")

    idx = person_ids.index(person_id)

    embeddings = index.reconstruct_n(0, index.ntotal)
    embeddings[idx] = embedding.astype("float32")

    rebuild_index(embeddings)

# -------------------------
# UPGRADE FLAT → IVF
# -------------------------
def upgrade_to_ivf():
    global index

    embeddings = index.reconstruct_n(0, index.ntotal)

    quantizer = faiss.IndexFlatIP(DIM)
    ivf = faiss.IndexIVFFlat(
        quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT
    )
    ivf.nprobe = NPROBE

    print("Training IVF index...")
    ivf.train(embeddings)
    ivf.add(embeddings)

    index = ivf
    print("IVF index ready")

# -------------------------
# REBUILD (SAFE)
# -------------------------
def rebuild_index(embeddings):
    global index

    embeddings = embeddings.astype("float32")

    if embeddings.shape[0] < NLIST:
        index = faiss.IndexFlatIP(DIM)
        index.add(embeddings)
    else:
        quantizer = faiss.IndexFlatIP(DIM)
        index = faiss.IndexIVFFlat(
            quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = NPROBE
        index.train(embeddings)
        index.add(embeddings)

    save_db()

# -------------------------
# SAVE
# -------------------------
def save_db():
    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(person_ids, dtype=object))




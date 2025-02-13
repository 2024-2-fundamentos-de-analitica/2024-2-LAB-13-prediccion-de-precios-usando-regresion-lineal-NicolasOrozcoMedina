import gzip
import pickle

MODEL_FILENAME = "files/models/model.pkl.gz"

with gzip.open(MODEL_FILENAME, "rb") as file:
    model = pickle.load(file)

print(f"📌 Tipo de objeto cargado: {type(model)}")
print(model)  # Muestra qué contiene exactamente

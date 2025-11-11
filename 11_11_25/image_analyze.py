import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_name = 'castle.png'
caminho_arquivo = f"11_11_25/{file_name}" 

image_bgr = cv2.imread(caminho_arquivo)

if image_bgr is None:
    print(f"Erro: Não foi possível carregar a imagem em {caminho_arquivo}. Verifique o caminho.")
    exit()

pixels_data = image_bgr.reshape(-1, 3).astype(np.float32)

inertia_errors = []
k_range = range(1, 16) 

print("--- Executando K-Means para o Método do Cotovelo (BGR) ---")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels_data)
    inertia_errors.append(kmeans.inertia_)
    
    if k % 5 == 0 or k == 1 or k == 15:
        print(f"K={k}: Inércia = {kmeans.inertia_:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_errors, marker='o', linestyle='--')
plt.title('Método do Cotovelo para Determinação de K (Quantização BGR)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia (Erro)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

k_optimal = 7 

kmeans_optimal = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
labels = kmeans_optimal.fit_predict(pixels_data)

new_colors = kmeans_optimal.cluster_centers_

print(f"\n--- Novas Cores Finais (Centróides BGR) para K={k_optimal} ---")
print(np.round(new_colors).astype(int))

quantized_data = new_colors[labels]

quantized_bgr_image = quantized_data.reshape(image_bgr.shape)
quantized_bgr_image = np.clip(quantized_bgr_image, 0, 255).astype(np.uint8)

final_name = f"quantized_full_color_k{k_optimal}_{file_name}"
cv2.imwrite(f"11_11_25/{final_name}", quantized_bgr_image)

print("\n--- Imagem Final Gerada ---")
print(f"Imagem quantizada com {k_optimal} cores únicas (BGR) salva como: {final_name}")
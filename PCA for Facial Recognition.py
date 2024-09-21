import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (100, 100))
    img_flattened = img_resized.flatten()
    return img_flattened


def load_celeb_faces(dataset_path, num_faces=10):
    celeb_faces = []
    count = 0
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(dataset_path, filename)
            celeb_faces.append(preprocess_image(img_path))
            count += 1
            if count >= num_faces:
                break
    return celeb_faces


def perform_pca(data_matrix, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data_matrix)
    return pca

def reconstruct_faces(pca, data_matrix):
    reconstructed_faces = pca.inverse_transform(pca.transform(data_matrix))
    return reconstructed_faces

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def find_closest_celeb_face(pca, your_face, celeb_faces):
    your_face_pca = pca.transform(your_face.reshape(1, -1))
    distances = [euclidean_distance(pca.transform(celeb_face.reshape(1, -1)), your_face_pca) for celeb_face in celeb_faces]
    closest_index = np.argmin(distances)
    return closest_index

def visualize_reconstructed_faces(original_face, reconstructed_faces, n_components_list):
    num_plots = len(n_components_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    for i, n_components in enumerate(n_components_list):
        reconstructed_face = reconstructed_faces[i].reshape(100, 100)
        axes[i].imshow(reconstructed_face, cmap='gray')
        axes[i].set_title(f'n_components={n_components}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

your_face_path = "/Users/b/Desktop/IMG_1477 2.jpeg"
celeb_faces_folder = "/Users/b/Downloads/archive-2/img_align_celeba/img_align_celeba"

your_face = preprocess_image(your_face_path)

celeb_faces = load_celeb_faces(celeb_faces_folder, num_faces=250)

celeb_faces_matrix = np.array(celeb_faces)
your_face_matrix = np.array([your_face])

n_components = 50
pca = perform_pca(celeb_faces_matrix, n_components)

closest_index = find_closest_celeb_face(pca, your_face_matrix, celeb_faces)

n_components_list = [20, 30, 40, 50]
reconstructed_faces = []
for n_components in n_components_list:
    pca = perform_pca(celeb_faces_matrix, n_components)
    reconstructed_face = reconstruct_faces(pca, your_face_matrix)
    reconstructed_faces.append(reconstructed_face)


plt.imshow(your_face.reshape(100, 100), cmap='gray')
plt.title("Original Face")
plt.axis('off')
plt.show()


plt.imshow(celeb_faces[closest_index].reshape(100, 100), cmap='gray')
plt.title("Closest Celebrity Face")
plt.axis('off')
plt.show()


visualize_reconstructed_faces(your_face, reconstructed_faces, n_components_list)
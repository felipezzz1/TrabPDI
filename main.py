import cv2
import numpy as np

def preprocessamento(imagem):
    img = cv2.imread(imagem)

    cv2.imshow('Imagem original', img)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # "Non-Local Means Denoising" para redução de ruído, deixando mais facil pra remover o fundo
    reducaoimagem = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    cv2.imshow("Imagem com reducao de ruidos e em escalas de cinza", reducaoimagem)
    
    return reducaoimagem


def mahalanobis(point, mean, covariancia):
    dif = point - mean
    return np.sqrt(np.dot(np.dot(dif.T, np.linalg.inv(covariancia)), dif))


def otsu_threshold_with_mahalanobis(image):
    # Calcular a distância de Mahalanobis da imagem para um modelo representando os pixels de interesse (FG)
    fg_mean = np.array([150.0])  # Valor médio da intensidade da imagem (mudr conforme imagem)
    fg_covariancia = np.array([[450.0]])  # Valor de covariância (mudar copnforme a foto)

    # Aplicar a distância de Mahalanobis em cada pixel da imagem
    rows, cols = image.shape
    for y in range(rows):
        for x in range(cols):
            pixel_value = np.array([image[y, x]])
            dist = mahalanobis(pixel_value, fg_mean, fg_covariancia)

            # Mudar isso conforme a imagem(geralmente entre 1 e 11)
            threshold = 4
            if dist < threshold:
                image[y, x] = 255  # Define como branco (possivel FG(geralmente é definitivo mesmo))
            else:
                image[y, x] = 0    # Define como preto (BG)

    # Limiarização de Otsu para segmentar primeiro e segundo plano
    _, imagem_binaria = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return imagem_binaria


if __name__ == '__main__':
    img3 = 'pessoa1.jpeg'

    semruido = preprocessamento(img3)
    resultadoOtsu = otsu_threshold_with_mahalanobis(semruido)

    cv2.imshow('Imagem processada', resultadoOtsu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

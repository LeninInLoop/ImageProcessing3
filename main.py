import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_gaussian_noise(shape: tuple[int,int], std: int) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("Number of samples should be a 1D array with length 2.")
    return std * np.random.randn(*shape)

def load_image_as_array(image_path: str) -> np.ndarray:
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
    except Exception as e:
        error_msg = f""" ↓\n
{'*' * 50 + "\n"}
Error loading image: {e}
{"\n"+'*' * 50}
""".strip()
        raise Exception(error_msg)
    return image_array

def save_image_array_as_tiff(image_array: np.ndarray, image_name: str = "image.tiff") -> None:
    image = Image.fromarray(image_array)
    image.save(image_name)
    return print(f"Image Saved as {image_name}")

def add_noise_to_image_array(image_array: np.ndarray, noise_std_list: [], number_of_images_to_generate: []) -> None:
        for std in noise_std_list:
            for n_images in number_of_images_to_generate:
                for n in range(n_images):
                    noisy_image = image_array.astype(np.float32)

                    noise = generate_gaussian_noise(noisy_image.shape, std)
                    noisy_image += noise

                    # Ensure values stay within valid 8-bit range
                    noisy_image = np.clip(noisy_image, 0, 255)

                    if not os.path.exists(f"Images/std_{std}/n_{n_images}"):
                        os.makedirs(f"Images/std_{std}/n_{n_images}")

                    print(50*"-")
                    save_image_array_as_tiff(
                        noisy_image.astype(np.uint8),
                        image_name=f"Images/std_{std}/n_{n_images}/noisy_image_iter{n}.tiff"
                    )

def load_images_by_dir(dir_path: str) -> np.ndarray:
    valid_extension = '.tiff'

    image_files = sorted([
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if file.lower().endswith(valid_extension)
    ])

    images = []
    for file in image_files:
        try:
            img = Image.open(file)
            images.append(np.array(img))
        except Exception as e:
            print(f"Error loading image {file}: {e}")

    return np.array(images)

def average_images(images: np.ndarray) -> np.ndarray:
    avg_image = np.mean(images.astype(np.float32), axis=0)
    avg_image = np.clip(np.round(avg_image), 0, 255).astype(np.uint8)
    return avg_image

def calculate_mse(orig_image: np.ndarray, averaged_image: np.ndarray):
    if orig_image.shape != averaged_image.shape:
        raise ValueError("Input images must have the same dimensions.")

    orig_image = orig_image.astype(np.float64)
    averaged_image = averaged_image.astype(np.float64)

    diff = orig_image - averaged_image
    mse = np.mean(np.square(diff))

    return mse

def save_mse_as_text(mse_value: float, file_name: str) -> None:
    with open(str(file_name), "w") as file:
        file.write(f"Mean Squared Error: {mse_value}\n**BETWEEN ORIGINAL IMAGE AND AVERAGED IMAGE**")
    return print(f"MSE value Saved as {file_name}")

def average_images_by_dir(orig_image: np.ndarray, main_path: str, noise_std_list: [], number_of_generated_images: []) -> None:
    for noise_std in noise_std_list:
        for n in number_of_generated_images:
            dir_path = os.path.join(main_path, f"std_{noise_std}", f"n_{n}")
            print(50*"-" + f"\nProcessing directory: {dir_path}")

            dir_images = load_images_by_dir(dir_path)
            averaged_image = average_images(dir_images)

            calculated_mse = calculate_mse(orig_image, averaged_image)

            save_image_array_as_tiff(
                image_array=averaged_image,
                image_name= dir_path + "/" + "averaged_image.tiff"
            )

            print("Calculated MSE:", calculated_mse)
            save_mse_as_text(
                mse_value = calculated_mse,
                file_name = dir_path + "/" + "mse_value.txt"
            )
    return

def main():
    image_path = r"C:\Users\ADIB\Desktop\Image Processing\3\Adib_Nikjou_403114114_DIP_3\Images\Fig0226(galaxy_pair_original).tif"
    image_array = load_image_as_array(image_path)
    print(50*"-","\nOriginal Image Array:\n",image_array)

    generated_noise = generate_gaussian_noise(image_array.shape, std=1)
    print(50*"-","\nGenerated Gaussian Noise Array:\n",generated_noise)

    add_noise_to_image_array(
        image_array,
        noise_std_list=[1,10,20,30,40,50],
        number_of_images_to_generate=[5, 10, 20, 50, 100]
    )

    average_images_by_dir(
        orig_image=image_array,
        main_path=r"Images",
        noise_std_list=[1, 10, 20, 30, 40, 50],
        number_of_generated_images=[5, 10, 20, 50, 100]
    )

if __name__ == '__main__':
    main()

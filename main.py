import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_gaussian_noise(
        shape: tuple[int,int],
        std: int
) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("Number of samples should be a 1D array with length 2.")
    return std * np.random.randn(*shape)

def load_image_as_array(
        image_path: str
) -> np.ndarray:
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

def save_image_array_as_tiff(
        image_array: np.ndarray,
        image_name: str = "image.tiff"
) -> None:
    image = Image.fromarray(image_array)
    image.save(image_name)
    return print(f"Image Saved as {image_name}")

def add_noise_to_image_array(
        image_array: np.ndarray,
        noise_std_list: [],
        number_of_images_to_generate: []
) -> None:
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

def load_images_by_dir(
        dir_path: str
) -> np.ndarray:
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

def average_images(
        images: np.ndarray
) -> np.ndarray:
    avg_image = np.mean(images.astype(np.float32), axis=0)
    avg_image = np.clip(np.round(avg_image), 0, 255).astype(np.uint8)
    return avg_image

def calculate_mse(
        orig_image: np.ndarray,
        averaged_image: np.ndarray
) -> np.ndarray:
    if orig_image.shape != averaged_image.shape:
        raise ValueError("Input images must have the same dimensions.")

    orig_image = orig_image.astype(np.float64)
    averaged_image = averaged_image.astype(np.float64)

    diff = orig_image - averaged_image
    mse = np.mean(np.square(diff))

    return mse

def save_mse_as_text(
        mse_value: float,
        file_name: str
) -> None:
    with open(str(file_name), "w") as file:
        file.write(f"Mean Squared Error: {mse_value}\n**BETWEEN ORIGINAL IMAGE AND AVERAGED IMAGE**")
    return print(f"MSE value Saved as {file_name}")

def average_images_by_dir(
        orig_image: np.ndarray,
        main_path: str,
        noise_std_list: [],
        number_of_generated_images: []
) -> None:
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
                mse_value = float(calculated_mse),
                file_name = dir_path + "/" + "mse_value.txt"
            )
    return

def load_averaged_images_by_std(
        main_path: str,
        noise_std_list: [],
        number_of_generated_images: []
) -> np.ndarray:
    averaged_images = []
    for noise_std in noise_std_list:
        for n in number_of_generated_images:
            dir_path = os.path.join(main_path, f"std_{noise_std}", f"n_{n}")
            averaged_image_path = os.path.join(dir_path, "averaged_image.tiff")

            averaged_image = Image.open(averaged_image_path)
            averaged_images.append(averaged_image)
    return np.array(averaged_images)

def load_one_noisy_image_by_std(
        main_path: str,
        noise_std_list: []
) -> np.ndarray:
    images = []
    for noise_std in noise_std_list:
            dir_path = os.path.join(main_path, f"std_{noise_std}")
            sub_dirs = next(os.walk(dir_path))[1]
            if sub_dirs:
                dir_path = os.path.join(dir_path, sub_dirs[0])
                files_path = os.listdir(dir_path)
                image_path = os.path.join(dir_path, files_path[3])
            else:
                return np.empty([])

            averaged_image = Image.open(image_path)
            images.append(averaged_image)

    return np.array(images)


def plot_images_with_std(
        image_vector: np.ndarray,
        noise_std_list: []
) -> None:
    if len(image_vector.shape) != 3:
        raise ValueError("Input images must have 3 dimensions.")

    if image_vector.shape[0] != 6:
        raise ValueError("Expected 6 images, but got {}.".format(image_vector.shape[0]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        img = image_vector[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Noise std={noise_std_list[idx]}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("Images/noisy_images.tiff")
    plt.show()

def plot_average_images_with_image_number(
        image_vector: np.ndarray,
        noise_std_list: [],
        number_of_generated_images_list: []
) -> None:
    total_images = image_vector.shape[0]
    group_size = len(number_of_generated_images_list)

    if total_images % group_size != 0:
        raise ValueError("The total number of images must be a multiple of 6.")

    num_groups = total_images // group_size

    if len(noise_std_list) != num_groups:
        raise ValueError(f"noise_std_list must have {num_groups} elements, one for each group.")

    if len(number_of_generated_images_list) != group_size:
        raise ValueError(f"number_of_generated_images_list must have {group_size} elements.")


    for group_idx in range(num_groups):

        group_images = image_vector[group_idx * group_size: group_idx * group_size + group_size]
        current_noise_std = noise_std_list[group_idx]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        fig.suptitle(f"Noise std = {current_noise_std}", fontsize=16)

        for idx, ax in enumerate(axes):
            img = group_images[idx]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Averaging Images = {number_of_generated_images_list[idx]}")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"Images/averaged_images_std_{current_noise_std}.tiff")
        plt.show()


def main():
    image_path = r"C:\Users\ADIB\Desktop\Image Processing\3\Adib_Nikjou_403114114_DIP_3\Images\Fig0226(galaxy_pair_original).tif"
    image_array = load_image_as_array(image_path)
    print(50*"-","\nOriginal Image Array:\n",image_array)

    generated_noise = generate_gaussian_noise(image_array.shape, std=1)
    print(50*"-","\nGenerated Gaussian Noise Array:\n",generated_noise)

    noise_std_list = [1, 20, 40, 60, 80, 100]
    number_of_generated_images_list = [5, 10, 20, 50, 100, 150]

    add_noise_to_image_array(
        image_array,
        noise_std_list=noise_std_list,
        number_of_images_to_generate=number_of_generated_images_list
    )

    average_images_by_dir(
        orig_image=image_array,
        main_path=r"Images",
        noise_std_list=noise_std_list,
        number_of_generated_images=number_of_generated_images_list
    )

    plot_images_with_std(
        image_vector=load_one_noisy_image_by_std(
            main_path=r"Images",
            noise_std_list=noise_std_list
        ),
        noise_std_list=noise_std_list
    )

    plot_average_images_with_image_number(
        image_vector=load_averaged_images_by_std(
            main_path=r"Images",
            noise_std_list=noise_std_list,
            number_of_generated_images=number_of_generated_images_list
        ),
        noise_std_list=noise_std_list,
        number_of_generated_images_list=number_of_generated_images_list
    )

if __name__ == '__main__':
    main()

from PIL import Image
import os

__all__ = [
    "save_gif"
]


def save_gif(filename: str = "./animation",
             database: str = None,
             dataName: str = None,
             fps: int = 20,
             loop: int = 0):
    """
    Description:
    The save_gif function is a helper utility designed to create an animated GIF from a series of image files.
    The images are read from a specified directory and filtered based on a given chart name. The resulting GIF is saved
    to a specified output file.

    Parameters:
    - filename (str): The path where the output GIF will be saved. Default is ./animation.
    - database (str): The directory containing the image files. This should be a valid directory path.
    - dataName (str): A substring to filter the image files in the origin directory. Only files containing this
      substring in their names will be included in the GIF.
    - fps (int): Frames per second for the GIF. This determines the speed of the animation. Default is 20.
    - loop (int): Number of times the GIF will loop. Default is 0, which means the GIF will loop indefinitely.
    """
    database += '/' if database[-1] != "/" else ''
    filesInOrigin = os.listdir(database)
    filesForAnimation = []
    for file in filesInOrigin:
        if dataName in file:
            filesForAnimation.append(file)
    print(f"Number of files found: {len(filesForAnimation)}")
    imgs = [Image.open(database+file) for file in filesForAnimation]
    imgs[0].save(fp=filename, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
    print(f"Animation file \"{filename}\" was created with {fps} fps")


if __name__ == "__main__":
    save_gif(filename="output.gif",
             database="/home/mario/Dokumente/cluster_hbrs_home/neuraloperator/tgv3D/0069/208050/plots",
             dataName="dissipation",
             fps=20)

import cv2
import matplotlib.pyplot as plt

content_path = "test_pics/c.jpeg"
style_path = "test_pics/d.jpeg"
figure_content = cv2.imread(content_path)
figure_style = cv2.imread(style_path)

def show_test():
    fig = plt.figure(figsize=(8,8))
    figures = [figure_content, figure_style]
    columns = 2
    rows = 1
    for i in range(1, columns* rows+1):
        img = cv2.cvtColor(figures[i-1], cv2.COLOR_BGR2BGRA)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
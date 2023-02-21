import math
import time

if __name__ == "__main__":
    with open("test_file.txt", "w", encoding="UTF-8") as file:
        start_time = time.time()
        operations = 0
        for i in range(1000):
            for j in range(1, 1000, 1):
                for k in range(1000):
                    x = (i+k)/j
                    operations += 1
        file.write("Did %d operations. \nTook %ss" % (operations, (time.time() - start_time)))
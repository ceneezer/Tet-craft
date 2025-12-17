import matplotlib.pyplot as plt
import numpy as np
base = 2500

def generate_hexadecimal_fractal(base):
    # (Same function as before)
    size = base - 1
    matrix = []
    for i in range(0 - size, 2 * size + 1):
        row = []
        out=[]
        for j in range(0 - size, 2 * size + 1):
            product = abs(i * j) % size
            if product == 0:
                product=size
            if i*j==0:
                product=0
            row.append(product)  # Store numerical values for easier mapping
            out.append(str(product)+" ")
        matrix.append(row)
        print("".join(out))
    return matrix

for i in range(base,5000,100):
    hex_fractal = generate_hexadecimal_fractal(i)
    hex_fractal_np = np.array(hex_fractal)  # Convert to NumPy array

    size=i-1
    plot_extent=(-size, 2*size, 2*size, -size)

    plt.imshow(hex_fractal_np, cmap='viridis', interpolation='nearest', extent=plot_extent)
    l="Base "+str(i)+" Digit Value"
    plt.colorbar(label=l)  # Add a colorbar
    t="Base "+str(i)+" Fractal of ceneeze"
    plt.title(t)
    plt.show()

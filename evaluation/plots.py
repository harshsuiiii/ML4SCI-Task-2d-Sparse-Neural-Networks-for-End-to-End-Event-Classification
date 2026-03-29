import matplotlib.pyplot as plt

def plot_flops_vs_error(flops, error):
    plt.figure()
    plt.plot(flops, error, marker='o')
    plt.xlabel("FLOPS")
    plt.ylabel("Error")
    plt.title("FLOPS vs Error")
    plt.grid()
    plt.show()
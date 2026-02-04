def show_graph(data_logger, labels, title="Genetic algorithm"):
    import matplotlib.pyplot as plt

    color = ['blue', 'orange', 'green', 'red', 'purple',
             'brown', 'pink', 'gray', 'olive', 'cyan']

    for (x, y), (label, flag), c in zip(data_logger.data, labels, color):
        if flag:
            plt.plot(x, y, label=label, color=f'tab:{c}')

    plt.grid(True)
    plt.xlabel("generation")
    plt.ylabel("min fitness")
    plt.title(title)
    plt.legend()

    plt.show()



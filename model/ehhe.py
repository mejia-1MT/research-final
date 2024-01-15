import tkinter as tk
from tkinter import simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def get_user_values():
    x_values = simpledialog.askstring("Input", "Enter x values separated by commas:")
    y_values = simpledialog.askstring("Input", "Enter y values separated by commas:")
    
    # Convert comma-separated values to lists
    x = list(map(float, x_values.split(',')))
    y = list(map(float, y_values.split(',')))
    
    return x, y

def plot_graph():
    # Get user values for x and y
    x, y = get_user_values()

    # Create a figure and axis
    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(1, 1, 1)

    # Plot data on the axis
    plot.plot(x, y)

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Matplotlib Graph in Tkinter")

    # Embed the matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Run the Tkinter main loop
    root.mainloop()

# Call the function to plot the graph
plot_graph()

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Scrollbar, messagebox

def display_prices_graph(prices_as_floats, initial_price, dynamic_daily_revenues, static_daily_revenues, d_demand_list, predicted_price_31, predicted_demand_31, total_revenue_dynamic, total_revenue_static):
    days = np.arange(1, len(prices_as_floats) + 1)
    fig = Figure(figsize=(6, 6))

    # Subplot 1 for Prices - Dynamic vs Static
    ax1 = fig.add_subplot(211)
    ax1.plot(days, prices_as_floats, marker='o', color='#0000FF', label='Dynamic Pricing')
    ax1.plot(days, [initial_price] * len(prices_as_floats), marker='o', color='#008000', label='Static Pricing')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Comparison between Dynamic and Static Pricing')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2 for Revenues - Dynamic vs Static
    ax2 = fig.add_subplot(212)
    ax2.plot(days, dynamic_daily_revenues, marker='o', color='#87CEEB', label='Dynamic Revenue')
    ax2.plot(days, static_daily_revenues, marker='o', color='#90EE90', label='Static Revenue')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Revenue')
    ax2.set_title('Revenue Comparison between Dynamic and Static Pricing')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()

    root = tk.Tk()
    root.title('Pricing Comparison')
    root.geometry('1000x600')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    text_frame = tk.Frame(root, width=500, height=750, bg="white")  # Left space for text
    text_frame.pack(side=tk.LEFT, fill=tk.BOTH, anchor=tk.NW)

    # Create a Text widget and Scrollbar
    text_widget = tk.Text(text_frame, wrap="none", font=("Times New Roman", 12))
    scrollbar = Scrollbar(text_frame, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.pack(side=tk.TOP, fill=tk.X, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    dynamic_text = "Dynamic Pricing Breakdown\n"
    for i in range(len(dynamic_daily_revenues)):
        dynamic_text += f"Day {i+1}\tPrice {prices_as_floats[i]}\tDemand Value {d_demand_list[i]}\tRevenue {dynamic_daily_revenues[i]}\n"

    static_text = "Static Pricing Breakdown\n"
    for i in range(len(static_daily_revenues)):
        static_text += f"Day {i+1}\tPrice {initial_price}\tDemand Value {d_demand_list[i]}\tRevenue {static_daily_revenues[i]}\n"

    # Append the revenue print statements to the static text
    static_text += f"\n\nTotal Revenue for Dynamic Pricing: {total_revenue_dynamic:.2f}\n"
    static_text += f"Total Revenue for Static Pricing: {total_revenue_static:.2f}\n"

    # Insert the text into the Text widget
    text_widget.insert(tk.END, dynamic_text + "\n" + static_text)

    # Function to show the dialog box
    def show_prediction():
        messagebox.showinfo("Prediction for Day 31", f"Predicted Demand: {predicted_demand_31:.2f}\nPredicted Price: {predicted_price_31:.2f}")

    # Button to trigger the dialog box
    prediction_button = tk.Button(root, text="Show Prediction for Day 31", command=show_prediction)
    prediction_button.pack()
    prediction_button.place(relx=0.125, rely=.95, anchor=tk.SW)
    root.mainloop()